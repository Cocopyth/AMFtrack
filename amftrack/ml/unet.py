import os

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from scipy.ndimage import label, generate_binary_structure, sum as ndi_sum
from torchvision.transforms import v2
from tqdm import tqdm

from amftrack.util.sys import path_code

torch.set_num_threads(1)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Calls constructor of parent nn.Module class
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(0.2),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    '''
    The U-Net model class
    '''

    # in_channels 3, RGB picture. out_channels 1, binary classification
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512]):
        # Calls constructor of parent nn.Module class
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of the UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature

        # Up part of the UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(in_channels=feature * 2, out_channels=feature))

        # The flat bit of the UNET
        self.flat = DoubleConv(in_channels=features[-1], out_channels=features[-1] * 2)

        # The last 64->2 2x2 conv layer of the UNET
        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=1, kernel_size=1, stride=1, padding=0,
                                    bias=False)

    def forward(self, x):
        horizontal_connections = []

        for down in self.downs:
            x = down(x)
            # Save the layer for the horizontal copy and crop
            horizontal_connections.append(x)
            x = self.pool(x)

        x = self.flat(x)
        horizontal_connections = horizontal_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            horizontal_connection = horizontal_connections[idx // 2]

            if x.shape != horizontal_connection.shape:
                x = TF.resize(x, size=horizontal_connection.shape[2:])

            concat_horizontal = torch.cat((horizontal_connection, x), dim=1)
            x = self.ups[idx + 1](concat_horizontal)

        return self.final_conv(x)

def load_checkpoint(checkpoint_name, model):
    checkpoint_path = checkpoint_name
    print(f"=> Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"The checkpoint {checkpoint_path} does not exist.")
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])
    print("=> Done!")


def transform_images(images, IMAGE_HEIGHT=512, IMAGE_WIDTH=512):
    transformed_images = []
    for idx, image in enumerate(images):
        transform = v2.Compose([
            v2.Grayscale(num_output_channels=1),
            # v2.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        t_image = transform(image).unsqueeze(0)
        transformed_images.append(t_image)
    return transformed_images


def predict_tiles(img_tiles, model):
    # Prepare images
    t_img_tiles = transform_images(images=img_tiles)
    preds = []
    with torch.no_grad():
        for _, t_img_tile in enumerate(tqdm(t_img_tiles, desc="Predicting tiles")):
            pred = torch.sigmoid(model(t_img_tile))
            pred = (pred > 0.5).float()
            pred[pred > 0] = 255.0
            preds.append(pred)

    return preds


# def predict_tiles(img_tiles, model, batch_size=32):
#     # Prepare images
#     t_img_tiles = transform_images(images=img_tiles)
#
#     # Convert to a single batch tensor if not already
#     t_img_tiles = torch.stack(t_img_tiles).squeeze(2)  # Remove dim=2
#     preds = []
#
#     # Process in batches to utilize parallelism
#     with torch.no_grad():  # Disable gradient calculation for inference
#         for i in tqdm(range(0, len(t_img_tiles), batch_size), desc="Predicting tiles"):
#             batch = t_img_tiles[i : i + batch_size]  # Get batch
#             batch_pred = torch.sigmoid(model(batch))  # Model prediction
#             batch_pred = (batch_pred > 0.5).float() * 255.0  # Binarize & scale
#             preds.append(batch_pred)
#
#     return torch.cat(preds, dim=0)  # Merge batches back into a single tensor

def cut_image(image, tile_height=512, tile_width=512, ignore_tile_size=False, overlap=0):
    '''
    Cuts an image into tiles with optional overlap
    '''
    img_width, img_height = image.size
    tiles = []
    positions = []

    height_step = tile_height - overlap
    width_step = tile_width - overlap

    # Loop over the image dimensions to cut tiles
    for h in range(0, img_height, height_step):
        for w in range(0, img_width, width_step):
            right = min(w + tile_width, img_width)
            bottom = min(h + tile_height, img_height)

            tile_border = (w, h, right, bottom)
            tile = image.crop(tile_border)
            tiles.append(tile)
            positions.append((w, h))  # Save the position of each tile

    return tiles, positions


import numpy as np


def stitch_tiles(size, tiles, positions, tile_size, overlap=0):
    '''
    Stitches binary tiles together by adding pixel values from overlapping areas,
    normalizing by the number of contributions, and then thresholding to get a binary image.

    Parameters:
      size (tuple): (img_width, img_height) of the full image.
      tiles (list): List of torch.Tensor tiles of shape (1, 1, H, W) or already squeezed.
      positions (list): List of (w, h) positions for each tile.
      tile_size (tuple): (tile_width, tile_height).
      overlap (int): Overlap between adjacent tiles.

    Returns:
      binary_img (np.ndarray): The stitched binary image (with values 0 and 1).
    '''
    img_width, img_height = size
    tile_width, tile_height = tile_size

    # Convert each tile (torch.Tensor with shape [1, 1, tile_height, tile_width])
    # to a numpy array with shape (tile_height, tile_width)
    tiles = [tile.squeeze().numpy() for tile in tiles]

    # Prepare arrays for accumulating pixel values and counting contributions.
    stitched_img = np.zeros((img_height, img_width), dtype=np.float32)  # Sum of pixel values
    weight_map = np.zeros((img_height, img_width), dtype=np.float32)  # Count of contributions

    for tile, (w, h) in zip(tiles, positions):
        # Ensure the tile is a float32 numpy array (should already be binary: 0 or 1)
        tile_array = np.array(tile, dtype=np.float32)

        # Determine the region where the tile contributes
        h_end = min(h + tile_height, img_height)
        w_end = min(w + tile_width, img_width)

        # Add pixel values from the tile to the corresponding region in the stitched image
        stitched_img[h:h_end, w:w_end] += tile_array[:(h_end - h), :(w_end - w)]
        weight_map[h:h_end, w:w_end] += 1  # Increase contribution count

    # Avoid division by zero; it shouldn't happen if every pixel is covered.
    weight_map[weight_map == 0] = 1

    # Compute the average value at each pixel.
    normalized = stitched_img / weight_map

    # Threshold the normalized image: any pixel with average >= 0.5 becomes 1, otherwise 0.
    binary_img = (normalized >= 0.5)
    binary_img = np.where(binary_img == 1, 255, 0).astype(np.uint8)

    return binary_img


'''
Makes a segmentation prediction for an image
Input: image_path (str)
Output: segmentation_prediction_array (numpy.ndarray)
'''
model_path = os.path.join(path_code[:-1], "ml", "models", "clean-512-e100eaug200.pth.tar")


def get_model(checkpoint_name=model_path, DEVICE='cpu'):
    model = UNET(in_channels=1, out_channels=1).to(device=DEVICE)
    load_checkpoint(checkpoint_name, model)
    model.eval()
    return (model)


def sort_labels(labeled_arr, n_features, min_size=0):
    # Compute the area of each component (labels 1 to n_features)
    component_sizes = ndi_sum(np.ones_like(labeled_arr), labeled_arr,
                              index=np.arange(1, n_features + 1))

    # Identify valid labels: those with a component size >= min_size
    valid_mask = component_sizes >= min_size
    valid_labels = np.arange(1, n_features + 1)[valid_mask]

    # If no component is large enough, return an array of zeros
    if valid_labels.size == 0:
        return np.zeros_like(labeled_arr, dtype=np.uint8)

    # Sort the valid labels by size (largest first)
    sorted_order = np.argsort(-component_sizes[valid_mask])
    sorted_valid_labels = valid_labels[sorted_order]

    # Create a mapping from the original label to the new label
    relabel_map = np.zeros(n_features + 1, dtype=int)
    relabel_map[sorted_valid_labels] = np.arange(1, sorted_valid_labels.size + 1)

    # Apply the mapping to obtain the relabeled array
    relabeled_arr = relabel_map[labeled_arr]
    return relabeled_arr.astype(np.uint8)


def labelise(image, features=1, min_size=0):
    if not isinstance(image, np.ndarray):
        arr = np.array(image)
    else:
        arr = image

    # This structure ensures that the labeling counts diagonal contact
    s = generate_binary_structure(2, 2)
    labeled_arr, n_features = label(arr, structure=s)

    # The labels in the array are not sorted by size, so we need a function to do that
    arr = sort_labels(labeled_arr, n_features, min_size=min_size)
    if features:
        arr = np.where(arr <= features, arr, 0)

    return arr, n_features


def make_segmentation_prediction(image, model, overlap=128):
    size = image.size
    tile_size = (512, 512)  # Tile size defined

    tiles, positions = cut_image(image=image, tile_height=512, tile_width=512, ignore_tile_size=True, overlap=overlap)
    prediction_tiles = predict_tiles(tiles, model)
    stitched_image = stitch_tiles(size=size, tiles=prediction_tiles, positions=positions, tile_size=tile_size,
                                  overlap=2 * overlap)
    return stitched_image
