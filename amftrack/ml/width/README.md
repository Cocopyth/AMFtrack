# Width measurement

## Objective
Computing the width of each of the segmented edges (Edge object) from the Prince images.
The Prince images are taken with a magnification of 2. And 1 pixel = 1.725 micrometers.
Hyphas have a width typically comprised between 2.5 micrometers and 8 micrometers.

## Difficulties
- the focus of the camere vary a lot
- the edge are very thin (only several pixels of width)
- the shade and optical effects vary a lot depending on the image, position, orientation, hypha, .. For example, some hypha have a shadow to the left, to the right, or some have a white reflection next to them.

## General strategy
We take 1D slices perpendicular to the edge in several points along the edge (after skeletonization and graph analysis).
These profile slices are then used to compute the width along the edge.
We then take the median as the width for this edge.

Step 1:
For computing the width from a given slice, we use a machine learning model trained with supervised learning.
The labels are set as segments with the app `labelme` on the original Prince images in areas where the focus is good enough to have a good appreciation of the width. The width derived from the segment is the used to label all slices extracted from this edge, making the assumption that the width is constant along the edge (which seems to be verified more or less). This enables to label also slices taken in out of focus areas as often the focus vary along the edge.

Step 2: TODO
Recalibrate the learned model with groundtruth taken at a 100x magnification.

## Implementation details
### Reading the csv from tf
Chosen:
https://www.tensorflow.org/api_docs/python/tf/data/experimental/CsvDataset
Other alternatives:
https://stackoverflow.com/questions/68923942/how-to-use-tf-data-in-tensorflow-to-read-csv-files
Tutorial with several methods:
https://www.tensorflow.org/tutorials/load_data/csv

### Data augmentation
https://www.tensorflow.org/tutorials/images/data_augmentation
As we have several custom data augmentation transformation, we have to use lambda layers that can't be serialized well. As a result, we apply those layers on the dataset but don't add the layers to the model.
Lambda layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Lambda

### Training data
As the slices look a lot alike along edges, we use different edges for training, validation, and test sets.
We use several transformations to augment the data.

## Important remarks
- for now the don't use the coordinates of the edge in the picture as well as its orientation in the model. This could maybe give better performance but would hamper use of the model for plates that haven't been segmented.
- we extract more slices from long edges which are thus over represented.

## Other ideas
- use Prince to generate several images with different focus of the same region for learning
- use imagenet or another pretrained network and only train the top