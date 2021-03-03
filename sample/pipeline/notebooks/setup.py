% %

import pandas as pd
from sample.util import get_dates_datetime, get_dirname, get_plate_number, get_postion_number

import ast
from sample.plotutil import plot_t_tp1
from scipy import sparse
from datetime import datetime
from sample.pipeline.functions.node_id import orient
import pickle
import scipy.io as sio
from pymatreader import read_mat
from matplotlib import colors
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi
from skimage import filters
from random import choice
import scipy.sparse
import os
from sample.pipeline.functions.extract_graph import from_sparse_to_graph, generate_nx_graph, sparse_to_doc
from skimage.feature import hessian_matrix_det
from sample.pipeline.functions.experiment_class_surf import Experiment

# %% md

** *Chose
plate
number and directory
of
interest
where
the
folders
with images are ** *

# %%

plate_number = 412
plate = get_postion_number(plate_number)
print(plate)
from sample.paths.directory import run_parallel, find_state

directory = "/projects/0/einf914/data/"
listdir = os.listdir(directory)
list_dir_interest = [name for name in listdir if name.split('_')[-1] == f'Plate{0 if plate < 10 else ""}{plate}']
dates_datetime = get_dates_datetime(directory, plate)
len(list_dir_interest)

# %%

get_dirname(dates_datetime[1], plate)

# %%

plate_number = get_plate_number(plate, dates_datetime[0])
print(0, plate_number)
for i in range(len(list_dir_interest)):
    new_plate_number = get_plate_number(plate, dates_datetime[i])
    if plate_number != new_plate_number:
        plate_number = new_plate_number
        print(i, plate_number)

# %% md

** *Chose
timestep
to
begin
with and folder to end with ** *

# %%

begin = 1
end = 165
print('begin =', dates_datetime[begin], '\n  end =', dates_datetime[end])

# %%

find_state(plate, begin, end, directory)

# %% md

** *Skeletonization ** *
- Only
4
skeletonization
processes
can
be
run in parallel
on
one
node

# %%

num_parallel = 4
time = '3:00:00'
low = 30
high = 80
extend = 30
args = [plate, low, high, extend, directory]
run_parallel('extract_skel_surf.py', args, begin, end, num_parallel, time, 'skeletonization')

# %% md

** *Compress
raw
image ** *

# %%

num_parallel = 4
time = '2:00'
args = [plate, directory]
run_parallel('compress_image.py', args, begin, end, num_parallel, time, 'compress')

# %% md

** *Check
Skeletonization ** *
- The
first
cell
loads
the
skeleton and the
compressed
raw
image
- The
second
cell
shows
the
overlap
of
the
skeleton and the
raw
image

# %%

start = 0
finish = start + 2
dates_datetime = get_dates_datetime(directory, plate)
dates_datetime_chosen = dates_datetime[start:finish + 1]
dates = dates_datetime_chosen
skels = []
ims = []
kernel = np.ones((5, 5), np.uint8)
itera = 1
for date in dates:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    skel_info = read_mat(path_snap + '/Analysis/skeleton_compressed.mat')
    skel = skel_info['skeleton']
    skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
    im = read_mat(path_snap + '/Analysis/raw_image.mat')['raw']
    ims.append(im)

# %% md

- Chose
start and finish
to
display
the
overlap
of
the
skeleton and the
raw
image: no
more
than
10
at
a
time or jupyter
will
crash.
- For
display
purpose, everything is compressed
so
connectivity
may
be
lost
- colors
are
just
a
feature
of
compression

# %%

plt.close('all')
start = 0
finish = start + 1
for i in range(start, finish):
    plot_t_tp1([], [], None, None, skels[i], ims[i])

# %% md

** *Check
specific
image ** *
- If
something
wrong is noticed in one
of
the
skeletons
one
can
chose
to
look
closer
at
one
of
the
images and the
skeletonization
process
- chose ** * i ** * equal
to
the
timestep
where
something
wrong
has
been
noticed

# %%

# chose i equal to the timestep where something wrong has been noticed
i = 122

dates_datetime = get_dates_datetime(directory, plate)
dates = dates_datetime

date = dates[i]
directory_name = get_dirname(date, plate)
path_snap = directory + directory_name
path_tile = path_snap + '/Img/TileConfiguration.txt.registered'
try:
    tileconfig = pd.read_table(path_tile, sep=';', skiprows=4, header=None, converters={2: ast.literal_eval},
                               skipinitialspace=True)
except:
    print('error_name')
    path_tile = path_snap + '/Img/TileConfiguration.registered.txt'
    tileconfig = pd.read_table(path_tile, sep=';', skiprows=4, header=None, converters={2: ast.literal_eval},
                               skipinitialspace=True)
xs = [c[0] for c in tileconfig[2]]
ys = [c[1] for c in tileconfig[2]]
dim = (int(np.max(ys) - np.min(ys)) + 4096, int(np.max(xs) - np.min(xs)) + 4096)
ims = []
for name in tileconfig[0]:
    imname = '/Img/' + name.split('/')[-1]
    ims.append(imageio.imread(directory + directory_name + imname))

# %%

tileconfig

# %% md

- Chose
a
x, y
position
where
you
want
to
see
how
the
skeletonization
process
went(x is the
scale
on
the
left
on
the
images and y is the
bottom
scale)
- You
can
chose
to
display
different
part
of
the
filter
par
commenting / uncommenting

# %%

plt.close('all')

# chose a spot where to look closer at
linex = 4200
liney = 1500

shape = (3000, 4096)
linex *= 5
liney *= 5
for index, im in enumerate(ims):
    boundaries = int(tileconfig[2][index][0] - np.min(xs)), int(tileconfig[2][index][1] - np.min(ys))
    if boundaries[1] <= linex < boundaries[1] + shape[0] and boundaries[0] <= liney < boundaries[0] + shape[1]:
        print(index)
        im_cropped = im
        im_blurred = cv2.blur(im_cropped, (200, 200))
        im_back_rem = (im_cropped + 1) / (im_blurred + 1) * 120
        im_back_rem[im_back_rem >= 130] = 130
        # # im_back_rem = im_cropped*1.0
        # # # im_back_rem = cv2.normalize(im_back_rem, None, 0, 255, cv2.NORM_MINMAX)
        frangised = frangi(im_back_rem, sigmas=range(1, 20, 4)) * 255
        # # frangised = cv2.normalize(frangised, None, 0, 255, cv2.NORM_MINMAX)
        hessian = hessian_matrix_det(im_back_rem, sigma=20)
        blur_hessian = cv2.blur(abs(hessian), (20, 20))
        #     transformed = (frangised+cv2.normalize(blur_hessian, None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
        #     transformed = (frangised+cv2.normalize(abs(hessian), None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
        transformed = (frangised - im_back_rem + 120) * (im_blurred >= 35)
        low = 40
        high = 80
        lowt = (transformed > low).astype(int)
        hight = (transformed > high).astype(int)
        hyst = filters.apply_hysteresis_threshold(transformed, low, high)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(hyst.astype(np.uint8) * 255, kernel, iterations=1)
        for i in range(3):
            dilation = cv2.erode(dilation.astype(np.uint8) * 255, kernel, iterations=1)
            dilation = cv2.dilate(dilation.astype(np.uint8) * 255, kernel, iterations=1)
        dilated = dilation > 0

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilated.astype(np.uint8),
                                                                                   connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1];
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 4000

        # your answer image
        img2 = np.zeros((dilated.shape))
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img2[output == i + 1] = 1
        skeletonized = cv2.ximgproc.thinning(np.array(255 * img2, dtype=np.uint8))
        nx_g = generate_nx_graph(from_sparse_to_graph(scipy.sparse.dok_matrix(skeletonized)))
        g, pos = nx_g
        tips = [node for node in g.nodes if g.degree(node) == 1]
        dilated_bis = np.copy(img2)
        for tip in tips:
            branch = np.array(orient(g.get_edge_data(*list(g.edges(tip))[0])['pixel_list'], pos[tip]))
            orientation = branch[0] - branch[min(branch.shape[0] - 1, 20)]
            orientation = orientation / (np.linalg.norm(orientation))
            window = 20
            x, y = pos[tip][0], pos[tip][1]
            if x - window >= 0 and x + window < dilated.shape[0] and y - window >= 0 and y + window < dilated.shape[1]:
                shape_tip = dilated[x - window:x + window, y - window:y + window]
                dist = 20
                for i in range(dist):
                    pixel = (pos[tip] + orientation * i).astype(int)
                    xp, yp = pixel[0], pixel[1]
                    if xp - window >= 0 and xp + window < dilated.shape[0] and yp - window >= 0 and yp + window < \
                            dilated.shape[1]:
                        dilated_bis[xp - window:xp + window, yp - window:yp + window] += shape_tip
        dilation = cv2.dilate(dilated_bis.astype(np.uint8) * 255, kernel, iterations=1)
        for i in range(3):
            dilation = cv2.erode(dilation.astype(np.uint8) * 255, kernel, iterations=1)
            dilation = cv2.dilate(dilation.astype(np.uint8) * 255, kernel, iterations=1)
        skeletonized = cv2.ximgproc.thinning(np.array(255 * (dilation > 0), dtype=np.uint8))
        print('Raw image')
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111)
        ax.imshow(im)
        print('Raw image with background removed')
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111)
        ax.imshow(im_back_rem)
        print('frangised image')
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111)
        ax.imshow(frangised)
        print('final transformed image')
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111)
        ax.imshow(transformed)
        print('threhsolded image')
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111)
        ax.imshow(dilated > 0)
        print('threhsolded image')
        fig = plt.figure(figsize=(10, 9))

        ax = fig.add_subplot(111)
        ax.imshow(img2 > 0)
        print('extended tips')
        fig = plt.figure(figsize=(10, 9))
        ax = fig.add_subplot(111)
        ax.imshow(dilated_bis)
        print('final_skel')
        #         fig=plt.figure(figsize=(10,9))
        #         ax = fig.add_subplot(111)
        #         ax.imshow(cv2.normalize(abs(hessian), None, 0, 255, cv2.NORM_MINMAX)-255*dilated)
        plot_t_tp1([], [], None, None, skeletonized, im_back_rem)

# %% md

** *Mask
baits and border
of
the
petri
dish ** *

# %%

num_parallel = 4
time = '5:00'
thresh = 60
args = [plate, thresh, directory]
run_parallel('mask_skel.py', args, begin, end, num_parallel, time, 'mask')

# %% md

** *Check
Masking ** *

# %%

start = 20
finish = start + 2
dates_datetime = get_dates_datetime(directory, plate)
dates_datetime_chosen = dates_datetime[start:finish + 1]
dates = dates_datetime_chosen
skels = []
ims = []
masks = []
kernel = np.ones((5, 5), np.uint8)
itera = 1
for date in dates:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    skel_info = read_mat(path_snap + '/Analysis/skeleton_masked_compressed.mat')
    skel = skel_info['skeleton']
    skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
    mask_info = read_mat(path_snap + '/Analysis/mask.mat')
    mask = mask_info['mask']
    masks.append(mask)
    im = read_mat(path_snap + '/Analysis/raw_image.mat')['raw']
    ims.append(im)

# %%

plt.close('all')
start = 0
finish = start + 2
for i in range(start, finish):
    plot_t_tp1([], [], None, None, skels[i], ims[i])

# %% md

** *Prune
Graph ** *

# %%

num_parallel = 20
time = '50:00'
threshold = 1
args = [plate, threshold, directory]
run_parallel('prune_skel.py', args, begin, end, num_parallel, time, 'prune_graph')

# %% md

** *Check
Pruned
Graphs ** *

# %%

start = 5
finish = start + 2
dates_datetime = get_dates_datetime(directory, plate)
dates_datetime_chosen = dates_datetime[start:finish + 1]
dates = dates_datetime_chosen
skels = []
ims = []
kernel = np.ones((5, 5), np.uint8)
itera = 1
for date in dates:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    skel_info = read_mat(path_snap + '/Analysis/skeleton_pruned_compressed.mat')
    skel = skel_info['skeleton']
    skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
    im = read_mat(path_snap + '/Analysis/raw_image.mat')['raw']
    ims.append(im)

# %%

plt.close('all')
start = 0
finish = start + 2
for i in range(start, finish):
    plot_t_tp1([], [], None, None, skels[i], ims[i])

# %%

plt.close('all')
kernel = np.ones((5, 5), np.uint8)
for i in range(len(compressed) - 1):
    plot_t_tp1([], [], None, None, cv2.dilate(compressed[i].astype(np.uint8), kernel, iterations=2),
               cv2.dilate(compressed[i + 1].astype(np.uint8), kernel, iterations=2))

# %% md

** *Realign ** *

# %%

num_parallel = 16
time = '1:00:00'
thresh = 10000
args = [plate, thresh, directory]
run_parallel('final_alignment.py', args, begin, end, num_parallel, time, 'realign')

# %% md

** *Check
Alignment ** *

# %%

plt.close('all')
start = 151
finish = start + 2
dates_datetime = get_dates_datetime(directory, plate)
dates_datetime_chosen = dates_datetime[start:finish + 1]
dates = dates_datetime_chosen
dilateds = []
skels = []
skel_docs = []
Rs = []
ts = []
for date in dates[1:]:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    skel_info = read_mat(path_snap + '/Analysis/skeleton.mat')
    skel = skel_info['skeleton']
    skels.append(skel)
    skel_doc = sparse_to_doc(skel)
    skel_docs.append(skel_doc)
    transform = sio.loadmat(path_snap + '/Analysis/transform.mat')
    R, t = transform['R'], transform['t']
    Rs.append(R)
    ts.append(t)
# start = 0
# for j in range(start,start + 5):
#     print(dates[j],j+begin)
#     skeleton1,skeleton2 = skel_docs[j],skel_docs[j+1]
#     R,t = Rs[j],ts[j]
#     skelet_pos = np.array(list(skeleton1.keys()))
#     samples = np.random.choice(skelet_pos.shape[0],20000)
#     X = np.transpose(skelet_pos[samples,:])
#     skelet_pos = np.array(list(skeleton2.keys()))
#     samples = np.random.choice(skelet_pos.shape[0],20000)
#     Y = np.transpose(skelet_pos[samples,:])
#     fig=plt.figure(figsize=(10,9))
#     ax = fig.add_subplot(111)
#     Yrep=np.transpose(np.transpose(np.dot(R,X))+t)
#     ax.scatter(np.transpose(Yrep)[:,0],np.transpose(Yrep)[:,1])
#     ax.scatter(np.transpose(Y)[:,0],np.transpose(Y)[:,1])

# %% md

** *Create
realigned
Skeleton ** *

# %%

num_parallel = 12
time = '1:00:00'
args = [plate, begin, end, directory]
run_parallel('realign_surf.py', args, begin, end, num_parallel, time, 'create_realign')

# %% md

** *Check
Fine
Alignment ** *

# %%

start = 5
finish = start + 10
dates_datetime = get_dates_datetime(directory, plate)
dates_datetime_chosen = dates_datetime[start:finish + 1]
dates = dates_datetime_chosen
skels = []
ims = []
kernel = np.ones((5, 5), np.uint8)
itera = 1
for date in dates:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    skel_info = read_mat(path_snap + '/Analysis/skeleton_realigned_compressed.mat')
    skel = skel_info['skeleton']
    #     skels.append(skel)
    skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
    im = read_mat(path_snap + '/Analysis/raw_image.mat')['raw']
    ims.append(im)

# %%

plt.close('all')
start = 4
finish = start + 1
for i in range(start, finish):
    plot_t_tp1([], [], None, None, skels[i], skels[i + 1])

# %%

plt.close('all')
directory = "/scratch/shared/mrozemul/Fiji.app/"
listdir = os.listdir(directory)
list_dir_interest = [name for name in listdir if name.split('_')[-1] == f'Plate{0 if plate < 10 else ""}{plate}']
ss = [name.split('_')[0] for name in list_dir_interest]
ff = [name.split('_')[1] for name in list_dir_interest]
dates_datetime = [datetime(year=int(ss[i][:4]), month=int(ss[i][4:6]), day=int(ss[i][6:8]), hour=int(ff[i][0:2]),
                           minute=int(ff[i][2:4])) for i in range(len(list_dir_interest))]
dates_datetime.sort()
begin = 0
end = 20
dates_datetime_chosen = dates_datetime[begin:end]
dates = [
    f'{0 if date.month < 10 else ""}{date.month}{0 if date.day < 10 else ""}{date.day}_{0 if date.hour < 10 else ""}{date.hour}{0 if date.minute < 10 else ""}{date.minute}'
    for date in dates_datetime_chosen]
zone = (13000, 13000 + 5000 + 3000, 20000, 20000 + 5000 + 4096)
skels_aligned = []
for i, date in enumerate(dates):
    directory_name = f'2020{dates[i]}_Plate{0 if plate < 10 else ""}{plate}'
    path_snap = '/scratch/shared/mrozemul/Fiji.app/' + directory_name
    skels_aligned.append(sio.loadmat(path_snap + '/Analysis/skeleton_realigned.mat')['skeleton'])
for i in range(11, 13):
    plot_t_tp1([], [], None, None, skels_aligned[i][zone[0]:zone[1], zone[2]:zone[3]].todense(),
               skels_aligned[i + 1][zone[0]:zone[1], zone[2]:zone[3]].todense())

# %%

plt.close('all')
zone = (6000, 13000, 12000, 22000)
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111)
ax.imshow(skels_aligned[11][zone[0]:zone[1], zone[2]:zone[3]].todense())

# %% md

** *Create
graphs ** *

# %%

num_parallel = 5
time = '30:00'
args = [plate, directory]
run_parallel('extract_nx_graph.py', args, begin, end, num_parallel, time, 'extract_nx')

# %% md

** *Extract
Width ** *

# %%

num_parallel = 16
time = '1:00:00'
args = [plate, directory]
run_parallel('extract_width.py', args, begin, end, num_parallel, time, 'extract_width')

# %% md

** *Identify
Nodes ** *

# %%

num_parallel = 1
time = '12:00:00'
args = [plate, begin, end, directory]
run_parallel('extract_nodes_surf.py', args, 0, 0, num_parallel, time, 'node_id')

# %% md

** *Check
Node
Id ** *

# %%

dates_datetime = get_dates_datetime(directory, plate)
dates_datetime_chosen = dates_datetime[begin:end + 1]
dates = dates_datetime_chosen
exp = Experiment(plate)
exp.load(dates)

# %%

plt.close('all')
t = 2
nodes = np.random.choice(exp.nx_graph[t].nodes, 100)
# exp.plot([t,t+1,t+2],[list(nodes)]*3)
exp.plot([t, t + 1, t + 2], [nodes] * 3)

# %% md

** *Hyphae
extraction ** *

# %%

num_parallel = 1
time = '2:00:00'
args = [plate, begin, end, directory]
run_parallel('hyphae_extraction.py', args, 0, 0, num_parallel, time, 'hyphae')

# %%


# %% md

** *Check
Hyphae ** *

# %%

dates_datetime = get_dates_datetime(directory, plate)
dates_datetime_chosen = dates_datetime[begin:end + 1]
dates = dates_datetime
exp = pickle.load(open(f'{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/experiment_{plate}.pick', "rb"))

# %%

hyph = choice(exp.hyphaes)
hyph.ts

# %%

plt.close('all')
hyph.end.show_source_image(hyph.ts[-1], hyph.ts[-1])

# %%

plt.close('all')
exp.plot([0, hyph.ts[-2], hyph.ts[-1]], [[hyph.root.label, hyph.end.label]] * 3)
