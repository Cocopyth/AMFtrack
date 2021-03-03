from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from sample.util import get_dates_datetime, get_dirname
import scipy.io as sio
from pymatreader import read_mat
import numpy as np
from sample.pipeline.scripts.extract_graph import (
    sparse_to_doc,
    get_degree3_nodes,
)
import open3d as o3d
from cycpd import rigid_registration


i = int(sys.argv[-1])
plate = int(sys.argv[1])
thresh = int(sys.argv[2])
directory = str(sys.argv[3])

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates_datetime_chosen = dates_datetime[i : i + 2]
print("========")
print(f"Matching plate {plate} at dates {dates_datetime_chosen}")
print("========")
dates = dates_datetime_chosen

dilateds = []
skels = []
skel_docs = []
for date in dates:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    skel_info = read_mat(path_snap + "/Analysis/skeleton_pruned.mat")
    skel = skel_info["skeleton"]
    skels.append(skel)
    skel_doc = sparse_to_doc(skel)
    skel_docs.append(skel_doc)
isnan = True
for order in [(0,1),(1,0)]:
    skeleton1, skeleton2 = skel_docs[order[0]], skel_docs[order[1]]
    skelet_pos = np.array(list(skeleton1.keys()))
    samples = np.random.choice(skelet_pos.shape[0], len(skeleton2.keys()) // 100)
    X = np.transpose(skelet_pos[samples, :])
    skelet_pos = np.array(list(skeleton2.keys()))
    samples = np.random.choice(skelet_pos.shape[0], len(skeleton2.keys()) // 100)
    Y = np.transpose(skelet_pos[samples, :])
    reg = rigid_registration(
        **{
            "X": np.transpose(X.astype(float)),
            "Y": np.transpose(Y.astype(float)),
            "scale": False, 'tolerance' : 1e-5, 'w' : 1e-5
        }
    )
    out = reg.register()
    Rfound = reg.R[0:2, 0:2]
    tfound = np.dot(Rfound, reg.t[0:2])
    if order == (0,1):
        t_init = -tfound
        Rot_init = Rfound
    else:
        Rot_init,t_init = np.linalg.inv(Rfound), np.dot(np.linalg.inv(Rfound),tfound)
    sigma2 = reg.sigma2
    if sigma2>=thresh:
        print("========")
        print(f"Failed to match plate {plate} at dates {dates_datetime_chosen}")
        print("========")
        continue
    isnan = np.isnan(tfound[0])
    if isnan:
        continue
#     X = np.transpose(
#         np.array([pos1[node] for node in pruned1 if pruned1.degree(node) == 3])
#     )
#     Y = np.transpose(
#         np.array([pos2[node] for node in pruned2 if pruned2.degree(node) == 3])
#     )
    skeleton1, skeleton2 = skel_docs[0], skel_docs[1]
    X = np.transpose(
        np.array(get_degree3_nodes(skeleton1))
    )
    Y = np.transpose(
        np.array(get_degree3_nodes(skeleton2))
    )
    # fig=plt.figure(figsize=(10,9))
    # ax = fig.add_subplot(111)
    # ax.scatter(X[0,:],X[1,:])
    # ax.scatter(Y[0,:],Y[1,:])
#     Xex = np.transpose(np.transpose(np.dot(Rot_init, X)) + t_init)
    # fig=plt.figure(figsize=(10,9))
    # ax = fig.add_subplot(111)
    # ax.scatter(Xex[0,:],Xex[1,:])
    # ax.scatter(Y[0,:],Y[1,:])
    X = np.insert(X, 2, values=0, axis=0)
    Y = np.insert(Y, 2, values=0, axis=0)
    print(X.shape, Y.shape)
    vectorX = o3d.utility.Vector3dVector(np.transpose(X))
    vectorY = o3d.utility.Vector3dVector(np.transpose(Y))
    source = o3d.geometry.PointCloud(vectorX)
    target = o3d.geometry.PointCloud(vectorY)
    threshold = 200
    trans_init = np.asarray(
        [
            [Rot_init[0, 0], Rot_init[0, 1], 0, t_init[0]],
            [Rot_init[1, 0], Rot_init[1, 1], 0, t_init[1]],
            [0, 0, 1, 0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    reg_p2p = o3d.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
    )
    print(reg_p2p)
    Rfound = reg_p2p.transformation[0:2, 0:2]
    tfound = reg_p2p.transformation[0:2, 3]
    print(Rfound, tfound)
    X, Y = X[0:2, :], Y[0:2, :]
    Yrep = np.transpose(np.transpose(np.dot(Rfound, X)) + tfound)
    # fig=plt.figure(figsize=(10,9))
    # ax = fig.add_subplot(111)
    # ax.scatter(np.transpose(Yrep)[:,0],np.transpose(Yrep)[:,1])
    # ax.scatter(np.transpose(Y)[:,0],np.transpose(Y)[:,1])
    break

if not isnan:    
    sio.savemat(path_snap + "/Analysis/transform.mat", {"R": Rfound, "t": tfound})
else :    
    sio.savemat(path_snap + "/Analysis/transform_corrupt.mat", {"R": np.array([[1,0],[0,1]]), "t": np.array([0,0])})
