import numpy as np
from amftrack.pipeline.functions.image_processing.extract_graph import (
    generate_nx_graph_from_skeleton,
)
from scipy import sparse
from amftrack.sparse_util import dilate, zhangSuen
from scipy.optimize import minimize
from time import time
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    prune_graph,
    from_nx_to_tab,
)
import open3d as o3d


def transform_skeleton(skeleton_doc, Rot, trans):
    transformed_skeleton = {}
    transformed_keys = np.round(
        np.transpose(np.dot(Rot, np.transpose(np.array(list(skeleton_doc.keys())))))
        + trans
    ).astype(np.int)
    i = 0
    for pixel in list(transformed_keys):
        i += 1
        transformed_skeleton[(pixel[0], pixel[1])] = 1
    return transformed_skeleton


def realign(
    skeleton1, nx_graphB, posB, convergence_threshold, window=500, maxdist=50, save=""
):
    converged = False
    nx_graphA, posA = generate_nx_graph_from_skeleton(skeleton1)
    t0 = np.array([0, 0])
    R0 = np.identity(2)
    while not converged:
        listeA, listeB = find_common_group_nodes(
            nx_graphA, nx_graphB, posA, posB, R0, t0, maxdist=maxdist, window=window
        )
        H = np.dot(
            np.transpose(np.array(listeA) - np.mean(listeA, axis=0)),
            np.array(listeB) - np.mean(listeB, axis=0),
        )
        U, S, V = np.linalg.svd(H)
        R = np.dot(V, np.transpose(U))
        t = np.mean(listeB, axis=0) - np.dot(R, np.mean(listeA, axis=0))
        print("number_common_nodes_found :", len(listeA))
        if np.linalg.norm(t) <= convergence_threshold:
            converged = True
        R0 = np.dot(R, R0)
        t0 = t + t0
    skeleton_transformed = transform_skeleton(skeleton1, R0, t0)
    skeleton_transformed = dilate(skeleton_transformed)
    skeleton_transformed = dilate(skeleton_transformed)
    skeleton_transformed = zhangSuen(skeleton_transformed)
    if len(save) >= 0:
        from_nx_to_tab(*generate_nx_graph_from_skeleton(skeleton_transformed)).to_csv(
            save + "_raw_aligned_skeleton.csv"
        )
        np.savetxt(save + "rot.txt", R0)
        np.savetxt(save + "trans.txt", t0)
    print("R0=", R0, "t0=", t0)
    return skeleton_transformed


def find_common_group_nodes(
    Sa, Sb, degree3_nodesa, degree3_nodesb, posa, posb, R0, t0, window=500, maxdist=50
):
    common_nodes_a = []
    common_nodes_b = []
    common_centroida = []
    common_centroidb = []
    t = time()
    posarottrans = {
        key: np.round(
            np.transpose(np.dot(R0, np.transpose(np.array(posa[key]))) + t0)
        ).astype(np.int)
        for key in degree3_nodesa
    }
    #     print("rotating translating",time()-t)
    for node in degree3_nodesa:
        t = time()
        posanchor = posarottrans[node]
        potential_surroundinga = Sa[
            posanchor[0] - 2 * window : posanchor[0] + 2 * window,
            posanchor[1] - 2 * window : posanchor[1] + 2 * window,
        ]
        potential_surroundingb = Sb[
            posanchor[0] - 2 * window : posanchor[0] + 2 * window,
            posanchor[1] - 2 * window : posanchor[1] + 2 * window,
        ]
        #         print("candidates",len(potential_surroundinga.data))
        #         print("finding_potential_surrounding",time()-t)
        t = time()
        surrounding_nodesa = [
            node
            for node in potential_surroundinga.data
            if (
                posanchor[0] - window
                < posarottrans[int(node)][0]
                < posanchor[0] + window
                and posanchor[1] - window
                < posarottrans[int(node)][1]
                < posanchor[1] + window
            )
        ]
        surrounding_nodesb = [
            node
            for node in potential_surroundingb.data
            if (
                posanchor[0] - window < posb[int(node)][0] < posanchor[0] + window
                and posanchor[1] - window < posb[int(node)][1] < posanchor[1] + window
            )
        ]
        #         print("finding_surrounding",time()-t)
        t = time()
        if len(surrounding_nodesa) == len(surrounding_nodesb):
            possurroundinga = [posarottrans[node] for node in surrounding_nodesa]
            possurroundingb = [posb[node] for node in surrounding_nodesb]
            centroida = np.mean(possurroundinga, axis=0)
            centroidb = np.mean(possurroundingb, axis=0)
            if np.linalg.norm(centroida - centroidb) <= maxdist:
                common_centroida.append(centroida)
                common_centroidb.append(centroidb)
    return (common_centroida, common_centroidb)


def realign2(
    skeleton1, skeleton2, convergence_threshold, window=500, maxdist=50, save=""
):
    converged = False
    tim = time()
    nx_graphA, posA = generate_nx_graph_from_skeleton(skeleton1)
    nx_graphB, posB = generate_nx_graph_from_skeleton(skeleton2)
    print("generate_nx_graph_from_skeleton, t=", tim - time())
    tim = time()
    t0 = np.array([0, 0])
    R0 = np.identity(2)
    degree3_nodesa = [node for node in nx_graphA if nx_graphA.degree(node) == 3]
    degree3_nodesb = [node for node in nx_graphB if nx_graphB.degree(node) == 3]
    print("lennodes=", len(degree3_nodesa))
    Sa = sparse.csr_matrix((26296, 49559))
    Sb = sparse.csr_matrix((26296, 49559))
    for node in degree3_nodesa:
        Sa[posA[node][0], posA[node][1]] = node
    for node in degree3_nodesb:
        Sb[posB[node][0], posB[node][1]] = node
    while not converged:
        listeA, listeB = find_common_group_nodes(
            Sa,
            Sb,
            degree3_nodesa,
            degree3_nodesb,
            posA,
            posB,
            R0,
            t0,
            maxdist=maxdist,
            window=window,
        )
        H = np.dot(
            np.transpose(np.array(listeA) - np.mean(listeA, axis=0)),
            np.array(listeB) - np.mean(listeB, axis=0),
        )
        U, S, V = np.linalg.svd(H)
        R = np.dot(V, np.transpose(U))
        t = np.mean(listeB, axis=0) - np.dot(R, np.mean(listeA, axis=0))
        print("number_common_nodes_found :", len(listeA))
        if np.linalg.norm(t) <= convergence_threshold:
            converged = True
        R0 = np.dot(R, R0)
        t0 = t + t0
    print("Find R and T, t=", tim - time())
    tim = time()
    skeleton_transformed = transform_skeleton(skeleton1, R0, t0)
    skeleton_transformed = dilate(skeleton_transformed)
    skeleton_transformed = zhangSuen(skeleton_transformed)
    print("transform, dilate and thin, t=", tim - time())
    tim = time()
    if len(save) >= 0:
        from_nx_to_tab(*generate_nx_graph_from_skeleton(skeleton_transformed)).to_csv(
            save + "_raw_aligned_skeleton.csv"
        )
        np.savetxt(save + "rot.txt", R0)
        np.savetxt(save + "trans.txt", t0)
    print("R0=", R0, "t0=", t0)
    return skeleton_transformed


def reconnect(skeleton):
    skeleton_transformed = dilate(skeleton)
    skeleton_transformed = zhangSuen(skeleton_transformed)
    return skeleton_transformed


def shift(skeleton1, skeleton2):
    skeleton1_dilated = dilate(dilate(skeleton1)).astype(float)
    skeleton2_dilated = dilate(dilate(skeleton2)).astype(float)

    def distance(shift):
        distance = 0
        #         print(shift)
        for pixel in skeleton1_dilated.keys():
            #             print(pixel[0]+shift[0],pixel[1]+shift[1])
            if (
                skeleton2_dilated.shape[0] > np.ceil(pixel[0] + shift[0]) >= 0
                and skeleton2_dilated.shape[1] > np.ceil(pixel[1] + shift[1]) >= 0
            ):
                shifted_pixel = (int(pixel[0] + shift[0]), int(pixel[1] + shift[1]))
                shifted_pixel_next = (
                    np.ceil(pixel[0] + shift[0]),
                    np.ceil(pixel[1] + shift[1]),
                )
                #                 print(shifted_pixel)
                prop = (
                    1
                    / 2
                    * (
                        pixel[0]
                        + shift[0]
                        - int(pixel[0] + shift[0])
                        + pixel[1]
                        + shift[1]
                        - int(pixel[1] + shift[1])
                    )
                )
                float_value = (1 - prop) * skeleton2_dilated[
                    shifted_pixel[0], shifted_pixel[1]
                ] + prop * (
                    skeleton2_dilated[shifted_pixel_next[0], shifted_pixel_next[1]]
                )
                distance += abs(skeleton1_dilated[pixel] - float_value)
            else:
                distance += 1
        #         for pixel in skeleton2_dilated.keys():
        #             if (skeleton2_dilated.shape[0]>pixel[0]-shift[0]>=0 and skeleton2_dilated.shape[1]>pixel[1]-shift[1]>=0):
        #                 shifted_pixel = (int(pixel[0]-shift[0]),int(pixel[1]-shift[1]))
        #                 distance+=abs(skeleton1_dilated[shifted_pixel[0],shifted_pixel[1]]^skeleton2_dilated[pixel])
        #             else:
        #                 distance+=1
        #         print(distance)
        return distance

    return minimize(
        distance,
        np.array([10, 10]),
        method="nelder-mead",
        options={"xatol": 1, "disp": True, "fatol": 0.1},
    )


def realign_final(skeleton1, skeleton2):
    nx_graph1, pos1 = generate_nx_graph(from_sparse_to_graph(skeleton1))
    nx_graph2, pos2 = generate_nx_graph(from_sparse_to_graph(skeleton2))
    pruned1 = prune_graph(nx_graph1)
    pruned2 = prune_graph(nx_graph2)
    X = np.transpose(
        np.array([pos1[node] for node in pruned1 if pruned1.degree(node) == 3])
    )
    Y = np.transpose(
        np.array([pos2[node] for node in pruned2 if pruned2.degree(node) == 3])
    )
    #     fig=plt.figure(figsize=(10,9))
    #     ax = fig.add_subplot(111)
    #     ax.scatter(X[0,:],X[1,:])
    #     ax.scatter(Y[0,:],Y[1,:])
    X = np.insert(X, 2, values=0, axis=0)
    Y = np.insert(Y, 2, values=0, axis=0)
    print(X.shape, Y.shape)
    vectorX = o3d.utility.Vector3dVector(np.transpose(X))
    vectorY = o3d.utility.Vector3dVector(np.transpose(Y))
    source = o3d.geometry.PointCloud(vectorX)
    target = o3d.geometry.PointCloud(vectorY)
    threshold = 200
    trans_init = np.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.0, 0.0, 0.0, 1.0]]
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
    #     fig=plt.figure(figsize=(10,9))
    #     ax = fig.add_subplot(111)
    #     ax.scatter(np.transpose(Y)[:,0],np.transpose(Y)[:,1])
    #     ax.scatter(np.transpose(Yrep)[:,0],np.transpose(Yrep)[:,1])
    skel_transformed = transform_skeleton(skeleton1, Rfound, tfound)
    skel_mat = np.zeros((26322, 49527), dtype=np.uint8)
    for pixel in skel_transformed.keys():
        skel_mat[pixel] = 1
    return (skel_mat, skel_transformed, Rfound, tfound)


def transform_skeleton_final(skeleton_doc, Rot, trans):
    skeleton_transformed = {}
    transformed_keys = np.round(
        np.transpose(np.dot(Rot, np.transpose(np.array(list(skeleton_doc.keys())))))
        + trans
    ).astype(np.int)
    for pixel in list(transformed_keys):
        if pixel[0] >= 0 and pixel[1] >= 0:
            skeleton_transformed[(pixel[0], pixel[1])] = 1
        else:
            # print("some negative pixels")
            pass
    skeleton_transformed = dilate(skeleton_transformed)
    skeleton_transformed = zhangSuen(skeleton_transformed)
    skeleton_transformed_sparse = sparse.lil_matrix((50000, 60000)).astype(np.uint8)
    for pixel in list(skeleton_transformed.keys()):
        # if pixel[0]<30000 and pixel[1]<60000:
        skeleton_transformed_sparse[(pixel[0], pixel[1])] = 1
    return skeleton_transformed_sparse
