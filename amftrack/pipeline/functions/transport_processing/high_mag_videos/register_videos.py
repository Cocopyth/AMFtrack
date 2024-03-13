import numpy as np
import pandas as pd
import open3d as o3d
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_all_edges,
)
from pathlib import Path
def get_segments_ends(vid_obj,shiftx,shifty,thresh_length = 0,R = np.array([[1,0],[0,1]]),t=0):
    segments = []
    for i in range(len(vid_obj.edge_objs)):
        edge = vid_obj.edge_objs[i]
        x_pos_video = edge.mean_data['xpos_network']
        y_pos_video = edge.mean_data['ypos_network']

        x_pos1 = edge.edge_infos['edge_xpos_1']*edge.space_res/1.725+x_pos_video-shiftx
        x_pos2 = edge.edge_infos['edge_xpos_2']*edge.space_res/1.725+x_pos_video-shiftx
        y_pos1 = edge.edge_infos['edge_ypos_1']*edge.space_res/1.725+y_pos_video-shifty
        y_pos2 = edge.edge_infos['edge_ypos_2']*edge.space_res/1.725+y_pos_video-shifty
        begin = transform(np.array([x_pos1,y_pos1]),R,t).tolist()
        end = transform(np.array([x_pos2,y_pos2]),R,t).tolist()
        length = np.linalg.norm(np.array([x_pos1,y_pos1])-np.array([x_pos2,y_pos2]))
        if length>thresh_length:
            # print(length)
            segments.append([begin,end])
    return(segments)

def register_rot_trans(vid_obj,exp,t,dist= 100,R = np.array([[1,0],[0,1]]),trans = 0):
    """Finds the rotation and translation to better align videos' edges on
    network edges"""
    positions = R@np.array(vid_obj.dataset[["xpos_network", "ypos_network"]])+trans
    shiftx = vid_obj.img_dim[0]*vid_obj.space_res/1.725/2
    shifty = vid_obj.img_dim[1]*vid_obj.space_res/1.725/2
    segments = get_segments_ends(vid_obj,shiftx,shifty,40,R,trans)
    edges = get_all_edges(exp, t)
    edges = [edge for edge in edges if dist_edge(edge,positions,t)<=dist]
    pixels = [pixel for edge in edges for pixel in edge.pixel_list(t)]
    pixels = [pixel for pixel in pixels if np.linalg.norm(pixel-positions)<=1.5*dist]
    segment_points = []
    for begin, end in segments:
        # Include the start point, interpolated points, and the end point
        interpolated_points = interpolate_points(begin, end)
        segment_points.extend(interpolated_points)
        segment_points.append(end)  # Ensure the end point is included

    segment_points = np.array(segment_points)
    Y = np.array(pixels)
    X = np.array(segment_points)
    transformation = find_rot_o3d(X,Y)
    Rfound = transformation[0:2, 0:2]
    tfound = transformation[0:2, 3]
    if np.linalg.det(Rfound)>0:
        return(Rfound,tfound)
    else:
        # print("negative det")
        transformation = find_rot_o3d(Y, X)
        Rfound = transformation[0:2, 0:2]
        tfound = transformation[0:2, 3]
        Rfound, tfound = np.linalg.inv(Rfound), np.dot(
            np.linalg.inv(Rfound), -tfound
        )
        return (Rfound, tfound)

def find_rot_o3d(X,Y):
    X = np.transpose(X)
    Y = np.transpose(Y)
    X = np.insert(X, 2, values=0, axis=0)
    Y = np.insert(Y, 2, values=0, axis=0)
    vectorX = o3d.utility.Vector3dVector(np.transpose(X))
    vectorY = o3d.utility.Vector3dVector(np.transpose(Y))
    source = o3d.geometry.PointCloud(vectorX)
    target = o3d.geometry.PointCloud(vectorY)
    threshold = 200
    trans_init = np.asarray(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    # print('registering')
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return(reg_p2p.transformation)

def dist_edge(edge, pos, t):
    pos = pos.astype(float)

    dists = np.linalg.norm(np.array(edge.pixel_list(t)) - pos, axis=1)

    return (np.min(dists))


def interpolate_points(start, end, step=1):
    """Interpolate points between start and end with a given step."""
    # Vector from start to end
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    direction_normalized = direction / length

    # Number of points to generate (excluding the end point)
    num_points = int(length // step)

    # Generate points
    points = [np.array(start) + direction_normalized * step * i for i in range(num_points + 1)]

    return points

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1-p2)

def average_min_distance_to_set(A, B_set):
    """Calculate the average minimum distance from each point in A to the closest point in a B set."""
    total_distance = 0
    for a in A:
        min_distance = min([euclidean_distance(a, b) for b in B_set])
        total_distance += min_distance
    return total_distance / len(A)

def transform(pos,R,t):
    return(R @ pos + t)

def find_index_min(A, B):
    avg_distances = [average_min_distance_to_set(A, B_set) for B_set in B]
    return (np.argmin(avg_distances),np.min(avg_distances))

def find_mapping(transport_edge_segment,network_edge_names,network_edge_segments):
    index,avg_distances = find_index_min(transport_edge_segment,network_edge_segments)
    return(network_edge_names[index],avg_distances)

def make_whole_mapping(vid_obj,exp,t,dist = 100,R = np.array([[1,0],[0,1]]),trans = 0):
    positions = np.array(vid_obj.dataset[["xpos_network", "ypos_network"]])
    shiftx = vid_obj.img_dim[0]*vid_obj.space_res/1.725/2
    shifty = vid_obj.img_dim[1]*vid_obj.space_res/1.725/2
    Rfound,tfound = register_rot_trans(vid_obj,exp,t,dist= dist,R=R,trans=trans)
    segments_final = get_segments_ends(vid_obj,shiftx,shifty,0,R,trans)
    edge_names = [edge.edge_name for edge in vid_obj.edge_objs]
    segments_final_interp = []
    for begin, end in segments_final:
        # Include the start point, interpolated points, and the end point
        interpolated_points = interpolate_points(begin, end)
        segments_final_interp.append(interpolated_points)
    edges = get_all_edges(exp, t)

    edges = [edge for edge in edges if dist_edge(edge,transform(positions,Rfound,tfound),t)<=100]
    network_edge_segments = [edge.pixel_list(t) for edge in edges]
    network_edge_names = edges
    mapping = {}
    avg_distances = []
    for transport_edge_name,transport_edge_segment in  zip(edge_names,segments_final_interp):
        mapping[transport_edge_name],avg_distance = find_mapping(transport_edge_segment,network_edge_names,network_edge_segments)
        avg_distances.append(avg_distance)
    return(mapping,np.mean(avg_distance),Rfound,tfound)

def add_attribute(edge_data_csv,vid_edge_obj,network_edge_attribute,name_new_col,mapping):
    new_attribute = network_edge_attribute(mapping[vid_edge_obj.edge_name])
    edge_data_csv.loc[edge_data_csv['edge_name'] == vid_edge_obj.edge_name, name_new_col] = new_attribute

def check_hasedges(vid_obj):
    shiftx = vid_obj.img_dim[0]*vid_obj.space_res/1.725/2
    shifty = vid_obj.img_dim[1]*vid_obj.space_res/1.725/2
    segments = get_segments_ends(vid_obj,shiftx,shifty,40)
    return(len(segments)>0)

def initialize_transformation():
    return np.array([[1,0],[0,1]]), np.array([0,0])

def update_transformation(Rcurrent, tcurrent, Rfound, tfound):
    Rcurrent = Rfound @ Rcurrent
    tcurrent = Rfound @ tcurrent + tfound
    return Rcurrent, tcurrent

def should_reset(Rfound):
    return np.linalg.det(Rfound) <= 0 or Rfound[0][0] <= 0.99

def attempt_mapping(vid_obj, exp, t, Rcurrent, tcurrent):
    try:
        mapping, dist, Rfound, tfound = make_whole_mapping(vid_obj, exp, t, dist=100, R=Rcurrent, trans=tcurrent)
    except IndexError:
        Rcurrent, tcurrent = initialize_transformation()
        mapping, dist, Rfound, tfound = make_whole_mapping(vid_obj, exp, t, dist=100, R=Rcurrent, trans=tcurrent)
    return mapping, dist, Rfound, tfound

def process_video_object(vid_obj, exp, t, Rcurrent, tcurrent):
    mapping, dist, Rfound, tfound = attempt_mapping(vid_obj, exp, t, Rcurrent, tcurrent)
    if np.linalg.det(Rfound) > 0 and Rfound[0][0] > 0.99:
        Rcurrent, tcurrent = update_transformation(Rcurrent, tcurrent, Rfound, tfound)
        if dist > 20:
            mapping, dist, Rfound, tfound = attempt_mapping(vid_obj, exp, t, Rcurrent, tcurrent)
            if should_reset(Rfound):
                Rcurrent, tcurrent = initialize_transformation()
    else:
        Rcurrent, tcurrent = initialize_transformation()
    return Rcurrent, tcurrent, mapping, dist

def register_dataset(data_obj, exp, t):
    Rcurrent, tcurrent = initialize_transformation()

    for index, vid_obj in enumerate(data_obj.video_objs[35:]):
        if check_hasedges(vid_obj):
            Rcurrent, tcurrent, mapping, dist = process_video_object(vid_obj, exp, t, Rcurrent, tcurrent)
            print(index, dist, Rcurrent, tcurrent)
            update_edge_attributes(vid_obj, mapping, dist, t)

def update_edge_attributes(vid_obj, mapping, dist, t):
    edge_data_csv = pd.read_csv(vid_obj.edge_adr)
    for edge in vid_obj.edge_objs:
        add_attribute(edge_data_csv, edge, lambda edge: edge.width(t), "width", mapping)
        add_attribute(edge_data_csv, edge, lambda edge: edge.end.label, "network_end", mapping)
        add_attribute(edge_data_csv, edge, lambda edge: edge.begin.label, "network_begin", mapping)
        add_attribute(edge_data_csv, edge, lambda edge: dist, "mapping_quality", mapping)
    edge_data_csv.to_csv(vid_obj.edge_adr)
