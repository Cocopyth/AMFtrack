from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.util.sys import temp_path
import pickle
import pandas as pd
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    load_graphs,
)
from amftrack.pipeline.functions.post_processing.extract_study_zone import (
    load_study_zone,
)
from amftrack.pipeline.functions.image_processing.hyphae_id_surf import (
    get_pixel_growth_and_new_children,
)
import numpy as np
from amftrack.pipeline.functions.post_processing.util import measure_length_um
import networkx as nx
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Node,
)
import os

directory = str(sys.argv[1])
overwrite = eval(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
list_f, list_args = pickle.load(open(f"{temp_path}/{op_id}.pick", "rb"))
folder_list = list(run_info["folder_analysis"])
directory_name = folder_list[i]
select = run_info.loc[run_info["folder_analysis"] == directory_name]
row = [row for index, row in select.iterrows()][0]
path = f'{directory}{row["folder_analysis"]}'
plate_num = row["Plate"]
path_exp = f'{directory}{row["path_exp"]}'
exp = pickle.load(open(path_exp, "rb"))
exp.save_location = "/".join(path_exp.split("/")[:-1])
try:
    exp.labeled
except AttributeError:
    exp.labeled = True

load_study_zone(exp)

load_graphs(exp, directory, post_process=True)
exp.dates.sort()


def get_hyph_infos(exp):
    select_hyph = {}
    for hyph in exp.hyphaes:
        select_hyph[hyph] = []
        for i, t in enumerate(hyph.ts[:-1]):
            tp1 = hyph.ts[i + 1]
            try:
                pixels, nodes = get_pixel_growth_and_new_children(hyph, t, tp1)
                speed = np.sum([get_length_um(seg) for seg in pixels]) / get_time(
                    exp, t, tp1
                )
                select_hyph[hyph].append((t, hyph.ts[i + 1], speed, pixels))
            except nx.NetworkXNoPath:
                pass
    return select_hyph

def get_time(exp, t, tp1):  # redefined here to avoid loop in import
    seconds = (exp.dates[tp1] - exp.dates[t]).total_seconds()
    return seconds / 3600

def get_rh_bas(exp):
    select_hyph = get_hyph_infos(exp)
    max_speeds = []
    total_growths = []
    lengths = []
    branch_frequ = []
    hyph_l = []
    RH = []
    BAS = []
    widths = []
    for hyph in exp.hyphaes:
        try:
            speeds = [c[2] for c in select_hyph[hyph]]
            ts = [c[0] for c in select_hyph[hyph]]
            tp1s = [c[1] for c in select_hyph[hyph]]
            if len(speeds) > 0:
                node = hyph.end
                length = np.linalg.norm(node.pos(node.ts()[0]) - node.pos(node.ts()[-1]))
                nodes = hyph.get_nodes_within(hyph.ts[-1])[0]
                max_speed = np.max(speeds)
                total_growth = np.sum(
                    [
                        speed * get_time(exp, ts[i], tp1s[i])
                        for i, speed in enumerate(speeds)
                    ]
                )
                if length>=1500:
                    RH.append(hyph)
                else:
                    BAS.append(hyph)
                lengths.append(length)
                max_speeds.append(max_speed)
                total_growths.append(total_growth)
                branch_frequ.append((len(nodes) - 1) / (length + 1))
                hyph_l.append(hyph)
                #             widths.append(get_width_hypha(hyph,hyph.ts[-1]))
                widths.append(5)

            else:
                BAS.append(hyph)
        except nx.NetworkXNoPath:
            pass
    return (
        RH,
        BAS,
        max_speeds,
        total_growths,
        widths,
        lengths,
        branch_frequ,
        select_hyph,
    )
def get_length_um(seg):
    pixel_conversion_factor = 1.725
    pixels = seg
    length_edge = 0
    for i in range(len(pixels) // 10 + 1):
        if i * 10 <= len(pixels) - 1:
            length_edge += np.linalg.norm(
                np.array(pixels[i * 10])
                - np.array(pixels[min((i + 1) * 10, len(pixels) - 1)])
            )
    #         length_edge+=np.linalg.norm(np.array(pixels[len(pixels)//10-1*10-1])-np.array(pixels[-1]))
    return length_edge * pixel_conversion_factor

def estimate_angle(exp):
    (
        RH,
        BAS,
        max_speeds,
        total_growths,
        widths,
        lengths,
        branch_frequ,
        select_hyph,
    ) = get_rh_bas(exp)
    branch_root = []
    branch_anastomose = []
    two_time = []
    angles = []
    for rh in RH:
        #     rh = choice(RH)
        t = rh.ts[-1]
        nodes, edges = rh.get_nodes_within(t)
        for i, node in enumerate(nodes[1:-1]):
            found = False
            for hyph in exp.hyphaes:
                if hyph.root.label == node:
                    if found:
                        two_time.append(hyph.root)
                    branch_root.append(hyph.root)
                    if t in hyph.ts:
                        nodes_h, edges_h = hyph.get_nodes_within(t)
                        if len(edges_h) > 0:
                            edge_main = edges[i + 1]
                            edge_branch = edges_h[0]
                            angle_main = get_orientation(rh, t, i + 1, 100)
                            angle_branch = get_orientation(hyph, t, 0, 100)
                            angles.append(((angle_main - angle_branch), (rh, hyph, t)))
                            #                         print(node,edges[i+1],edges_h[0],angle_main-angle_branch)
                            #                         exp.plot([t],[[node,edge_main.begin.label,edge_main.end.label,edge_branch.begin.label,edge_branch.end.label]])
                            found = True
            if not found:
                branch_anastomose.append(Node(node, exp))
    angles_180 = [(angle + 180) % 360 - 180 for angle, infos in angles]
    angles_rh = [(c[0] + 180) % 360 - 180 for c in angles if c[1][1] in RH]
    angles_bas = [(c[0] + 180) % 360 - 180 for c in angles if c[1][1] in BAS]
    return (angles_rh, angles_bas)

def get_orientation(hypha, t, start, length=50):
    nodes, edges = hypha.get_nodes_within(t)
    pixel_list_list = []
    #     print(edges[start:])
    for edge in edges[start:]:
        pixel_list_list += edge.pixel_list(t)
    pixel_list = np.array(pixel_list_list)
    vector = pixel_list[min(length, len(pixel_list) - 1)] - pixel_list[0]
    unit_vector = vector / np.linalg.norm(vector)
    vertical_vector = np.array([-1, 0])
    dot_product = np.dot(vertical_vector, unit_vector)
    if (
        vertical_vector[1] * vector[0] - vertical_vector[0] * vector[1] >= 0
    ):  # determinant
        angle = np.arccos(dot_product) / (2 * np.pi) * 360
    else:
        angle = -np.arccos(dot_product) / (2 * np.pi) * 360
    return angle

angles_rh, angles_bas = estimate_angle(exp)
np.save(os.path.join(path,"rh_angle"),angles_rh)
np.save(os.path.join(path,"bas_angle"),angles_bas)