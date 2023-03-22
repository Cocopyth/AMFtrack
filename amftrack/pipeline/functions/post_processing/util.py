from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Node,
    Edge,
)
import numpy as np


def measure_length_um(seg):
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


def measure_length_um_edge(edge, t):
    pixel_conversion_factor = 1.725
    length_edge = 0
    pixels = edge.pixel_list(t)
    for i in range(len(pixels) // 10 + 1):
        if i * 10 <= len(pixels) - 1:
            length_edge += np.linalg.norm(
                np.array(pixels[i * 10])
                - np.array(pixels[min((i + 1) * 10, len(pixels) - 1)])
            )
    #             length_edge+=np.linalg.norm(np.array(pixels[len(pixels)//10-1*10-1])-np.array(pixels[-1]))
    return length_edge * pixel_conversion_factor


def get_length_um_node_list(node_list, exp, t):
    total_length = 0
    for i in range(len(node_list) - 1):
        nodea = Node(node_list[i], exp)
        nodeb = Node(node_list[i + 1], exp)
        edge_obj = Edge(nodea, nodeb, exp)
        total_length += measure_length_um_edge(edge_obj, t)
    return total_length


def is_in_study_zone_pos(pos, exp, t, radius, dist, is_circle=False):
    compress = 25
    if is_circle:
        center = np.array([4700 * 5, 4400 * 5])
        radius = 3900 * 5
        return is_in_circle(pos, center, radius)
    else:
        y, x = pos[0], pos[1]
        center = np.array(exp.center)
        x0, y0 = exp.center
        direction = exp.orthog
        pos_line = np.array((x0, y0)) + dist * compress * direction
        x_line, y_line = pos_line[0], pos_line[1]
        orth_direct = np.array([direction[1], -direction[0]])
        x_orth, y_orth = orth_direct[0], orth_direct[1]
        a = y_orth / x_orth
        b = y_line - a * x_line
        dist_center = np.linalg.norm(np.flip(pos) - center)
    return (dist_center < radius * compress, a * x + b > y)


def is_in_study_zone(node, t, radius, dist, is_circle=False):
    exp = node.experiment
    pos = node.pos(t)
    return is_in_study_zone_pos(pos, exp, t, radius, dist, is_circle)


def is_in_circle(pos, center, radius):
    return np.linalg.norm(pos - center) <= radius
