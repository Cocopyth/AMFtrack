import sys

sys.path.insert(0, "/home/cbisot/pycode/MscThesis/")
import pandas as pd
from amftrack.pipeline.launching.run_super import directory_project
from time import time

# import multiprocessing
from joblib import Parallel, delayed
from time import sleep
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.util.sys import temp_path

directory = directory_project
skip = False
i = 330
op_id = 1635343105134052723

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
plate = list(run_info["PrincePos"])[i]

folder_list = list(run_info["folder"])
directory_name = folder_list[i]
exp = Experiment(plate, directory)
exp.load(run_info.loc[run_info["folder"] == directory_name], suffix="")
path_snap = directory + directory_name
suffix = "/Analysis/nx_graph_pruned.p"

(G, pos) = exp.nx_graph[0], exp.positions[0]

num_cores = 1

t = 0
resolution = 50
skip = False
experiment = exp
edge_width = {}
graph = experiment.nx_graph[t]
# print(len(list(graph.edges)))
# print(len(graph.edges))
end = 10
inputs = list(graph.edges)[:end]
print(type(inputs[0]))
# inputs = range(end)

from skimage.measure import profile_line
from amftrack.notebooks.analysis.util import *
from scipy.optimize import curve_fit
import numpy as np

a = 2.3196552
from scipy import special


def func2(x, lapse, lapse2, c, d, e):
    return (
        -c * (special.erf(e * (x - lapse)) - special.erf(e * (x - lapse - lapse2))) + d
    )


def func3(x, lapse, lapse2, c, d, e, lapse4):
    return (
        -c * (special.erf(e * (x - lapse)) - special.erf(e * (x - (lapse + lapse2))))
        + d
        + c
        * (
            special.erf(e * (x - (lapse + lapse2)))
            - special.erf(e * (x - (lapse + lapse2 + lapse4)))
        )
    )


def func4(x, lapse, lapse2, c, d, e, lapse4):
    return (
        -c * (special.erf(e * (x - lapse)) - special.erf(e * (x - (lapse + lapse2))))
        + d
        + c * (special.erf(e * (x - (lapse - lapse4))) - special.erf(e * (x - (lapse))))
    )


def func5(x, sigma, mean, fact, offset):
    return -fact * np.exp(-((x - mean) ** 2) / sigma**2) + offset


def func5(x, sigma, mean, fact, offset):
    return -fact * np.exp(-((x - mean) ** 2) / sigma**2) + offset


def get_source_image(experiment, pos, t, local, force_selection=None):
    x, y = pos[0], pos[1]
    ims, posimg = experiment.find_image_pos(x, y, t, local)
    if force_selection is None:
        dist_border = [
            min([posimg[1][i], 3000 - posimg[1][i], posimg[0][i], 4096 - posimg[0][i]])
            for i in range(posimg[0].shape[0])
        ]
        j = np.argmax(dist_border)
    else:
        dist_last = [
            np.linalg.norm(
                np.array((posimg[1][i], posimg[0][i])) - np.array(force_selection)
            )
            for i in range(posimg[0].shape[0])
        ]
        j = np.argmin(dist_last)
    return (ims[j], (posimg[1][j], posimg[0][j]))


def get_width_pixel(
    edge,
    index,
    im,
    pivot,
    before,
    after,
    t,
    size=20,
    width_factor=60,
    averaging_size=100,
    threshold_averaging=10,
):
    imtab = im
    #     print(imtab.shape)
    #     print(int(max(0,pivot[0]-averaging_size)),int(pivot[0]+averaging_size))
    threshold = np.mean(
        imtab[
            int(max(0, pivot[0] - averaging_size)) : int(pivot[0] + averaging_size),
            int(max(0, pivot[1] - averaging_size)) : int(pivot[1] + averaging_size),
        ]
        - threshold_averaging
    )
    orientation = np.array(before) - np.array(after)
    perpendicular = (
        [1, -orientation[0] / orientation[1]] if orientation[1] != 0 else [0, 1]
    )
    perpendicular_norm = np.array(perpendicular) / np.sqrt(
        perpendicular[0] ** 2 + perpendicular[1] ** 2
    )
    point1 = np.around(np.array(pivot) + width_factor * perpendicular_norm)
    point2 = np.around(np.array(pivot) - width_factor * perpendicular_norm)
    point1 = point1.astype(int)
    point2 = point2.astype(int)
    p = profile_line(imtab, point1, point2, mode="constant")
    xdata = np.array(range(len(p)))
    ydata = np.array(p)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.plot(xdata,ydata)
    #     ax.plot(xdata, func5(xdata, *popt0), 'g-')
    try:
        p00 = [10, 60, 60, 160]
        popt0, pcov = curve_fit(
            func5, xdata, ydata, bounds=([0, 0, 0, 0], 4 * [np.inf]), p0=p00
        )
        p0a = [60, 10, 100, 180, 0.1]
        popt1, pcov = curve_fit(
            func2,
            xdata,
            ydata,
            bounds=([0, 0, 0, 0, 0], [120, 120, 200] + 2 * [np.inf]),
            p0=p0a,
        )
        p0b = list(popt1) + [10]
        popt2, pcov = curve_fit(
            func3,
            xdata,
            ydata,
            bounds=([0, 0, 0, 0, 0, 0], [120, 120, 200] + 2 * [np.inf] + [120]),
            p0=p0b,
        )
        residuals = ydata - func3(xdata, *popt2)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared1 = 1 - (ss_res / ss_tot)
        popt3, pcov = curve_fit(
            func4,
            xdata,
            ydata,
            bounds=([0, 0, 0, 0, 0, 0], [120, 120, 200] + 2 * [np.inf] + [120]),
            p0=p0b,
        )
        residuals = ydata - func4(xdata, *popt3)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared2 = 1 - (ss_res / ss_tot)
        #     ax.plot(xdata, func2(xdata, *popt1), 'r-')
        if r_squared1 > r_squared2:
            #         ax.plot(xdata, func3(xdata, *popt2), 'b-')
            popt = popt2
        else:
            #         ax.plot(xdata, func4(xdata, *popt3), 'b-')
            popt = popt3
        background = popt[3]
    except RuntimeError:
        print("failed")
        background = np.mean(p)
    #     print(popt[3],popt0[3])
    #     width_pix = popt0[0]*popt0[2]
    width_pix = -np.sum(
        (np.log10(np.array(p) / background) <= 0) * np.log10(np.array(p) / background)
    )
    #     print(width_pix)
    #     p0=[165,100,165,45,10,10,10]
    #     popt, pcov = curve_fit(func, xdata, ydata,bounds = ([-np.inf,-np.inf,-np.inf,-np.inf,0,0,0],np.inf),p0=p0)
    #     width_pix = popt[-2]
    #     ax.plot(xdata, func(xdata, *popt), 'r-')
    #     derivative = [p[i+1]-p[i] for i in range(len(p)-1)]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.plot([np.mean(derivative[5*i:5*i+5]) for i in range(len(derivative)//5)])
    #     problem=False
    #     arg = len(p)//2
    #     if p[arg]>threshold:
    #         arg = np.argmin(p)
    # #     we_plot=randrange(1000)
    #     while  p[arg]<=threshold:
    #         if arg<=0:
    # #             we_plot=50
    #             problem=True
    #             break
    #         arg-=1
    #     begin = arg
    #     arg = len(p)//2
    #     if p[arg]>threshold:
    #         arg = np.argmin(p)
    #     while  p[arg]<=threshold:
    #         if arg>=len(p)-1:
    # #             we_plot=50
    #             problem=True
    #             break
    #         arg+=1
    #     end = arg
    # #     print(end-begin,threshold)
    #     print(np.linalg.norm(point1-point2),len(p),width_pix)
    return a * np.sqrt(max(0, np.linalg.norm(point1 - point2) * (width_pix) / len(p)))


def get_width_edge(pixel_list, resolution, t, local=False, threshold_averaging=10):
    # edge = Edge(Node(edge[0],experiment),Node(edge[1],experiment),experiment)
    # pixel_conversion_factor = 1.725
    # pixel_list = edge.pixel_list(t)
    pixels = []
    indexes = []
    source_images = []
    poss = []
    widths = {}
    if len(pixel_list) > 3 * resolution:
        for i in range(0, len(pixel_list) // resolution):
            index = i * resolution
            indexes.append(index)
            pixel = pixel_list[index]
            pixels.append(pixel)
            source_img, pos = get_source_image(experiment, pixel, t, local)
            source_images.append(source_img)
            poss.append(pos)
    else:
        indexes = [0, len(pixel_list) // 2, len(pixel_list) - 1]
        for index in indexes:
            pixel = pixel_list[index]
            pixels.append(pixel)
            source_img, pos = get_source_image(experiment, pixel, t, local)
            source_images.append(source_img)
            poss.append(pos)
    #     print(indexes)
    for i, index in enumerate(indexes[1:-1]):
        source_img = source_images[i + 1]
        pivot = poss[i + 1]
        _, before = get_source_image(experiment, pixels[i], t, local, pivot)
        _, after = get_source_image(experiment, pixels[i + 2], t, local, pivot)
        #         plot_t_tp1([0,1,2],[],{0 : pivot,1 : before, 2 : after},None,source_img,source_img)
        width = get_width_pixel(
            edge,
            index,
            source_img,
            pivot,
            before,
            after,
            t,
            threshold_averaging=threshold_averaging,
        )
        #         print(width*pixel_conversion_factor)
        widths[pixel_list[index]] = width * pixel_conversion_factor
    #         if i>=1:
    #             break
    # edge.experiment.nx_graph[t].get_edge_data(edge.begin.label,edge.end.label)['width'] = widths
    return widths


def get_width_info(experiment, t, resolution=50, skip=False):
    print(not skip)
    edge_width = {}
    graph = experiment.nx_graph[t]
    #     print(len(list(graph.edges)))
    # print(len(graph.edges))
    for edge in graph.edges:
        if not skip:
            #         print(edge)
            # edge_exp = Edge(Node(edge[0],experiment),Node(edge[1],experiment),experiment)
            mean = np.mean(
                list(get_width_edge(edge, experiment, resolution, t).values())
            )
            #         print(np.mean(list(get_width_edge(edge_exp,resolution,t).values())))
            edge_width[edge] = mean
            # print(mean)
        else:
            # Maybe change to Nan if it doesnt break the rest
            edge_width[edge] = 40
    return edge_width


def extract_width(pixel_list):
    # edge_exp = Edge(Node(edge[0],experiment),Node(edge[1],experiment),experiment)
    # mean = np.mean(list(get_width_edge(pixel_list,resolution,t).values()))
    # print(mean)
    # return(mean)
    sleep(12)


edges = list(graph.edges)[:end]
inputs = []
for edge in edges:
    edge_obj = Edge(Node(edge[0], experiment), Node(edge[1], experiment), experiment)
    pixel_conversion_factor = 1.725
    pixel_list = edge_obj.pixel_list(t)
    inputs.append(pixel_list)
t0 = time()
if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(extract_width)(i) for i in inputs
    )
print("1 core", time() - t0)
num_cores = 10

t0 = time()

if __name__ == "__main__":
    processed_list = Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(extract_width)(i) for i in inputs
    )
print("10 core", time() - t0)

t0 = time()

processed_list = [extract_width(i) for i in inputs]
print("no_parallel", time() - t0)
