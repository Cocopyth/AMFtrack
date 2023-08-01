import os
import matplotlib.pyplot as plt
import os
import pandas as pd
from amftrack.util.sys import get_analysis_folders, get_time_plate_info_from_analysis
import numpy as np
import imageio
import os
import cv2
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib as mpl

cmap1 = mpl.cm.get_cmap("spring")
cmap2 = mpl.cm.get_cmap("winter")


def wave(xt, c, lamb, K, x0):
    x = xt[0, :]
    t = xt[1, :]
    return K * (1 / (1 + np.exp(lamb * (x0 + x - c * t))))


def dwave(xt, c, lamb, K, x0):
    """This function represents a mathematical model that describes the
        amplitude of a traveling wave along a one-dimensional medium.
         The function takes in the input variables: xt, c, lamb, K, and x0,
         and returns the amplitude of the wave.

    Parameters:

        xt (ndarray): A 2xN numpy array representing the spatiotemporal
         coordinates of the wave. The first row contains the spatial
         coordinates x, and the second row contains the temporal coordinates t.
        c (float): The speed of the wave.
        lamb (float): The wave attenuation coefficient.
        K (float): The maximum amplitude of the wave.
        x0 (float): The initial position of the wave.

    Returns:

        ndarray: A 1D numpy array of the same length as xt[0] representing the amplitude of the wave at each spatiotemporal coordinate in xt."""
    x = xt[0, :]
    t = xt[1, :]
    return K * (
        np.exp(lamb * (x0 + x - c * t)) / (1 + np.exp(lamb * (x0 + x - c * t))) ** 2
    )


def dS(t, lamb, C, t0):
    return C * ((np.exp(lamb * (t0 - t)) / (1 + np.exp(lamb * (t0 - t))) ** 2))


def S(t, lamb, C, t0):
    return C * (1 / (1 + np.exp(lamb * (t0 - t))))


def get_wave_fit(time_plate_info, plate, timesteps, max_indexes, lamb=-1, C=0.2,suffix = ""):
    table = time_plate_info.loc[time_plate_info["Plate"] == plate]
    table = table.replace(np.nan, -1)
    ts = list(table["timestep"])
    table = table.set_index("timestep")
    ts.sort()
    dic = {}
    tot_t = list(table.index)
    tot_t.sort()
    # timesteps = [tot_t[0],tot_t[80]]

    ts = []
    xs = []
    ys = []
    for time in timesteps:
        #     ax.set_yscale("log")

        maxL = np.sqrt(1900)
        X = np.linspace(0, maxL, 100)
        incr = 100

        def density(x):
            area = x**2
            index = int(area // incr)
            column = f"ring_density_incr-100_index-{index}{suffix}"
            return float(table[column][time])

        xvalues = np.array([np.sqrt(100 * i) for i in range(max_indexes[plate])])
        yvalues = [density(x) for x in xvalues]
        xvalues = np.sqrt((xvalues**2 + table["area_sep_comp"][0]) / (np.pi / 2))
        xvalues = list(xvalues)
        tvalues = [table["time_since_begin_h"][time] for x in xvalues]
        ts += tvalues
        xs += xvalues
        ys += yvalues
    xt = np.array((xs, ts))
    popt_f, cov = curve_fit(
        wave,
        xt,
        ys,
        bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        p0=[0.2, -lamb, C, 0],
    )
    # popt_f, cov = curve_fit(
    #     wave,
    #     xt,
    #     ys,
    #     bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
    #     p0=[0.2] + list(popt_f[1:]),
    # )
    # popt_f[0]/=1.5

    popt_f
    residuals = ys - wave(xt, *popt_f)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r_squared_dens = 1 - (ss_res / ss_tot)
    ts = []
    xs = []
    ys = []
    for time in timesteps:
        #     ax.set_yscale("log")

        maxL = np.sqrt(1900)
        X = np.linspace(0, maxL, 100)
        incr = 100

        def density(x):
            area = x**2
            index = int(area // incr)
            column = f"ring_active_tips_density_incr-100_index-{index}"
            return float(table[column][time])

        xvalues = np.array([np.sqrt(100 * i) for i in range(max_indexes[plate])])
        yvalues = [density(x) for x in xvalues]
        xvalues = np.sqrt((xvalues**2 + table["area_sep_comp"][0]) / (np.pi / 2))
        xvalues = list(xvalues)
        tvalues = [table["time_since_begin_h"][time] for x in xvalues]
        ts += tvalues
        xs += xvalues
        ys += yvalues
    xt = np.array((xs, ts))
    ys = np.array(ys)
    pos = np.where(ys >= 0)[0]
    xt = xt[:, pos]
    ys = ys[pos]
    popt_f2, cov = curve_fit(
        dwave,
        xt,
        ys,
        bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
        p0=[0.2, 1, 0.3, 0],
    )
    residuals = ys - dwave(xt, *popt_f2)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r_squared_tips = 1 - (ss_res / ss_tot)
    return (popt_f, r_squared_dens, popt_f2, r_squared_tips)


def plot_single_plate(
    plate,
    time_plate_info,
    timestep_max,
    ax,
    maxi=10,
    max_area=50,
    savefig=None,
    unique_id=False,
):
    ax.set_title(f"plate {plate}")
    ax2 = ax.twinx()
    if unique_id:
        table = time_plate_info.loc[time_plate_info["unique_id"] == plate].copy()
    else:
        table = time_plate_info.loc[time_plate_info["Plate"] == plate].copy()
    table = table.loc[table["timestep"] <= timestep_max]
    table = table.set_index("timestep")
    ts = []
    ys = []
    ys2 = []
    Cs = []
    lambs = []
    indexes = []
    t0s = []
    ds = []
    for index in range(1, maxi):
        column = f"ring_density_incr-100_index-{index}"
        column2 = f"ring_active_tips_density_incr-100_index-{index}"

        start = np.min(table.loc[table[column] >= 400]["time_since_begin"])
        if not np.isnan(start):
            table[f"time_since_begin_{index}"] = table["time_since_begin"] - start

            area = np.sqrt(table["area_sep_comp"][0] + 100 * index)

            selection_fit = table
            try:
                popt0, pcov = curve_fit(
                    S,
                    selection_fit[f"time_since_begin_{index}"],
                    selection_fit[column],
                    bounds=([0, 0, -np.inf], 3 * [np.inf]),
                    p0=[1, 1, 0],
                )
            except:
                # print(selection_fit[column2])
                continue
            lamb, C, t0 = list(popt0)

            table[f"time_since_begin_{index}"] = table[f"time_since_begin_{index}"] - t0

            ax.scatter(
                table[f"time_since_begin_{index}"],
                table[column],
                alpha=0.5,
                color=cmap2(area / max_area),
            )
            ax2.scatter(
                table[f"time_since_begin_{index}"],
                table[column2],
                alpha=0.5,
                color=cmap1(area / max_area),
            )
            Cs.append(C)
            lambs.append(lamb)
            indexes.append(index)
            t0s.append(t0 + start)
            ds.append(int(area / np.sqrt((np.pi / 2))))
            x = np.linspace(-50, 50, 100)
            ax.plot(
                x,
                S(x + t0, lamb, C, t0),
                color=cmap2(area / max_area),
                label=f"d = {int(area / np.sqrt((np.pi / 2)))}mm",
            )
            try:
                popt1, _ = curve_fit(
                    dS,
                    selection_fit[f"time_since_begin_{index}"],
                    selection_fit[column2],
                    bounds=([0, 0, -np.inf], 3 * [np.inf]),
                    p0=[0.2, 0.5, 0],
                )
            except:
                continue
            lamb, C, t1 = list(popt1)

            ax2.plot(
                x,
                dS(x + t0, lamb, C, t1),
                color=cmap1(area / max_area),
                label=f"d = {int(area / np.sqrt((np.pi / 2)))}mm",
            )
            ts += table[f"time_since_begin_{index}"].to_list()
            ys += table[column].to_list()
            ys2 += table[column2].astype(float).to_list()
    df = pd.DataFrame(
        (np.array((ts, ys, ys2))).transpose(), columns=("ts", "ys", "ys2")
    )
    factor = 4
    df["ts_round"] = (df["ts"] / factor).astype(int) * factor
    meancurve = df.groupby("ts_round")["ys"].mean()
    ax.plot(meancurve.index, meancurve, label="mean", color="black")
    meancurve2 = df.groupby("ts_round")["ys2"].mean()
    # ax2.plot(meancurve.index, meancurve2, label=plate, color = 'red')
    ax.set_xlim((-30, 30))
    ax2.set_ylim((0, 0.25))
    ax.set_ylim((0, 2500))

    ax.set_ylabel("network density ($\mu m.mm^{-2}$)")
    ax.set_xlabel("shifted time ($h$)")
    ax2.set_ylabel("active tips density ($mm^{-2}$)")
    ax.tick_params(axis="y", colors="blue")
    ax2.tick_params(axis="y", colors="red")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if not savefig is None:
        plt.savefig(savefig)
    return (Cs, lambs, ds, indexes, t0s, meancurve, meancurve2)


def plot_single_plate_biovolume(
    plate,
    time_plate_info,
    timestep_max,
    ax,
    maxi=10,
    max_area=50,
    savefig=None,
    unique_id=False,
):
    ax.set_title(f"plate {plate}")
    ax2 = ax.twinx()
    if unique_id:
        table = time_plate_info.loc[time_plate_info["unique_id"] == plate].copy()
    else:
        table = time_plate_info.loc[time_plate_info["Plate"] == plate].copy()
    table = table.loc[table["timestep"] <= timestep_max]
    table = table.set_index("timestep")
    ts = []
    ys = []
    ys2 = []
    Cs = []
    lambs = []
    indexes = []
    t0s = []
    ds = []
    for index in range(1, maxi):
        column = f"ring_biovolume_density_incr-100_index-{index}"
        column2 = f"ring_active_tips_density_incr-100_index-{index}"

        start = np.min(table.loc[table[column] >= 5000]["time_since_begin"])
        if not np.isnan(start):
            table[f"time_since_begin_{index}"] = table["time_since_begin"] - start

            area = np.sqrt(table["area_sep_comp"][0] + 100 * index)

            selection_fit = table
            try:
                popt0, pcov = curve_fit(
                    S,
                    selection_fit[f"time_since_begin_{index}"],
                    selection_fit[column],
                    bounds=([0, 0, -np.inf], 3 * [np.inf]),
                    p0=[1, 1, 0],
                )
            except:
                # print(selection_fit[column2])
                continue
            lamb, C, t0 = list(popt0)

            table[f"time_since_begin_{index}"] = table[f"time_since_begin_{index}"] - t0

            ax.scatter(
                table[f"time_since_begin_{index}"],
                table[column],
                alpha=0.5,
                color=cmap2(area / max_area),
            )
            ax2.scatter(
                table[f"time_since_begin_{index}"],
                table[column2],
                alpha=0.5,
                color=cmap1(area / max_area),
            )
            Cs.append(C)
            lambs.append(lamb)
            indexes.append(index)
            t0s.append(t0 + start)
            ds.append(int(area / np.sqrt((np.pi / 2))))
            x = np.linspace(-50, 50, 100)
            ax.plot(
                x,
                S(x + t0, lamb, C, t0),
                color=cmap2(area / max_area),
                label=f"d = {int(area / np.sqrt((np.pi / 2)))}mm",
            )
            try:
                popt1, _ = curve_fit(
                    dS,
                    selection_fit[f"time_since_begin_{index}"],
                    selection_fit[column2],
                    bounds=([0, 0, -np.inf], 3 * [np.inf]),
                    p0=[0.2, 0.5, 0],
                )
            except:
                continue
            lamb, C, t1 = list(popt1)

            ax2.plot(
                x,
                dS(x + t0, lamb, C, t1),
                color=cmap1(area / max_area),
                label=f"d = {int(area / np.sqrt((np.pi / 2)))}mm",
            )
            ts += table[f"time_since_begin_{index}"].to_list()
            ys += table[column].to_list()
            ys2 += table[column2].astype(float).to_list()
    df = pd.DataFrame(
        (np.array((ts, ys, ys2))).transpose(), columns=("ts", "ys", "ys2")
    )
    factor = 4
    df["ts_round"] = (df["ts"] / factor).astype(int) * factor
    meancurve = df.groupby("ts_round")["ys"].mean()
    ax.plot(meancurve.index, meancurve, label="mean", color="black")
    meancurve2 = df.groupby("ts_round")["ys2"].mean()
    # ax2.plot(meancurve.index, meancurve2, label=plate, color = 'red')
    ax.set_xlim((-30, 30))
    ax2.set_ylim((0, 0.25))
    ax.set_ylim((0, 30000))

    ax.set_ylabel("network biovolume density ($\mu^{3} m.mm^{-2}$)")
    ax.set_xlabel("shifted time ($h$)")
    ax2.set_ylabel("active tips density ($mm^{-2}$)")
    ax.tick_params(axis="y", colors="blue")
    ax2.tick_params(axis="y", colors="red")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if not savefig is None:
        plt.savefig(savefig)
    return (Cs, lambs, ds, indexes, t0s, meancurve, meancurve2)
