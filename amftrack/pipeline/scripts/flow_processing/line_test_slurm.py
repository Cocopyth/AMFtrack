from amftrack.pipeline.development.high_mag_videos.kymo_class import *
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm, colors
from tqdm import tqdm
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    clean_degree_4,
)
import scipy
import scipy.stats as stats
# from path import path_code_dir
import sys
# sys.path.insert(0, path_code_dir)
import pandas as pd
from amftrack.util.sys import temp_path


class Line:
    """
    Creates a line object with a speed, offset and width. Lines are drawn in an image with gen_image. 
    Currently only gauss type lines work. These are all normalized such that the sum over space (x) = 1. 
    With this normalization we would expect the net transport over time graph to increase by one for every line that it crosses. 
    """
    def __init__(self,speed, offset, line_type="gauss", width=5, intens = 1):
        self.line_type=line_type
        self.width = width
        self.speed = speed
        self.direc = np.sign(self.speed)
        self.offset = offset
        return None
        
    def block(self, array, speed_offset, normalize=False):
        if not normalize:
            return (np.heaviside(array - speed_offset + self.width, 0.5) - np.heaviside(array - speed_offset - self.width, 0.5))
        else:
            out_arr = (np.heaviside(array - speed_offset + self.width, 0.5) - np.heaviside(array - speed_offset - self.width, 0.5))
            return out_arr / np.sum(out_arr)
        
    def gauss(self, array, speed_offset, normalize = False):
        return stats.norm.pdf(array, speed_offset, self.width)
        
    def gen_image(self, x_res, y_res, normalize = False):
        x_line = np.linspace(0, x_res-1, x_res)
        line_pos = np.linspace(self.offset, self.offset + self.speed*y_res, y_res)
        if self.line_type == "block":
            line_img = [self.block(x_line, v_offset, normalize=normalize) for v_offset in line_pos]
        elif self.line_type == "gauss":
            line_img = [self.gauss(x_line, v_offset, normalize=normalize) for v_offset in line_pos]
        return line_img
    
# def gen_feather(img, height, zero_frac):
#     zero_thresh = int(height * zero_frac)
#     feather = np.linspace(0,1, height - zero_thresh)
#     out_img = img.copy()
#     for i in range(zero_thresh):
#         out_img[-1-i] *= 0
#     for i, j in enumerate(range(zero_thresh, height)):
#         out_img[-1-j] *= feather[i]
#     return out_img

def kymo_titles(axis, title):
    axis.set_title(title)
    axis.set_xlabel("space (x)")
    axis.set_ylabel("time (t)")
    return None

def flux_int(img, line = 200):
    flux_max = np.max(abs(img.flatten()))
    [x_res, y_res] = np.shape(img)
    int_line = img.transpose()[line]
    int_plot = np.cumsum(int_line)
    fig, ax = plt.subplots(1,2)
    fig.tight_layout()
    
    ax[0].imshow(img, cmap='bwr', vmin = -1*flux_max, vmax = flux_max)
    ax[0].vlines(line, 0, x_res-1, colors='black')
    kymo_titles(ax[0], "Flux with integral line")
    
    ax[1].plot(int_plot, label="Ground Truth")
    ax[1].grid()
    ax[1].hlines(0, 0, x_res, colors='black', label="")
    ax[1].set_title("Net Transport")
    ax[1].set_xlabel("time (t)")
    ax[1].set_ylabel("transport (q)")
    ax[1].legend()
    return ax[1]


def noisy(noise_typ,image, gauss_params=[0, 0.1]):
    if noise_typ == "gauss":
        row,col= image.shape
        mean = gauss_params[0]
        var = gauss_params[1]
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)        
        noisy = image + image * gauss
        return noisy

def extract_net_transport(line, orig_img, recon_img, margin=10):
    int_line = orig_img.transpose()[line][margin:-margin]
    int_plot = np.cumsum(int_line)
    fourier_adj_int = np.nancumsum(recon_img.transpose()[line][margin:-margin])
    return np.array([int_plot[-1] - int_plot[0], fourier_adj_int[-1] - fourier_adj_int[0]])


# def do_line_test(x_res          = 300,
#                  y_res          = 400,
#                  nr_forw_lines  = 10, 
#                  nr_back_lines  = 10,
#                  nr_stat_lines  = 10,
#                  forw_intens    = 1.0,
#                  back_intens    = 1.0,
#                  stat_intens    = 1.0,
#                  forw_speed     = 1.0,
#                  back_speed     = 1.0,
#                  forw_width     = 1.0,
#                  back_width     = 1.0,
#                  stat_width     = 1.0,
#                  forw_speed_var = 1.0,
#                  back_speed_var = 1.0,
#                  forw_width_var = 1.0,
#                  back_width_var = 1.0,
#                  stat_width_var = 1.0, 
#                  noise_thresh   = 0.01, 
#                  speed_thresh   = 10, 
#                  display_figs   = False):
    
def do_line_test(
    pd,
    noise_thresh = 0.01,
    speed_thresh = 10,
    display_figs = False):
    
    x_res          = int(pd.loc['x_res'])
    y_res          = int(pd.loc['y_res'])
    nr_forw_lines  = int(pd.loc['nr_forw_lines'])
    nr_back_lines  = int(pd.loc['nr_back_lines'])
    nr_stat_lines  = int(pd.loc['nr_stat_lines'])
    forw_intens    = pd.loc['forw_intens']
    back_intens    = pd.loc['back_intens']
    stat_intens    = pd.loc['stat_intens']
    forw_speed     = pd.loc['forw_speed']
    back_speed     = pd.loc['back_speed']
    forw_width     = pd.loc['forw_width']
    back_width     = pd.loc['back_width']
    stat_width     = pd.loc['stat_width']
    forw_speed_var = pd.loc['forw_speed_var']
    back_speed_var = pd.loc['back_speed_var']
    forw_width_var = pd.loc['forw_width_var']
    back_width_var = pd.loc['back_width_var']
    stat_width_var = pd.loc['stat_width_var']
    
    img_stack = []
    img_forw_stack = []
    img_back_stack = []
    flux_stack = []
    flux_stack_back = []
    flux_stack_forw = []
    params_forw = np.array([
                    np.random.normal(forw_speed, forw_speed_var, size=nr_forw_lines),
                    np.random.uniform(-y_res, x_res, nr_forw_lines),
                    abs(np.random.normal(forw_width, forw_width_var, size=nr_forw_lines))
                  ]).T
    params_back = np.array([
                    np.random.normal(-1*back_speed, back_speed_var, size=nr_back_lines),
                    np.random.uniform(0, x_res + y_res, nr_back_lines),
                    abs(np.random.normal(back_width, back_width_var, size=nr_back_lines))
                  ]).T
#     params_back = np.random.rand(nr_back_lines, 3)
    params_stat = np.random.rand(nr_stat_lines, 3)
        
    for i in range(nr_forw_lines):
        params_proc = params_forw
        line = Line(params_proc[i][0], params_proc[i][1], width=params_proc[i][2])
        img_forw = np.array(line.gen_image(x_res, y_res, normalize=True)) * forw_intens
        img_stack.append(img_forw)
        img_forw_stack.append(img_forw)
        flux_img_forw = img_forw * params_proc[i][0]
        flux_stack_forw.append(flux_img_forw)
        flux_stack.append(flux_img_forw)
        
    for i in range(nr_back_lines):
        params_proc = params_back
        line = Line(params_proc[i][0], params_proc[i][1], width=params_proc[i][2])
        img_back = np.array(line.gen_image(x_res, y_res, normalize=True)) * back_intens
        img_stack.append(img_back)
        img_back_stack.append(img_back)
        flux_img_back = img_back * params_proc[i][0]
        flux_stack_back.append(flux_img_back)
        flux_stack.append(flux_img_back)
        
    for i in range(nr_stat_lines):
        params = np.random.rand(3)
        params_proc = (0, params_stat[i][1] * x_res, (params_stat[i][2]+stat_width) * stat_width_var)
        line = Line(params_proc[0], params_proc[1], width=params_proc[2])
        img_stat = np.array(line.gen_image(x_res, y_res, normalize=True)) * stat_intens
        img_stack.append(img_stat)
        flux_img_stat = img_stat * params_proc[0]
        flux_stack.append(flux_img_stat)
        
    tot_img       = np.sum(img_stack, axis=0)
    tot_img       = noisy("gauss", tot_img, gauss_params=[0, 0.0001])
    tot_img_forw  = np.sum(img_forw_stack, axis=0)
    tot_img_back  = np.sum(img_back_stack, axis=0)
    tot_flux      = np.sum(flux_stack, axis=0)
    tot_flux_forw = np.sum(flux_stack_forw, axis=0)
    tot_flux_back = np.sum(flux_stack_back, axis=0)
    img_max       = np.max(abs(tot_img.flatten()))
    flux_max      = np.max(abs(tot_flux.flatten()))
    kymo_anal     = KymoEdgeAnalysis(kymo = tot_img)
    forw_thresh, back_thresh = kymo_anal.fourier_kymo(1, test_plots=False)
#     forw_back = np.add(forw, back)
#     forw_thresh = forw
#     back_thresh = back 
#     forw_back_thresh = np.add(forw_thresh, back_thresh)
    speeds, times = kymo_anal.extract_speeds(7, w_start=5, C_thresh=0.95, C_thresh_falloff = 0.0, blur_size = 7, preblur=True, speed_thresh=speed_thresh, plots=False)
#     spd_max = np.nanmax(abs(speeds.flatten()))
    spds_forw = speeds[0][1]
    spds_back = speeds[0][0]
#     spds_tot = np.nansum(np.dstack((spds_back,spds_forw)),2)
    flux_tot = np.nansum((np.prod((spds_forw, forw_thresh[0]), 0), np.prod((spds_back, back_thresh[0]), 0)), 0)
    
    trans_plots = np.array([extract_net_transport(line, tot_flux, flux_tot) for line in range(5, flux_tot.shape[1] -5)]).transpose()
    
    if display_figs:    
        fig, ax = plt.subplots(1,2, sharey = True, figsize=(8,4))
        ax[0].imshow(tot_img)
        kymo_titles(ax[0], "Original kymo")
        # ax[1].imshow(feather_img)
        # kymo_titles(ax[1], "Feathered kymo")
        flx = ax[1].imshow(tot_flux, cmap='bwr', vmin = -1*flux_max, vmax = flux_max)
        kymo_titles(ax[1], "Flux kymo")
        cbar = plt.colorbar(flx, fraction=0.056, pad=0.02)
        cbar.ax.set_ylabel("Speed*Intensity", rotation=270)

        fig.tight_layout()

        fig, ax = plt.subplots(1,2, figsize=(8,4))
        for i in [0,1]:
            thi = ax[i].imshow([tot_flux_forw, tot_flux_back][i], cmap='bwr', vmin = -1*flux_max, vmax = flux_max)
            ax[i].set_title("Flux {}".format(["forward", "backward"][i]))

        cbar2 = plt.colorbar(thi, fraction=0.056, pad=0.02)
        cbar2.ax.set_ylabel("Speed*Intensity", rotation=270)

        fig.tight_layout()
#     print(f'Pointwise error mean \t{np.mean((trans_plots[1] - trans_plots[0])):.4}')
#     print(f'Pointwise error var \t {np.std((trans_plots[1] - trans_plots[0])):.4}')
    
    return np.mean(trans_plots[1]), np.mean(trans_plots[0])

print(str(sys.argv[1]))
directory = str(sys.argv[1])
test_len = str(sys.argv[2])
folders = str(sys.argv[3])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

dataframe = pd.read_json(f"{temp_path}/{op_id}.json")
dataframe = dataframe.iloc[i]
print(dataframe)

results = np.zeros(4)

t = {}
for j in tqdm(range(int(test_len))):
    t[f'test_{j}'] = do_line_test(dataframe, display_figs=False)
testframe = pd.DataFrame(t)
results[0] = testframe.mean(axis=1).iloc[0]
results[1] = testframe.std(axis=1).iloc[0]
results[2] = testframe.mean(axis=1).iloc[1]
results[3] = testframe.std(axis=1).iloc[1]

print(results)

results_pd = {
    'est_mean' : results[0],
    'est_var'  : results[1],
    'gtr_mean' : results[2],
    'gtr_var'  : results[3],
}
r_pd = pd.DataFrame(results_pd, index=[0])
r_pd.to_json(f"{directory}/{i}_of_{len(folders)} {op_id}.json")
testframe.to_json(f"{directory}/{op_id}_{i:04}_line_test.json")