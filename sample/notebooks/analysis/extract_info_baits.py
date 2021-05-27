import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
sys.path.append( '/home/cbisot/pycode/MscThesis/sample/pipeline/functions')

from sample.notebooks.analysis.util import *
from sample.paths.directory import path_code, directory_scratch, directory_project
from sample.notebooks.analysis.data_info import *
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import matplotlib.pyplot as plt
import cv2
import imageio

path_code = "/home/cbisot/pycode/"
plate_info = pd.read_excel(path_code + 'MscThesis/plate_info/SummaryAnalizedPlates.xlsx',engine='openpyxl',header=3,)
pixel_conversion_factor = 1.725
window = 800
def get_pos_nut(inst, t, nutrient):
    plate_num = plate_number[inst]
    side_P = get_Pside(plate_num).values[0]
    if nutrient == 'P':
        side_nut = side_P
    else:
        side_nut ='left' if side_P=='right' else 'left'
    if side_nut not in ['left','right']:
        side_nut = 'right'
    if inst in bait_positions.keys():
        pos_nut = bait_positions[inst][t][side_nut]
    else:
        pos_nut = bait_positions[(10,0,10)][0][side_nut]
    return(pos_nut,side_nut)

bait_positions={}
for treatment in ['25','baits']:
    insts = treatments[treatment]
    for inst in insts:
        bait_positions[inst] = pickle.load(open(f'{path_code}/MscThesis/Results/baits_{inst}.pick', "rb"))
        
def get_Pside(plate_number):
    return(plate_info.loc[plate_info['Plate #'] == plate_number]['P-bait'])

def get_angle(veca,vecb):
    begin = veca/np.linalg.norm(veca)
    end = vecb/np.linalg.norm(vecb)
    dot_product = min(np.dot(begin, end),1)
    if (begin[1] * end[0] - end[1] * begin[0] >= 0):  # determinant
        angle = -np.arccos(dot_product) / (2 * np.pi) * 360
    else:
        angle = np.arccos(dot_product) / (2 * np.pi) * 360
    return(angle)

def get_density_maps(exp,t,compress,kern_sizes):
    skeletons = [sparse.csr_matrix(skel) for skel in exp.skeletons]
    window=compress
    densities=np.zeros((skeletons[t].shape[0]//compress,skeletons[t].shape[1]//compress),dtype=np.float)
    for xx in range(skeletons[t].shape[0]//compress):
        for yy in range(skeletons[t].shape[1]//compress):
            x = xx*compress
            y = yy*compress
            skeleton=skeletons[t][x-window:x+window,y-window:y+window]
            density = skeleton.count_nonzero()/((window*1.725)**2)
            densities[xx,yy]=density
    results = {}
    for kern_size in kern_sizes:
        density_filtered = gaussian_filter(densities,kern_size)
        sx = ndimage.sobel(density_filtered ,axis=0,mode='constant')
        sy = ndimage.sobel(density_filtered ,axis=1,mode='constant')
        sobel=np.hypot(sx,sy)
        results[kern_size] = density_filtered,sx,sy,sobel
    return(results)

def get_curvature_density_bait(window, path,kern_sizes=[]):
    column_names = ["plate","inst", "treatment", "angle", "curvature","density","growth","speed",
                    "straightness","dist_P","dist_N","angle_to_P","angle_to_N","t","hyph",'x','y','vx','vy','xinit','yinit']
    for kern_size in kern_sizes:
        column_names.append(f'density{kern_size}')
        column_names.append(f'grad_density_x{kern_size}')
        column_names.append(f'grad_density_y{kern_size}')
        column_names.append(f'grad_density_norm{kern_size}')
        column_names.append(f'area{kern_size}')
    compress = 100
    infos = pd.DataFrame(columns=column_names)
    for treatment in ['25*','25','baits']:
        insts = treatments[treatment]
        for inst in insts:
            exp = get_exp(inst,path)
            exp.load_compressed_skel()
            density_maps = [get_density_maps(exp,t,compress,kern_sizes) for t in range(exp.ts)]
            for index,density_map in enumerate(density_maps):
                plt.close('all')
                fig=plt.figure(figsize=(14,12))
                ax = fig.add_subplot(111)
                im= density_map[kern_size][0]
                figure = ax.imshow(im>=0.0005,vmax = 0.01)
                plt.colorbar(figure,orientation = 'horizontal')
                save = f'/home/cbisot/pycode/MscThesis/sample/notebooks/plotting/Figure/im*{index}.png'
                plt.savefig(save)
            img_array = []
            for index in range(len(density_maps)):
                img = cv2.imread(f'/home/cbisot/pycode/MscThesis/sample/notebooks/plotting/Figure/im*{index}.png')
                img_array.append(img)
            imageio.mimsave(f'/home/cbisot/pycode/MscThesis/sample/notebooks/plotting/Figure/movie_dense_{kern_size}_{plate_number[inst]}_thresh.gif', img_array,duration = 1)            
            for index,density_map in enumerate(density_maps):
                plt.close('all')
                fig=plt.figure(figsize=(14,12))
                ax = fig.add_subplot(111)
                im= density_map[kern_size][0]
                figure = ax.imshow(im,vmax = 0.01)
                plt.colorbar(figure,orientation = 'horizontal')
                save = f'/home/cbisot/pycode/MscThesis/sample/notebooks/plotting/Figure/im*{index}.png'
                plt.savefig(save)
            img_array = []
            for index in range(len(density_maps)):
                img = cv2.imread(f'/home/cbisot/pycode/MscThesis/sample/notebooks/plotting/Figure/im*{index}.png')
                img_array.append(img)
            imageio.mimsave(f'/home/cbisot/pycode/MscThesis/sample/notebooks/plotting/Figure/movie_dense_{kern_size}_{plate_number[inst]}.gif', img_array,duration = 1)
            skeletons = [sparse.csr_matrix(skel) for skel in exp.skeletons]
            RH, BAS, max_speeds, total_growths, widths_sp, lengths, branch_frequ,select_hyph = get_rh_bas(exp,criter)
            for hyph in RH:
                for i,t in enumerate(hyph.ts[:-1]):
                    pos_P, side_P = get_pos_nut(inst,t,'P')
                    pos_N, side_N = get_pos_nut(inst,t,'N')
                    tp1=hyph.ts[i+1]
                    segs,nodes = get_pixel_growth_and_new_children(hyph,t,tp1)
                    speed = np.sum([get_length_um(seg) for seg in segs])/get_time(exp,t,tp1)
                    total_growth = speed * get_time(exp,t,tp1)
                    curve = [pixel for seg in segs for pixel in seg]
                    pos = hyph.end.pos(t)
                    pos_tp1 = hyph.end.pos(tp1)
                    v_vector = np.array(pos_tp1)-np.array(pos)
                    vx = v_vector[0]
                    vy = v_vector[1]
                    dist_P = np.linalg.norm(np.array(pos)-np.array(pos_P))*pixel_conversion_factor
                    dist_N = np.linalg.norm(np.array(pos)-np.array(pos_N))*pixel_conversion_factor
                    x,y = pos[0],pos[1]
                    straight_distance = np.linalg.norm(hyph.end.pos(t)-hyph.end.pos(tp1))
                    skeleton=skeletons[t][x-window:x+window,y-window:y+window]
                    density = skeleton.count_nonzero()/(window*1.725)
                    curve_array = np.array(curve)
                    res = 50
                    index = min(res,curve_array.shape[0]-1)
                    vec1 = curve_array [index]-curve_array [0]
                    vec2 = curve_array [-1]-curve_array [-index-1]
                    if np.linalg.norm(vec1)>0 and np.linalg.norm(vec2)>0:
                        angle = get_angle(vec1,vec2)
                        vec_growth =  v_vector
                        vec_to_P = np.array(pos_P)-np.array(pos)
                        angle_to_P = get_angle(vec_growth,vec_to_P)
                        vec_to_N = np.array(pos_N)-np.array(pos)
                        angle_to_N = get_angle(vec_growth,vec_to_N)
                        inv_tortuosity = (straight_distance*1.725)/total_growth
                        if hyph.end.degree(tp1)<3 and inv_tortuosity>0.8 and inv_tortuosity<1.1:
                            if np.isnan((angle/total_growth)):
                                print(angle,total_growth,dot_product)
                            new_line_dic = {   "plate": [plate_number[inst]],
                                    "inst": [inst],
                                    "treatment": [treatment],
                                    "angle": [angle],
                                    "curvature": [angle/total_growth],
                                    "density": [density],
                                    "growth": [total_growth],
                                    "speed": [speed],
                                    "straightness": [inv_tortuosity],
                                     "t": [get_time(exp,0,t)],
                                     "hyph": [hyph.end.label],
                                    "dist_P":[dist_P],
                                     "dist_N":[dist_N],
                                    "angle_to_P":[angle_to_P],
                                    "angle_to_N":[angle_to_N],
                                    "x":[x],
                                    "y":[y], 
                                    "vx":[vx],
                                    "vy" : [vy],
                                    "xinit" : [vec1[0]],
                                    "yinit" : [vec1[1]]
                                }
                            for kern_size in kern_sizes:
                                pos_comp = x//compress,y//compress
                                new_line_dic[f'density{kern_size}'] = density_maps[t][kern_size][0][pos_comp]
                                new_line_dic[f'grad_density_x{kern_size}'] = density_maps[t][kern_size][1][pos_comp]
                                new_line_dic[f'grad_density_y{kern_size}'] = density_maps[t][kern_size][2][pos_comp]
                                new_line_dic[f'grad_density_norm{kern_size}'] = density_maps[t][kern_size][3][pos_comp]
                                new_line_dic[f'area{kern_size}'] = np.sum(density_maps[t][kern_size][0]>=0.0005)*compress**2
                            new_line = pd.DataFrame(new_line_dic

                            )  # index 0 for
                            # mothers need to be modified to resolve multi mother issue
                            infos = infos.append(new_line, ignore_index=True)
    return(infos)

infos = get_curvature_density_bait(window, directory_project,[5,10,20])
pickle.dump(infos, open(f'{path_code}/MscThesis/Results/straight_bait_{window}.pick', "wb"))