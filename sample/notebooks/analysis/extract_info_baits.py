import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
sys.path.append( '/home/cbisot/pycode/MscThesis/sample/pipeline/functions')

from sample.notebooks.analysis.util import *
from sample.paths.directory import path_code, directory_scratch, directory_project
from sample.notebooks.analysis.data_info import *
import pandas as pd

path_code = "/home/cbisot/pycode/"
plate_info = pd.read_excel(path_code + 'MscThesis/plate_info/SummaryAnalizedPlates.xlsx',engine='openpyxl',header=3,)

window = 800
def get_pos_nut(inst, t, nutrient):
    plate_num = plate_number[inst]
    side_P = get_Pside(plate_num).values[0]
    if nutrient == 'P':
        side_nut = side_P
    else:
        side_nut ='left' if side_P=='right' else 'left'
    if side_nut == 'none':
        side_nut = 'right'
    pos_nut = bait_positions[inst][t][side_nut]
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

def get_curvature_density_bait(window, path,pos_baits):
    column_names = ["plate","inst", "treatment", "angle", "curvature","density","growth","speed",
                    "straightness","dist_P","dist_N","angle_to_P","angle_to_N","t","hyph"]
    infos = pd.DataFrame(columns=column_names)
    for treatment in ['25','baits']:
        insts = treatments[treatment]
        for inst in insts:
            exp = get_exp(inst,path)
            skeletons = [sparse.csr_matrix(skel) for skel in exp.skeletons]
            RH, BAS, max_speeds, total_growths, widths_sp, lengths, branch_frequ,select_hyph = get_rh_bas(exp,criter)
            pos_baits = bait_positions[inst]
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
                    dist_P = np.linalg.norm(np.array(pos)-np.array(pos_P))
                    dist_N = np.linalg.norm(np.array(pos)-np.array(pos_N))
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
                        vec_growth =  (vec1+vec2)/2
                        vec_to_P = np.array(pos_P)-np.array(pos)
                        angle_to_P = get_angle(vec_growth,vec_to_P)
                        vec_to_N = np.array(pos_N)-np.array(pos)
                        angle_to_N = get_angle(vec_growth,vec_to_N)
                        inv_tortuosity = (straight_distance*1.725)/total_growth
                        if hyph.end.degree(tp1)<3 and inv_tortuosity>0.8 and inv_tortuosity<1.1:
                            if np.isnan((angle/total_growth)):
                                print(angle,total_growth,dot_product)
                            new_line = pd.DataFrame(
                                {   "plate": [plate_number[inst]],
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
                                    "angle_to_N":[angle_to_N]
                                }
                            )  # index 0 for
                            # mothers need to be modified to resolve multi mother issue
                            infos = infos.append(new_line, ignore_index=True)
    return(infos)

infos = get_curvature_density_bait(window, directory_project,bait_positions)
pickle.dump(infos, open(f'{path_code}/MscThesis/Results/straight_bait_{window}.pick', "wb"))