import numpy as np
from util import get_path, get_dates_datetime
import pickle
from directory import directory
from hyphae_id_surf import clean_and_relabel, get_mother, save_hyphaes, resolve_ambiguity_two_ends,solve_degree4, clean_obvious_fake_tips, get_pixel_growth_and_new_children
from experiment_class_surf import Edge,Node
from random import choice
import networkx as nx
from directory import directory, path_code
from scipy import sparse
from hyphae_id_surf import clean_and_relabel, get_mother, save_hyphaes, resolve_ambiguity_two_ends,solve_degree4, clean_obvious_fake_tips, get_pixel_growth_and_new_children

def get_time(exp,t,tp1):
    seconds = (exp.dates[tp1]-exp.dates[t]).total_seconds()
    return(seconds/3600)

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

def get_length_um_edge(edge,t):
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

def get_length_um_node_list(node_list,exp,t):
    total_length = 0
    for i in range(len(node_list)-1):
        nodea=Node(node_list[i],exp)
        nodeb=Node(node_list[i+1],exp)
        edge_obj =Edge(nodea,nodeb,exp)
        total_length += get_length_um_edge(edge_obj,t)
    return(total_length)

def get_exp(inst,directory=directory):
    plate = inst[0]
    begin = inst[1]
    end = inst[2]
    dates_datetime = get_dates_datetime(directory,plate)
    print('begin =',dates_datetime[begin],'\n  end =',dates_datetime[end])
    dates_datetime_chosen=dates_datetime[begin:end+1]
    dates = dates_datetime_chosen
    exp = pickle.load( open(f'{directory}/Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/experiment_{plate}.pick', "rb" ) )
    return(exp)

def get_hyph_infos(exp):
    select_hyph = {}
    for hyph in exp.hyphaes:
        select_hyph[hyph] = []
        for i,t in enumerate(hyph.ts[:-1]):
            tp1=hyph.ts[i+1]
            pixels,nodes = get_pixel_growth_and_new_children(hyph,t,tp1)
            speed = np.sum([get_length_um(seg) for seg in pixels])/get_time(exp,t,tp1)
            select_hyph[hyph].append((t,hyph.ts[i+1],speed,pixels))
    return(select_hyph)

def get_rh_bas(exp,criter):
    select_hyph = get_hyph_infos(exp)
    max_speeds = []
    total_growths = []
    lengths = []
    branch_frequ = []
    hyph_l = []
    RH=[]
    BAS=[]
    for hyph in exp.hyphaes:
        speeds = [c[2] for c in select_hyph[hyph]]
        ts = [c[0] for c in select_hyph[hyph]]
        tp1s = [c[1] for c in select_hyph[hyph]]
        if len(speeds)>0:
            length = hyph.get_length_um(hyph.ts[-1])
            nodes = hyph.get_nodes_within(hyph.ts[-1])[0]
            max_speed = np.max(speeds)
            total_growth = np.sum([speed * get_time(exp,ts[i],tp1s[i]) for i,speed in enumerate(speeds)])
            if criter(max_speed,length):
                RH.append(hyph)
            else:
                BAS.append(hyph)
            lengths.append(length)
            max_speeds.append(max_speed)
            total_growths.append(total_growth)
            branch_frequ.append((len(nodes)-1)/(length+1))
            hyph_l.append(hyph)
        else:
            BAS.append(hyph)            
    return(RH, BAS, max_speeds, total_growths, lengths, branch_frequ,select_hyph)
a= 0.0005
b= 0.01
thresh = 2

def estimate_bas_freq_mult(insts,samples,back,criter):
    bas_frequs=[]
    for inst in insts:
        bas_frequ=[]
        t0 = inst[2]-inst[1]-back
        exp = get_exp(inst)
        RH, BAS, max_growths, total_growths, lengths, branch_frequ, select_hyph = get_rh_bas(exp,criter)
        bas_roots = [hyph.root for hyph in BAS]
        for k in range(samples):
            node1 = Node(choice(list(exp.nx_graph[t0].nodes)),exp)
            node2 = Node(choice(list(exp.nx_graph[t0].nodes)),exp)
            if np.linalg.norm(node1.pos(t0)-node2.pos(t0))>=5000:
                nodes = nx.shortest_path(exp.nx_graph[t0], source = node1.label, target = node2.label)
    #             exp.plot([t],[nodes])
                bass=[]
                for node in nodes:
                    if exp.get_node(node) in  bas_roots:
                        bass.append(node)
            #             print(hyph)
            #             if hyph.ts[0] in rh.ts:
            #                 print(hyph,hyph_info[hyph],hyph in BAS)
            #                 hyph.end.show_source_image(hyph.ts[-1],hyph.ts[-1])
                bas_frequ.append(len(bass)/get_length_um_node_list(nodes,exp,t0)*10000)
        pickle.dump(bas_frequ, open(f'{path_code}/MscThesis/Results/bas_{inst}.pick', "wb"))
        bas_frequs.append(bas_frequ)
    return(bas_frequs)

def get_curvature_density(inst,window):
    exp = get_exp(inst)
    skeletons = [sparse.csr_matrix(skel) for skel in exp.skeletons]
    RH, BAS, max_growths, total_growths, lengths, branch_frequ, select_hyph = get_rh_bas(exp,criter)
    angles = []
    curvatures = []
    densities = []
    growths = []
    speeds = []
    tortuosities = []
    hyphs = []
    ts = []
    print(f'There is {len(RH)} RH')
    for hyph in RH:
        for i,t in enumerate(hyph.ts[:-1]):
            tp1=hyph.ts[i+1]
            segs,nodes = get_pixel_growth_and_new_children(hyph,t,tp1)
            speed = np.sum([get_length_um(seg) for seg in segs])/get_time(exp,t,tp1)
            total_growth = speed * get_time(exp,t,tp1)
            curve = [pixel for seg in segs for pixel in seg]
            pos = hyph.end.pos(t)
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
                begin = vec1/np.linalg.norm(vec1)
                end = vec2/np.linalg.norm(vec2)
                dot_product = min(np.dot(begin, end),1)
                if (begin[1] * end[0] - end[1] * begin[0] >= 0):  # determinant
                    angle = -np.arccos(dot_product) / (2 * np.pi) * 360
                else:
                    angle = np.arccos(dot_product) / (2 * np.pi) * 360
                inv_tortuosity = (straight_distance*1.725)/total_growth
                if  speed>=100 and speed <400 and hyph.end.degree(tp1)<3 and inv_tortuosity>0.8 and inv_tortuosity<1.1:
                    if np.isnan((angle/total_growth)):
                        print(angle,total_growth,dot_product)
                    angles.append(angle)
                    curvatures.append(angle/total_growth)
                    densities.append(density)
                    growths.append(total_growth)
                    speeds.append(speed)
                    tortuosities.append(inv_tortuosity)
                    ts.append(get_time(exp,0,t))
                    hyphs.append(hyph.end.label)
    return(angles, curvatures, densities,growths,speeds,tortuosities,ts,hyphs)

def estimate_growth(inst,criter):
    exp = get_exp(inst)
    RH, BAS, max_growths, total_growths, lengths, branch_frequ, select_hyph = get_rh_bas(exp,criter)
    print(inst,len(RH))
    group = (max_growths,total_growths ,lengths)
    return(group)

def criter(max_growth,length):
    return(a*length+b*max_growth>=thresh and max_growth>=50)