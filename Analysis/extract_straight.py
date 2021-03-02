import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
from Analysis.util import *
from sample.paths.directory import path_code
from Analysis.data_info import *
window=800
for treatment in treatments.keys():
    insts = treatments[treatment]
    for inst in insts:
        result = get_curvature_density(inst,window)
        pickle.dump(result, open(f'{path_code}/MscThesis/Results/straight_{window}_{inst}.pick', "wb"))