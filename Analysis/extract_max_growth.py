import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
from Analysis.util import *
from sample.paths.directory import path_code
from Analysis.data_info import *

for treatment in treatments.keys():
    insts = treatments[treatment]
    for inst in insts:
        result = estimate_growth(inst, criter)
        pickle.dump(result, open(f'{path_code}/MscThesis/Results/maxgrowth_{inst}.pick', "wb"))