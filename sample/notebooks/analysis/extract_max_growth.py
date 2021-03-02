import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
from sample.notebooks.analysis.util import *
from sample.paths.directory import path_code
from sample.notebooks.analysis.data_info import *

for treatment in treatments.keys():
    insts = treatments[treatment]
    for inst in insts:
        result = estimate_growth(inst, criter)
        pickle.dump(result, open(f'{path_code}/MscThesis/Results/maxgrowth_{inst}.pick', "wb"))