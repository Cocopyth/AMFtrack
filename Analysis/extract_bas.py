import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
from Analysis.util import *
from sample.paths.directory import path_code
from Analysis.data_info import *

for treatment in treatments.keys():
    insts = treatments[treatment]
    bas_frequs = estimate_bas_freq_mult(insts,1000,0,criter)
    pickle.dump(bas_frequs, open(f'{path_code}/MscThesis/Results/bas_{treatment}.pick', "wb"))