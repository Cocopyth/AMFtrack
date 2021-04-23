import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
sys.path.append( '/home/cbisot/pycode/MscThesis/sample/pipeline/functions')

from sample.notebooks.analysis.util import *
from sample.paths.directory import path_code, directory_scratch, directory_project
from sample.notebooks.analysis.data_info import *

for treatment in treatments.keys():
    insts = treatments[treatment]
    for inst in insts:
        angles = estimate_angle(inst,criter, directory_project)
        pickle.dump(angles, open(f'{path_code}/MscThesis/Results/angle_{inst}.pick', "wb"))