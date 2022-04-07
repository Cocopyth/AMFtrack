import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
sys.path.append( '/home/cbisot/pycode/MscThesis/amftrack/pipeline/functions')

from amftrack.notebooks.analysis.util import *
from amftrack.pipeline.pipeline.paths.directory import path_code, directory_scratch, directory_project
from amftrack.notebooks.analysis.data_info import *

for treatment in treatments.keys():
    insts = treatments[treatment]
    for inst in insts:
        result = estimate_growth(inst, criter, directory_project)
        pickle.dump(result, open(f'{path_code}/MscThesis/Results/maxgrowth_{inst}.pick', "wb"))