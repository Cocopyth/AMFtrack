import sys

sys.path.insert(0, "/home/cbisot/pycode/MscThesis/")
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
plt.rcParams.update(
    {"font.family": "verdana", "font.weight": "normal", "font.size": 20}
)
from amftrack.pipeline.launching.run_super import *
from amftrack.util.sys import *
from amftrack.notebooks.post_processing.util import *
import pickle


def sum_of_lists(N):

    directory = directory_project
    update_analysis_info(directory)
    analysis_info = get_analysis_info(directory)
    select = analysis_info
    num = 1
    rows = [row for (index, row) in select.iterrows()]
    for index, row in enumerate(rows):
        path = f'{directory}{row["folder_analysis"]}'
        print(index, row["Plate"])
        try:
            a = np.load(f"{path}/center.npy")
        except:
            print(index, row["Plate"])
        if index == num:
            path_exp = f'{directory}{row["path_exp"]}'
            exp = pickle.load(open(path_exp, "rb"))
            exp.dates.sort()
            break
