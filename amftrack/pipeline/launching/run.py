import os
import pandas as pd
from amftrack.util.sys import (
    path_code,
    update_plate_info,
    get_current_folders,
)
from amftrack.util.dbx import temp_path
from typing import List

from time import time_ns
from tqdm.autonotebook import tqdm
import subprocess


def make_stitching_loop(directory: str, dirname: str, op_id: int) -> None:
    """
    For the acquisition directory `directory`, prepares the stitching script used to stitch the image.
    """
    # TODO(FK): handle the case where there is no stiching_loops directory
    a_file = open(
        os.path.join(path_code, "pipeline/scripts/stitching_loops/stitching_loop.ijm"),
        "r",
    )

    list_of_lines = a_file.readlines()
    list_of_lines[4] = f"mainDirectory = \u0022{directory}\u0022 ;\n"
    list_of_lines[29] = f"\t if(startsWith(list[i],\u0022{dirname}\u0022)) \u007b\n"

    file_name = f"{temp_path}/stitching_loops/stitching_loop{op_id}.ijm"
    a_file = open(file_name, "w")
    a_file.writelines(list_of_lines)
    a_file.close()


def run_stitch(directory: str, folders: pd.DataFrame) -> None:
    """
    Runs the stiching loop.
    :param directory: is the folder containing the directories for each acquisition
    :param folder: is a pandas dataframe with information on the folder
    """
    # TODO(FK): Should encapsulate and factorize to have a version working with str instead of pd frame
    folder_list = list(folders["folder"])
    folder_list.sort()
    with tqdm(total=len(folder_list), desc="stitched") as pbar:
        for folder in folder_list:
            op_id = time_ns()
            im = imageio.imread(f"{directory}/{folder}/Img/Img_r03_c05.tif")
            for x in range(1, 11):
                for y in range(1, 16):
                    strix = str(x) if x >= 10 else f"0{x}"
                    striy = str(y) if y >= 10 else f"0{y}"
                    path = f"{directory}/{folder}/Img/Img_r{strix}_c{striy}.tif"
                    if not os.path.isfile(path):
                        f = open(path, "w")
                    if os.path.getsize(path) == 0:
                        imageio.imwrite(path, im * 0)
            make_stitching_loop(directory, folder, op_id)
            command = [
                fiji_path,
                "--mem=8000m",
                "--headless",
                "--ij2",
                "--console",
                "-macro",
                f'{os.getenv("TEMP")}/stitching_loops/stitching_loop{op_id}.ijm',
            ]
            print(" ".join(command))
            process = subprocess.run(command)
            pbar.update(1)


def run(code: str, args: List, folders: pd.DataFrame) -> None:
    """
    Run the chosen script `code` localy.
    :param code: name of the script file such as "prune.py", it has to be in the image_processing file
    :param args: list of arguments used by the script
    """
    op_id = time_ns()
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    folder_list = list(folders["folder"])
    folder_list.sort()
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    with tqdm(total=len(folder_list), desc="stitched") as pbar:
        for index, folder in enumerate(folder_list):
            command = (
                ["python3", f"{path_code}pipeline/scripts/image_processing/{code}"]
                + args_str
                + [f"{op_id}", f"{index}"]
            )
            print(" ".join(command))
            process = subprocess.run(command)
            pbar.update(1)


if __name__ == "__main__":
    # directory = "/data/felix/width1/full_plates/"  # careful: must have the / at the end
    # # update_plate_info(directory)
    # folder_df = get_current_folders(directory)
    # folders = folder_df.loc[folder_df["Plate"] == "907"]
    # time = "3:00:00"
    # threshold = 0.1
    # args = [threshold, directory]
    # run("prune_skel.py", args, folders)

    directory = "/data/felix/width1/full_plates/"  # careful: must have the / at the end
    # update_plate_info(directory)
    folder_df = get_current_folders(directory)
    folders = folder_df.loc[folder_df["Plate"] == "907"]
    time = "2:00:00"
    args = [directory]
    run("extract_nx_graph.py", args, folders)
