import hashlib

from tqdm.autonotebook import tqdm
import dropbox

from zipfile import ZipFile, ZIP_DEFLATED
import os
import requests

# create a ZipFile object
from subprocess import call
from time import sleep
from decouple import Config, RepositoryEnv
from time import time_ns
import pandas as pd
import hashlib
import shutil
import numpy as np

DOTENV_FILE = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + "/local.env"
)
env_config = Config(RepositoryEnv(DOTENV_FILE))

path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"

app_key = env_config.get("APP_KEY")
app_secret = env_config.get("APP_SECRET")
refresh_token = env_config.get("REFRESH_TOKEN")
folder_id = env_config.get("FOLDER_ID")
user_id = env_config.get("USER_ID")
temp_path = env_config.get("TEMP_PATH")
DROPBOX_HASH_CHUNK_SIZE = 4 * 1024 * 1024


def load_dbx():
    dbx = dropbox.DropboxTeam(
        app_key=app_key, app_secret=app_secret, oauth2_refresh_token=refresh_token
    )
    p = dropbox.common.PathRoot.namespace_id(folder_id)
    dbx = dbx.with_path_root(p)
    dbx = dbx.as_user(user_id)
    return dbx


def zip_file(origin, target, depth=2):
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as zipObj:
        # Iterate over all the files in directory
        tot = 0
        for folderName, subfolders, filenames in os.walk(origin):
            for filename in filenames:
                tot += 1
        with tqdm(total=tot, desc="zipping", leave=False) as pbar:
            for folderName, subfolders, filenames in os.walk(origin):
                for filename in filenames:
                    # create complete filepath of file in directory
                    filePath = os.path.join(folderName, filename)
                    filePath = os.path.normpath(filePath)
                    # Add file to zip
                    place = filePath.replace(origin, "")
                    zipObj.write(filePath, place)
                    pbar.update(1)


def unzip_file(origin, target, depth=2):
    with ZipFile(origin, "r") as zipy:
        zipy.extractall(target)


def upload(file_path, target_path, chunk_size=4 * 1024 * 1024, catch_exception=True):
    """Uploads a file placed in file path to a target path in dropbox. The dropbox path is relative to root
     of the team folder and should start with \
     If catch exception is True, the function runs in no error mode and retry uploading if uploading fails"""
    # TODO (CB) rewrite the catch exception with a decorator to make it more elegant
    dbx = load_dbx()
    with open(file_path, "rb") as f:
        file_size = os.path.getsize(file_path)
        while True:
            try:
                if file_size <= chunk_size:
                    dbx.files_upload(
                        f.read(), target_path, mode=dropbox.files.WriteMode.overwrite
                    )
                    # Overwriting files by default
                else:

                    with tqdm(total=file_size, desc="Uploaded", leave=False) as pbar:
                        upload_session_start_result = dbx.files_upload_session_start(
                            f.read(chunk_size)
                        )
                        pbar.update(chunk_size)
                        cursor = dropbox.files.UploadSessionCursor(
                            session_id=upload_session_start_result.session_id,
                            offset=f.tell(),
                        )
                        commit = dropbox.files.CommitInfo(
                            path=target_path, mode=dropbox.files.WriteMode.overwrite
                        )  # Overwriting files by default
                        while f.tell() < file_size:
                            if (file_size - f.tell()) <= chunk_size:
                                dbx.files_upload_session_finish(
                                    f.read(chunk_size), cursor, commit
                                )
                            else:
                                dbx.files_upload_session_append(
                                    f.read(chunk_size),
                                    cursor.session_id,
                                    cursor.offset,
                                )
                                cursor.offset = f.tell()
                            pbar.update(chunk_size)
            except (
                requests.exceptions.RequestException,
                dropbox.exceptions.ApiError,
            ) as e:
                if catch_exception:
                    print("error")
                    sleep(60)
                    continue
                else:
                    print(e)
                    return e
            break


def upload_folder(path, target_drop, delete=False):
    for root, dirs, files in os.walk(path):
        for filename in files:
            # construct the full local path
            local_path = os.path.join(root, filename)

            # construct the full Dropbox path
            relative_path = os.path.relpath(local_path, path)

            dropbox_path = os.path.join(target_drop, relative_path)
            dropbox_path = dropbox_path.replace("\\", "/")
            upload(local_path, dropbox_path, catch_exception=False)
            if delete:
                os.remove(local_path)
    if delete:
        shutil.rmtree(path)


def get_size_dbx(path):
    dbx = load_dbx()
    response = dbx.files_list_folder(path, recursive=True)
    listfiles = []
    while response.has_more:
        listfiles += [
            file
            for file in response.entries
            if type(file) == dropbox.files.FileMetadata
        ]
        response = dbx.files_list_folder_continue(response.cursor)
    listfiles += [file for file in response.entries]
    size = sum(file.size for file in listfiles)
    return size / 10**9


def download(file_path, target_path, end="", catch_exception=True, unzip=False):
    """
    Downloads a file placed in file path on dropbox to a local target path. The dropbox path is relative to root
     of the team folder and should start with \
     If catch exception is True, the function runs in no error mode and retry uploading if uploading fails
     """
    # TODO (CB) rewrite the catch exception with a decorator to make it more elegant
    dbx = load_dbx()
    while True:
        try:
            with open(target_path, "wb") as f:
                metadata, res = dbx.files_download(path=f"{file_path}{end}")
                f.write(res.content)
        except (requests.exceptions.RequestException, dropbox.exceptions.ApiError) as e:
            if catch_exception:
                print("error")
                sleep(60)
                continue
            else:
                print(e)
                return e
        break
    if unzip:
        path_snap = target_path.split(".")[0]
        unzip_file(target_path, path_snap)
        os.remove(target_path)


class UploadError(Exception):
    pass


def upload_zip(path_total, target, catch_exception=True, delete=False):
    """
    Handles upload of files and folders taking only a path as an argument. If the total path is a
    file it is uploaded directly. Otherwise the folder is zipped and uploaded. Upload of individual files is
    inneficient.
    """
    is_dir = os.path.isdir(path_total)
    if is_dir:
        stamp = time_ns()
        path_zip = f"{temp_path}/{stamp}.zip"
        zip_file(path_total, path_zip)
        upload(
            path_zip,
            target,
            chunk_size=256 * 1024 * 1024,
            catch_exception=catch_exception,
        )
        path_final = path_zip

    else:
        upload(
            path_total,
            target,
            chunk_size=256 * 1024 * 1024,
            catch_exception=catch_exception,
        )
        path_final = path_total
    if delete and path_total.split("/")[-1] != "param.m":
        dbx = load_dbx()
        response = dbx.files_get_metadata(target)
        hash_drop = response.content_hash
        hash_local = compute_dropbox_hash(path_final)
        if hash_drop == hash_local:
            if is_dir:
                shutil.rmtree(path_total)
            else:
                os.remove(path_total)
        else:
            raise (UploadError)
    if is_dir:
        os.remove(path_zip)


def sync_fold(origin, target):
    """
    Launches a sync between two directory. Will only work on linux system.
    """
    cmd = f"rsync --update -avh {origin} {target}"
    # print(cmd)
    call(cmd, shell=True)



def get_dropbox_folders(dir_drop: str, skip_size: bool = True) -> pd.DataFrame:
    print("go")
    dbx = load_dbx()
    response = dbx.files_list_folder(dir_drop, recursive=True)
    # for fil in response.entries:
    listfiles = []
    folders_interest = ["param.m", "experiment.pick"]
    while response.has_more:
        listfiles += [
            file
            for file in response.entries
            if file.name.split("/")[-1] in folders_interest
        ]
        response = dbx.files_list_folder_continue(response.cursor)
    listfiles += [file for file in response.entries if file.name in folders_interest]
    # print([((file.path_lower.split(".")[0]) + "_info.json") for file in listfiles if (file.name.split(".")[-1] == "zip") &
    #        (((file.path_lower.split(".")[0]) + "_info.json") not in listjson)])
    listfiles.reverse()
    names = [file.path_display.split("/")[-2] for file in listfiles]
    path_drop = [
        os.path.join(*file.path_lower.split('/')[:-1]) for file in listfiles
    ]
    print(path_drop)
    id_uniques = [path.split(os.path.sep)[-2] for path in path_drop]

    path_drop = [
        path for k, path in enumerate(path_drop) if len(id_uniques[k].split("_")) >= 2
    ]
    names = [name for k, name in enumerate(names) if len(id_uniques[k].split("_")) >= 2]

    id_uniques = [idi for idi in id_uniques if len(idi.split("_")) >= 2]
    plate_num = [idi.split("_")[0] for idi in id_uniques]
    date_cross = [idi.split("_")[1] for idi in id_uniques]
    if skip_size:
        sizes = [np.nan for path in path_drop]
    else:
        sizes = []
        with tqdm(total=len(path_drop), desc="sizes") as pbar:
            for path in path_drop:
                try:
                    sizes.append(get_size_dbx("/" + path))
                except:
                    sizes.append(None)
                pbar.update(1)
    # modified = [file.client_modified for file in listfiles]
    modified = [0 for file in listfiles]

    df = pd.DataFrame(
        (names, sizes, modified, path_drop, plate_num, date_cross, id_uniques)
    ).transpose()
    df = df.rename(
        columns={
            0: "folder",
            1: "size",
            2: "change_date",
            3: "tot_path_drop",
            4: "Plate",
            5: "CrossDate",
            6: "unique_id",
        }
    )
    return df


def save_dropbox_state(dir_drop: str, skip_size: bool = True):
    df = get_dropbox_folders(dir_drop, skip_size)
    source = os.path.join(f'{os.getenv("TEMP")}', f"dropbox_info.json")
    df.to_json(source)
    target = f"{dir_drop}/folder_info.json"
    upload(
        source,
        target,
        chunk_size=256 * 1024 * 1024,
    )

def read_saved_dropbox_state(dir_drop: str, skip_size: bool = True):
    target = os.path.join(f'{os.getenv("TEMP")}', f"dropbox_info.json")
    source = f"{dir_drop}/folder_info.json"
    download(
        source,
        target,
    )
    df = pd.read_json(target)
    return(df)


def upload_folders(
    folders: pd.DataFrame, dir_drop="DATA", catch_exception=True, delete=False
):
    """
    Upload all the folders in the dataframe to a location on dropbox
    """
    run_info = folders.copy()
    folder_list = list(run_info["folder"])
    with tqdm(total=len(folder_list), desc="transferred") as pbar:
        for folder in folder_list:
            directory_name = folder
            line = run_info.loc[run_info["folder"] == directory_name]
            try:
                id_unique = (
                    str(int(line["Plate"].iloc[0]))
                    + "_"
                    + str(int(str(line["CrossDate"].iloc[0]).replace("'", "")))
                )
            except:
                id_unique = "faulty_param_files"
            path_snap = line["total_path"].iloc[0]
            for subfolder in os.listdir(path_snap):
                path_total = os.path.join(path_snap, subfolder)
                suffix = ".zip" if os.path.isdir(path_total) else ""
                target = f"/{dir_drop}/{id_unique}/{directory_name}/{subfolder}{suffix}"
                upload_zip(
                    path_total, target, catch_exception=catch_exception, delete=delete
                )
            pbar.update(1)
            if delete:
                if len(os.listdir(path_snap)) == 1:
                    os.remove(os.path.join(path_snap, "param.m"))
                    os.rmdir(path_snap)


def download_folders_old(folders_drop: pd.DataFrame, directory_target):
    for index, row in folders_drop.iterrows():
        path = row["tot_path_drop"]
        folder = row["folder"]
        download(path, os.path.join(directory_target, f"{folder}.zip"), unzip=True)


def download_folders_drop(folders_drop: pd.DataFrame, directory_target):
    dbx = load_dbx()
    for index, row in folders_drop.iterrows():
        path = "/" + row["tot_path_drop"]
        response = dbx.files_list_folder(path, recursive=False)
        listfiles = []
        while response.has_more:
            listfiles += [file for file in response.entries]
            response = dbx.files_list_folder_continue(response.cursor)
        listfiles += [file for file in response.entries]
        folder = row["folder"]
        path_folder = os.path.join(directory_target, folder)
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)
        for file in listfiles:
            path_drop = file.path_display
            path_local = os.path.join(
                directory_target, folder, path_drop.split("/")[-1]
            )
            print(path_drop, path_local)
            if (
                file.name != "time_hypha_info"
                and not "validation" in file.name
                and not "time_hull_info" in file.name
                and not "time_edge_info" in file.name
            ):
                download(path_drop, path_local, unzip=(path_drop[-4:] == ".zip"))


def compute_dropbox_hash(filename):
    file_size = os.stat(filename).st_size
    with open(filename, "rb") as f:
        block_hashes = b""
        while True:
            chunk = f.read(DROPBOX_HASH_CHUNK_SIZE)
            if not chunk:
                break
            block_hashes += hashlib.sha256(chunk).digest()
        return hashlib.sha256(block_hashes).hexdigest()


DROPBOX_HASH_CHUNK_SIZE = 4 * 1024 * 1024
