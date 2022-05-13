from tqdm.autonotebook import tqdm
import dropbox
import os

from zipfile import ZipFile, ZIP_DEFLATED
import os
from os.path import basename
import requests
# create a ZipFile object
from subprocess import call
from time import sleep
from decouple import Config, RepositoryEnv
from time import time_ns

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

def load_dbx():
    dbx = dropbox.DropboxTeam(app_key = app_key,
                app_secret = app_secret, oauth2_refresh_token = refresh_token)
    p = dropbox.common.PathRoot.namespace_id(folder_id)
    dbx = dbx.with_path_root(p)
    dbx = dbx.as_user(user_id)
    return(dbx)

def zip_file(origin,target,depth=2):
    with ZipFile(target, 'w',compression=ZIP_DEFLATED) as zipObj:
       # Iterate over all the files in directory
        tot=0
        for folderName, subfolders, filenames in os.walk(origin):
            for filename in filenames:
                tot+=1
        with tqdm(total=tot, desc="zipping", leave=False) as pbar:
            for folderName, subfolders, filenames in os.walk(origin):
                for filename in filenames:
                    #create complete filepath of file in directory
                    filePath = os.path.join(folderName, filename)
                    filePath = os.path.normpath(filePath)
                    # Add file to zip
                    place = filePath.replace(origin,'')
                    zipObj.write(filePath, place)
                    pbar.update(1)
                    
def unzip_file(origin,target,depth=2):
    with ZipFile(origin, 'r') as zipy:
        zipy.extractall(target)
        
def upload(file_path, target_path, chunk_size=4 * 1024 * 1024, catch_exception=True):
    dbx = load_dbx()
    with open(file_path, "rb") as f:
        file_size = os.path.getsize(file_path)
        while True:
            try:
                if file_size <= chunk_size:
                    dbx.files_upload(f.read(), target_path, mode=dropbox.files.WriteMode.overwrite)
                    #Overwriting files by default
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
                            commit = dropbox.files.CommitInfo(path=target_path, mode=dropbox.files.WriteMode.overwrite) #Overwriting files by default
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
            except (requests.exceptions.RequestException,dropbox.exceptions.ApiError) as e:
                if catch_exception:
                    print("error")
                    sleep(60)
                    continue
                else:
                    return(None)
            break
            
                
                    
def download(file_path, target_path, end=''):
    dbx = load_dbx()
    while True:
        try:
            with open(target_path, "wb") as f:
                metadata, res = dbx.files_download(path=f'{file_path}{end}')
                f.write(res.content)
        except (requests.exceptions.RequestException,dropbox.exceptions.ApiError) as e:
            sleep(60)
            continue
        break

def upload_zip(path_total,target,trhesh = 4 * 1024 * 1024):
    if os.path.isdir(path_total):
        stamp = time_ns()
        path_zip = f'{temp_path}/{stamp}.zip'
        zip_file(path_total, path_zip)
        upload(path_zip, target,
               chunk_size=256 * 1024 * 1024)
        os.remove(path_zip)

    else:
        upload(path_total, target,
               chunk_size=256 * 1024 * 1024)

def sync_fold(origin,target):
    cmd = f'rsync --update -avh {origin} {target}'
    # print(cmd)
    call(cmd,shell=True)

def upload_folders(folders,dir_drop = 'DATA',catch_exception=True):
    run_info = folders.copy()
    folder_list = list(run_info['folder'])
    with tqdm(total=len(folder_list), desc="transferred") as pbar:
        for folder in folder_list:
            directory_name = folder
            run_info['unique_id'] = run_info['Plate'].astype(str) + "_" + run_info['CrossDate'].astype(str)
            line = run_info.loc[run_info['folder'] == directory_name]
            id_unique = line['unique_id'].iloc[0]

            path_snap = line['total_path'].iloc[0]
            path_info = f'{temp_path}/{directory_name}_info.json'

            for subfolder in ["Img", "Analysis"]:
                path_zip = f'{temp_path}/{directory_name}_{subfolder}.zip'
                line.to_json(path_info)
                zip_file(os.path.join(path_snap, subfolder), path_zip)
                upload(path_zip, f'/{dir_drop}/{id_unique}/{directory_name}/{subfolder}.zip',chunk_size=256 * 1024 * 1024,catch_exception=catch_exception)
            upload(path_info, f'/{dir_drop}/{id_unique}/{directory_name}_info.json', chunk_size=256 * 1024 * 1024,
                   catch_exception=catch_exception)
            os.remove(path_info)
            os.remove(path_zip)
            pbar.update(1)