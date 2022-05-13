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
        
def upload(file_path, target_path, chunk_size=4 * 1024 * 1024):
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
                print("error")
                sleep(60)
                continue
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
        
def sync_fold(origin,target):
    cmd = f'rsync --update -avh {origin} {target}'
    # print(cmd)
    call(cmd,shell=True)