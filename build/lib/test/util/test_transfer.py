import unittest
from decouple import Config, RepositoryEnv
import os
from amftrack.util.dbx import upload, download

DOTENV_FILE = (
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    + "/local.env"
)
env_config = Config(RepositoryEnv(DOTENV_FILE))
target = env_config.get("DATA_PATH")
target2 = env_config.get("TEMP_PATH")


class TestTransfer(unittest.TestCase):
    def test_upload(self):
        upload(target, "/test/test")

    def test_download(self):
        download("/test/test", os.path.join(target2, "test.json"))
