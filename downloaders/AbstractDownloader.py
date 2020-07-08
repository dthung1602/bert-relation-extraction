import os
import shutil
import zipfile
from abc import ABC, abstractmethod

ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) if __file__ != '<input>' else '.'
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data/raw')


class AbstractDownloader(ABC):
    DATASET_NAME = ''
    DATASET_DIR = ''

    def download(self, force_redownload: bool):
        print(f"\n---> Downloading dataset {self.DATASET_NAME} <---")

        if not os.path.exists(RAW_DATA_DIR):
            print("Creating raw data directory " + RAW_DATA_DIR)
            os.mkdir(RAW_DATA_DIR)

        # check data has been downloaded
        if os.path.exists(self.DATASET_DIR):
            if force_redownload:
                print(f"Removing old raw data {self.DATASET_DIR}")
                shutil.rmtree(self.DATASET_DIR)
            else:
                print(f"Directory {self.DATASET_DIR} exists, skip downloading.\nSee option --force-redownload.")
                return

        # download
        self._download()

    @abstractmethod
    def _download(self):
        pass

    def _extract_zip(self, zip_file_path: str, extract_dir: str, remove_zip_file=True):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            print("Extracting to " + extract_dir)
            zip_ref.extractall(extract_dir)

        if remove_zip_file:
            print("Removing zip file")
            os.unlink(zip_file_path)
