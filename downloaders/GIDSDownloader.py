import os

import gdown

from .AbstractDownloader import AbstractDownloader, RAW_DATA_DIR


class GIDSDownloader(AbstractDownloader):
    DATASET_NAME = 'GIDS'
    DATASET_DIR = os.path.join(RAW_DATA_DIR, 'gids_data')
    URL = 'https://drive.google.com/uc?id=1gTNAbv8My2QDmP-OHLFtJFlzPDoCG4aI&export=download'

    def _download(self):
        tmp_file_path = os.path.join(RAW_DATA_DIR, self.DATASET_NAME + '.zip')
        gdown.download(self.URL, tmp_file_path, use_cookies=False)
        self._extract_zip(tmp_file_path, RAW_DATA_DIR, remove_zip_file=True)
