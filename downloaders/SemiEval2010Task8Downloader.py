import os
import sys

import requests

from .AbstractDownloader import AbstractDownloader, RAW_DATA_DIR


class SemiEval2010Task8Downloader(AbstractDownloader):
    DATASET_NAME = 'SemiEval2010Task8'
    DATASET_DIR = os.path.join(RAW_DATA_DIR, 'SemEval2010_task8_all_data')
    URL = 'https://github.com/sahitya0000/Relation-Classification/blob/master/corpus' \
          '/SemEval2010_task8_all_data.zip?raw=true'

    def _download(self):
        tmp_file_path = os.path.join(RAW_DATA_DIR, self.DATASET_NAME + '.zip')
        self._download_from_url(self.URL, tmp_file_path)
        self._extract_zip(tmp_file_path, RAW_DATA_DIR, remove_zip_file=True)

    def _download_from_url(self, url: str, save_path: str, chunk_size: int = 1024) -> None:
        with open(save_path, "wb") as f:
            print(f"Downloading...\nFrom: {url}\nTo: {save_path}")
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                # show a progress bar
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=chunk_size):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()
                print()