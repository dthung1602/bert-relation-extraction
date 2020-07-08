#!/usr/bin/env python3

import os
import sys
import zipfile
from argparse import ArgumentParser

import requests

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) if __file__ != '<input>' else '.'
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'data/raw')

DATASET_MAPPING = {
    'SemiEval2010Task8': 'https://github.com/sahitya0000/Relation-Classification/blob/master/corpus'
                         '/SemEval2010_task8_all_data.zip?raw=true',
}


def download_dataset(data_set_name: str, force_redonwload: bool) -> None:
    # create data directory
    if not os.path.exists(RAW_DATA_DIR):
        print("Creating directory " + RAW_DATA_DIR)
        os.mkdir(RAW_DATA_DIR)

    # check data has been downloaded
    dataset_dir = os.path.join(RAW_DATA_DIR, data_set_name)
    if os.path.exists(dataset_dir):
        if force_redonwload:
            print(f"Removing old raw data {dataset_dir}")
            os.unlink(dataset_dir)
        else:
            print(f"Directory {dataset_dir} exists, skip downloading. See option --force-redownload.")
            return

    # download & extract
    dataset_url = DATASET_MAPPING[data_set_name]
    tmp_file_path = os.path.join(RAW_DATA_DIR, data_set_name + '.zip')
    download_from_url(dataset_url, tmp_file_path)
    with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
        print("Extracting to " + RAW_DATA_DIR)
        zip_ref.extractall(RAW_DATA_DIR)

    print("Removing zip file")
    os.unlink(tmp_file_path)


def download_from_url(url: str, save_path: str, chunk_size: int = 1024) -> None:
    with open(save_path, "wb") as f:
        print("Downloading " + url)
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
        print(f"Download from {url} completed")


def add_download_args(parser: ArgumentParser):
    all_datasets = list(DATASET_MAPPING.keys())
    parser.add_argument("--datasets", type=str, default=all_datasets, nargs="*",
                        choices=all_datasets, help="List of datasets to download")
    parser.add_argument("--force-redownload", default=False, action="store_true", help="Force re-download")


if __name__ == '__main__':
    parser = ArgumentParser()
    add_download_args(parser)
    args = parser.parse_args()

    for ds in args.datasets:
        download_dataset(ds, args.force_redownload)
