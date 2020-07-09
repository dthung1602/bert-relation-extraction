from .AbstractDownloader import RAW_DATA_DIR, AbstractDownloader
from .GIDSDownloader import GIDSDownloader
from .SemiEval2010Task8Downloader import SemiEval2010Task8Downloader


class DownloaderFactory:
    MAPPING = {
        'SemiEval2010Task8': SemiEval2010Task8Downloader,
        'GIDS': GIDSDownloader
    }

    def get_downloader(self, dataset: str) -> AbstractDownloader:
        return self.MAPPING.get(dataset)()

    @classmethod
    def add_download_args(cls, subparser):
        all_datasets = list(cls.MAPPING.keys())
        parser = subparser.add_parser('download')
        parser.add_argument("--datasets", type=str, default=all_datasets, nargs="*",
                            choices=all_datasets, help="List of datasets to download")
        parser.add_argument("--force-redownload", default=False, action="store_true", help="Force re-download")
