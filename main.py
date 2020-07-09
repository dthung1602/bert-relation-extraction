#!/usr/bin/env python3

from argparse import ArgumentParser

from pytorch_lightning import Trainer

from downloaders import DownloaderFactory
from model import BERTModule
from preprocessors import PreprocessorFactory

if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    DownloaderFactory.add_download_args(subparsers)
    PreprocessorFactory.add_preprocess_args(subparsers)
    BERTModule.add_train_args(subparsers)
    BERTModule.add_test_args(subparsers)

    args = parser.parse_args()

    if args.command == 'download':
        downloader_factory = DownloaderFactory()
        for ds in args.datasets:
            downloader = downloader_factory.get_downloader(ds)
            downloader.download(args.force_redownload)

    elif args.command == 'preprocess':
        preprocessor_factory = PreprocessorFactory()
        preprocessor = preprocessor_factory.get_preprocessor(args.dataset, args.pretrain_weight)
        preprocessor.preprocess_data(args.reprocess)

    elif args.command == 'train':
        model = BERTModule(args)
        trainer = Trainer(
            gpus=args.gpus,
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs
        )
        trainer.fit(model)
