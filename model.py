import pickle
from argparse import ArgumentParser, Namespace
from typing import Union, List

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

from preprocessors import PreprocessorFactory


class GenericDataset(Dataset):

    def __init__(self, dataset: str, data_type: str):
        preprocessor_class = PreprocessorFactory.DATASET_MAPPING[dataset]
        if data_type not in ['train', 'val', 'test']:
            raise ValueError('data_type must be train, val or test')
        with open(preprocessor_class.get_pickle_file_name(data_type), 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index: int):
        return self.data['x'][index], self.data['y'][index]

    def __len__(self) -> int:
        return len(self.data['x'])


class BERTModule(LightningModule):

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)

    @classmethod
    def add_train_args(cls, subparser):
        parser: ArgumentParser = subparser.add_parser('test')
        parser.add_argument("--dataset", choices=PreprocessorFactory.DATASET_MAPPING.keys(),
                            default="SemiEval2010Task8", help="Which dataset to train on")
        parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
        parser.add_argument("--min-epochs", type=int, default=1,
                            help="Force training for at least these many epochs")
        parser.add_argument("--max-epochs", type=int, default=5,
                            help="Stop training once this number of epochs is reached")

    @classmethod
    def add_test_args(cls, subparser):
        pass

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return self.__get_dataloader('train')

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.__get_dataloader('val')

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.__get_dataloader('test')

    def __get_dataloader(self, data_type):
        return DataLoader(
            GenericDataset(self.hparams.dataset, data_type),
            batch_size=self.hparams.batch_size
        )

    def forward(self, *args, **kwargs):
        pass
