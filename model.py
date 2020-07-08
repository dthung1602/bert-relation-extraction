from typing import Union, List

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


class BERTModule(LightningModule):

    def __init__(self):
        super().__init__()

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        pass

    def forward(self, *args, **kwargs):
        pass
