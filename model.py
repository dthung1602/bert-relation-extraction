import pickle
from argparse import ArgumentParser, Namespace
from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import *

from preprocessors import PreprocessorFactory


class GenericDataset(Dataset):

    def __init__(self, dataset: str, subset: str):
        preprocessor_class = PreprocessorFactory.DATASET_MAPPING[dataset]
        if subset not in ['train', 'val', 'test']:
            raise ValueError('subset must be train, val or test')
        with open(preprocessor_class.get_pickle_file_name(subset), 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        return (torch.tensor(self.data['input_ids'][index]),
                torch.tensor(self.data['attention_mask'][index]),
                torch.tensor(self.data['label'][index]))

    def __len__(self) -> int:
        return len(self.data['label'])


MODEL_MAPPING = {
    'bert': BertModel,
    'distilbert': DistilBertModel,
    'roberta': RobertaModel,
}


class BERTModule(LightningModule):

    def __init__(self, hparams: Namespace):
        print("---> Start building model <----")
        super().__init__()
        self.save_hyperparameters(hparams)

        model_class = MODEL_MAPPING[hparams.bert_variant]
        self.bert = model_class.from_pretrained(hparams.pretrain_weight, output_attentions=True)

        self.num_classes = 10  # TODO
        self.linear = nn.Linear(self.bert.config.hidden_size, self.num_classes)
        print("Done building model\n")

    def on_train_start(self) -> None:
        print("\n---> Start training <----")

    @classmethod
    def add_train_args(cls, subparser):
        parser: ArgumentParser = subparser.add_parser('train')
        parser.add_argument("--dataset", choices=PreprocessorFactory.DATASET_MAPPING.keys(),
                            default="SemiEval2010Task8", help="Which dataset to train on")
        parser.add_argument("--bert-variant", choices=MODEL_MAPPING.keys(),
                            default="distilbert", help="Which BERT variant to use")
        parser.add_argument("--pretrain-weight", choices=PreprocessorFactory.TOKENIZER_MAPPING.keys(),
                            default="distilbert-base-cased", help="Which pretrain weight to use with BERT")
        parser.add_argument("--learning-rate", type=float, default=2e-05, help="Learning rate")
        parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
        parser.add_argument("--min-epochs", type=int, default=1,
                            help="Force training for at least these many epochs")
        parser.add_argument("--max-epochs", type=int, default=5,
                            help="Stop training once this number of epochs is reached")

    @classmethod
    def add_test_args(cls, subparser):
        pass

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return self.__get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self.__get_dataloader('val')

    def test_dataloader(self) -> DataLoader:
        return self.__get_dataloader('test')

    def __get_dataloader(self, subset: str) -> DataLoader:
        print(f"Loading {subset} data")
        return DataLoader(
            GenericDataset(self.hparams.dataset, subset),
            batch_size=self.hparams.batch_size,
            shuffle=(subset == 'train')
        )

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate,
            eps=1e-08
        )

    def forward(self, input_ids, attention_mask) -> Tensor:
        bert_output, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        bert_cls = bert_output[:, 0]
        logits = self.linear(bert_cls)
        return logits

    def training_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, label = batch

        y_hat = self(input_ids, attention_mask)

        loss = F.cross_entropy(y_hat, label)
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, label = batch

        y_hat = self(input_ids, attention_mask)

        loss = F.cross_entropy(y_hat, label)

        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = torch.tensor(accuracy_score(y_hat.cpu(), label.cpu()))
        val_f1 = torch.tensor(f1_score(y_hat.cpu(), label.cpu()))

        return {'val_loss': loss, 'val_acc': val_acc, 'val_f1': val_f1}

    def validation_epoch_end(self, outputs) -> dict:
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_val_loss, 'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_val_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, label = batch

        y_hat = self(input_ids, attention_mask)

        a, y_hat = torch.max(y_hat, dim=1)
        y_hat = y_hat.cpu()
        label = label.cpu()
        test_acc = accuracy_score(y_hat, label)  # TODO check
        test_f1 = f1_score(y_hat, label)
        test_pre = precision_score(y_hat, label)
        test_recall = recall_score(y_hat, label)

        return {
            'test_acc': torch.tensor(test_acc),
            'test_f1': torch.tensor(test_f1),
            'test_pre': torch.tensor(test_pre),
            'test_recall': torch.tensor(test_recall),
        }

    def test_epoch_end(self, outputs) -> dict:
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
