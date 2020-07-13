import pickle
from argparse import ArgumentParser, Namespace
from typing import Tuple
import os
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


ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) if __file__ != '<input>' else '.'
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')

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
        parser.add_argument("--max-epochs", type=int, default=4,
                            help="Stop training once this number of epochs is reached")
        parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR,
                            help="Where to save checkpoints")

    @classmethod
    def add_test_args(cls, subparser):
        parser: ArgumentParser = subparser.add_parser('test')
        parser.add_argument("--dataset", choices=PreprocessorFactory.DATASET_MAPPING.keys(),
                            default="SemiEval2010Task8", help="Which dataset to test on")
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
        parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
        parser.add_argument("--checkpoint", type=str, help="Path of checkpoint to load")

    def prepare_data(self):
        # create checkpoint dir
        if not os.path.exists(CHECKPOINT_DIR):
            print(f"Creating checkpoint directory at {CHECKPOINT_DIR}")
            os.makedirs(CHECKPOINT_DIR)

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
        y_hat = y_hat.cpu()
        label = label.cpu()

        return {
            'val_loss': loss,
            'val_pre': torch.tensor(precision_score(label, y_hat, average='micro')),
            'val_rec': torch.tensor(recall_score(label, y_hat, average='micro')),
            'val_acc': torch.tensor(accuracy_score(label, y_hat)),
            'val_f1': torch.tensor(f1_score(label, y_hat, average='micro'))
        }

    def validation_epoch_end(self, outputs) -> dict:
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_pre = torch.stack([x['val_pre'] for x in outputs]).mean()
        avg_val_rec = torch.stack([x['val_rec'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()

        tensorboard_logs = {
            'val_loss': avg_val_loss,
            'avg_val_pre': avg_val_pre,
            'avg_val_rec': avg_val_rec,
            'avg_val_acc': avg_val_acc,
            'avg_val_f1': avg_val_f1,
        }
        return {'val_loss': avg_val_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb) -> dict:
        input_ids, attention_mask, label = batch

        y_hat = self(input_ids, attention_mask)

        a, y_hat = torch.max(y_hat, dim=1)
        y_hat = y_hat.cpu()
        label = label.cpu()
        test_pre = precision_score(label, y_hat, average='micro')
        test_rec = recall_score(label, y_hat, average='micro')
        test_acc = accuracy_score(label, y_hat)
        test_f1 = f1_score(label, y_hat, average='micro')

        return {
            'test_pre': torch.tensor(test_pre),
            'test_rec': torch.tensor(test_rec),
            'test_acc': torch.tensor(test_acc),
            'test_f1': torch.tensor(test_f1),
        }

    def test_epoch_end(self, outputs) -> dict:
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_test_pre = torch.stack([x['test_pre'] for x in outputs]).mean()
        avg_test_rec = torch.stack([x['test_rec'] for x in outputs]).mean()
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_test_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()

        tensorboard_logs = {
            'test_loss': avg_test_loss,
            'avg_test_pre': avg_test_pre,
            'avg_test_rec': avg_test_rec,
            'avg_test_acc': avg_test_acc,
            'avg_test_f1': avg_test_f1,
        }
        return {'test_loss': avg_test_loss, 'progress_bar': tensorboard_logs}
