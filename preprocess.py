#!/usr/bin/env python3

import os
import pickle
from abc import ABC, abstractmethod
from argparse import ArgumentParser

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import PreTrainedTokenizer, BertTokenizer, \
    DistilBertTokenizer, RobertaTokenizer

from download import RAW_DATA_DIR

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) if __file__ != '<input>' else '.'
DATA_DIR = os.path.join(ROOT_DIR, 'data/processed')


class Preprocessor(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def preprocess_data(self):
        pass

    def pickle_data(self, data, file_name: str):
        if not os.path.exists(DATA_DIR):
            print("Creating directory " + DATA_DIR)
            os.mkdir(DATA_DIR)

        print(f"Saving to pickle file {file_name}")
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def pad_sequence(self, sequence: list, pad: int, length: int):
        if len(sequence) > length:
            return np.array(sequence[:length])
        else:
            return np.array(sequence + [pad] * (length - len(sequence)))


class SemiEval2010Task8Preprocessor(Preprocessor):
    RAW_TRAIN_FILE_NAME = os.path.join(RAW_DATA_DIR,
                                       'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT')
    RAW_TEST_FILE_NAME = os.path.join(RAW_DATA_DIR,
                                      'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
    RAW_TRAIN_DATA_SIZE = 8000
    RAW_TEST_DATA_SIZE = 2717
    RANDOM_SEED = 2020
    VAL_DATA_PROPORTION = 0.2

    def preprocess_data(self):
        print("---> Preprocessing SemEval2010 dataset <---")

        print("Processing training data")
        train_x, train_y = self._get_data_from_file(
            self.RAW_TRAIN_FILE_NAME,
            self.RAW_TRAIN_DATA_SIZE
        )

        print("Processing test data")
        test_x, test_y = self._get_data_from_file(
            self.RAW_TEST_FILE_NAME,
            self.RAW_TEST_DATA_SIZE
        )

        print("Encoding labels to integers")
        le = LabelEncoder()
        le.fit(train_y)
        train_y = le.transform(train_y)
        test_y = le.transform(test_y)

        print("Splitting train & validate data")
        train_x, val_x, train_y, val_y = train_test_split(
            train_x, train_y,
            test_size=self.VAL_DATA_PROPORTION,
            random_state=self.RANDOM_SEED
        )

        lc = locals()
        for key in ['train', 'val', 'test']:
            data = {
                'x': lc[f'{key}_x'],
                'y': lc[f'{key}_y'],
            }
            file_name = os.path.join(DATA_DIR, f'semieval_2010_{key}.pkl')
            self.pickle_data(data, file_name)

    def _get_data_from_file(self, file_name: str, dataset_size: int):
        sentences = []
        labels = []
        with open(file_name) as f:
            for i in range(dataset_size):
                sentences.append(self._process_sentence(f.readline()))
                labels.append(self._process_label(f.readline()))
                f.readline()
                f.readline()
        return np.array(sentences), np.array(labels)

    def _process_sentence(self, sentence: str):
        sentence = sentence.split("\t")[1][1:-2] \
            .replace("<e1>", "[").replace("</e1>", "]") \
            .replace("<e2>", "{").replace("</e2>", "}")
        sequence = self.tokenizer.encode(sentence)
        return self.pad_sequence(sequence, 0, 512)

    def _process_label(self, label: str):
        return label[:-8]


TOKENIZER_MAPPING = {
    'bert-base-uncased': BertTokenizer,
    'distilbert-base-cased': DistilBertTokenizer,
    'roberta-base': RobertaTokenizer,
}

PREPROCESSOR_MAPPING = {
    'SemiEval2010Task8': SemiEval2010Task8Preprocessor
}


def add_preprocessing_args(parser: ArgumentParser):
    all_preprocessors = list(PREPROCESSOR_MAPPING.keys())
    all_pretrain_weights = list(TOKENIZER_MAPPING.keys())
    parser.add_argument("--preprocessor", type=str, default='SemiEval2010Task8',
                        choices=all_preprocessors, help="Preprocessor to run")
    parser.add_argument("--pretrain-weight", type=str, default='distilbert-base-cased',
                        choices=all_pretrain_weights, help="Which pretrain weight to use")


if __name__ == '__main__':
    parser = ArgumentParser()
    add_preprocessing_args(parser)
    args = parser.parse_args()

    preprocessor_class = PREPROCESSOR_MAPPING[args.preprocessor]
    tokenizer_class = TOKENIZER_MAPPING[args.pretrain_weight]

    tokenizer = tokenizer_class.from_pretrained(args.pretrain_weight)
    preprocessor = preprocessor_class(tokenizer)

    preprocessor.preprocess_data()
