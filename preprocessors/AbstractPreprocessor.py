import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
from transformers import PreTrainedTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) if __file__ != '<input>' else '.'
DATA_DIR = os.path.join(ROOT_DIR, 'data/processed')


class AbstractPreprocessor(ABC):
    SUB_START_CHAR = '{'
    SUB_END_CHAR = '}'
    OBJ_START_CHAR = '['
    OBJ_END_CHAR = ']'
    PAD_CHAR = 0
    SENTENCE_LENGTH = 512
    PROCESSED_FILE_NAME = ''

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def preprocess_data(self):
        pass

    def _pickle_data(self, train_x, train_y, val_x, val_y, test_x, test_y):
        if not os.path.exists(DATA_DIR):
            print("Creating directory " + DATA_DIR)
            os.mkdir(DATA_DIR)

        lc = locals()
        for key in ['train', 'val', 'test']:
            data = {
                'x': lc[f'{key}_x'],
                'y': lc[f'{key}_y'],
            }
            file_name = os.path.join(DATA_DIR, f'{self.PROCESSED_FILE_NAME}_{key}.pkl')

            print(f"Saving to pickle file {file_name}")
            with open(file_name, 'wb') as f:
                pickle.dump(data, f)

    def _pad_sequence(self, sequence: list):
        if len(sequence) > self.SENTENCE_LENGTH:
            return np.array(sequence[:self.SENTENCE_LENGTH])
        else:
            return np.array(sequence + [self.PAD_CHAR] * (self.SENTENCE_LENGTH - len(sequence)))
