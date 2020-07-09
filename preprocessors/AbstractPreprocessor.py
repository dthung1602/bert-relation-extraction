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
    DATASET_NAME = ''

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def preprocess_data(self, reprocess: bool):
        print(f"\n---> Preprocessing {self.DATASET_NAME} dataset <---")

        # stop preprocessing if file existed
        pickled_file_names = [self.__get_pickle_file_name(k) for k in ('train', 'val', 'test')]
        existed_files = [fn for fn in pickled_file_names if os.path.exists(fn)]
        if existed_files:
            file_text = "- " + "\n- ".join(existed_files)
            if not reprocess:
                print("The following files already exist:")
                print(file_text)
                print("Preprocessing is skipped. See option --reprocess.")
                return
            else:
                print("The following files will be overwritten:")
                print(file_text)

        self._preprocess_data()

    @abstractmethod
    def _preprocess_data(self):
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
            file_name = self.__get_pickle_file_name(key)

            print(f"Saving to pickle file {file_name}")
            with open(file_name, 'wb') as f:
                pickle.dump(data, f)

    def _pad_sequence(self, sequence: list):
        if len(sequence) > self.SENTENCE_LENGTH:
            return np.array(sequence[:self.SENTENCE_LENGTH])
        else:
            return np.array(sequence + [self.PAD_CHAR] * (self.SENTENCE_LENGTH - len(sequence)))

    def __get_pickle_file_name(self, key: str):
        return os.path.join(DATA_DIR, f'{self.DATASET_NAME.lower()}_{key}.pkl')
