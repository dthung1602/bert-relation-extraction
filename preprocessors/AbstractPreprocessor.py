import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
from transformers import PreTrainedTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(__file__)) if __file__ != '<input>' else '.'
DATA_DIR = os.path.join(ROOT_DIR, 'data/processed')


class AbstractPreprocessor(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def preprocess_data(self):
        pass

    def _pickle_data(self, data, file_name: str):
        if not os.path.exists(DATA_DIR):
            print("Creating directory " + DATA_DIR)
            os.mkdir(DATA_DIR)

        print(f"Saving to pickle file {file_name}")
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def _pad_sequence(self, sequence: list, pad: int, length: int):
        if len(sequence) > length:
            return np.array(sequence[:length])
        else:
            return np.array(sequence + [pad] * (length - len(sequence)))
