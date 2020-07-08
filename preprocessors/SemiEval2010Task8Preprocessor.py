import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from downloaders import RAW_DATA_DIR
from .AbstractPreprocessor import AbstractPreprocessor


class SemiEval2010Task8Preprocessor(AbstractPreprocessor):
    PROCESSED_FILE_NAME = 'semieval_2010'
    RAW_TRAIN_FILE_NAME = os.path.join(RAW_DATA_DIR,
                                       'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT')
    RAW_TEST_FILE_NAME = os.path.join(RAW_DATA_DIR,
                                      'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
    RAW_TRAIN_DATA_SIZE = 8000
    RAW_TEST_DATA_SIZE = 2717
    RANDOM_SEED = 2020
    VAL_DATA_PROPORTION = 0.2

    def preprocess_data(self):
        print("\n---> Preprocessing SemEval2010 dataset <---")

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

        self._pickle_data(train_x, train_y, val_x, val_y, test_x, test_y)

    def _get_data_from_file(self, file_name: str, dataset_size: int):
        sentences = []
        labels = []
        with open(file_name) as f:
            for _ in tqdm(range(dataset_size)):
                sentences.append(self._process_sentence(f.readline()))
                labels.append(self._process_label(f.readline()))
                f.readline()
                f.readline()
        return np.array(sentences), np.array(labels)

    def _process_sentence(self, sentence: str):
        # TODO distinguish e1 e2 sub obj
        sentence = sentence.split("\t")[1][1:-2] \
            .replace("<e1>", self.SUB_START_CHAR) \
            .replace("</e1>", self.SUB_END_CHAR) \
            .replace("<e2>", self.OBJ_START_CHAR) \
            .replace("</e2>", self.OBJ_END_CHAR)
        sequence = self.tokenizer.encode(sentence)
        return self._pad_sequence(sequence)

    def _process_label(self, label: str):
        return label[:-8]
