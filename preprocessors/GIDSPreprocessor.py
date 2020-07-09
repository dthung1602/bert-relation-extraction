import json
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from downloaders import RAW_DATA_DIR
from .AbstractPreprocessor import AbstractPreprocessor


class GIDSPreprocessor(AbstractPreprocessor):
    DATASET_NAME = 'GIDS'
    RAW_TRAIN_FILE_NAME = os.path.join(RAW_DATA_DIR, 'gids_data/gids_train.json')
    RAW_VAL_FILE_NAME = os.path.join(RAW_DATA_DIR, 'gids_data/gids_dev.json')
    RAW_TEST_FILE_NAME = os.path.join(RAW_DATA_DIR, 'gids_data/gids_test.json')

    def _preprocess_data(self):
        print("Processing train data")
        train_x, train_y = self._get_data_from_file(self.RAW_TRAIN_FILE_NAME)
        print("Processing validate data")
        val_x, val_y = self._get_data_from_file(self.RAW_VAL_FILE_NAME)
        print("Processing test data")
        test_x, test_y = self._get_data_from_file(self.RAW_TEST_FILE_NAME)

        print("Encoding labels to integers")
        le = LabelEncoder()
        le.fit(train_y)
        train_y = le.transform(train_y)
        val_y = le.transform(val_y)
        test_y = le.transform(test_y)

        self._pickle_data(train_x, train_y, val_x, val_y, test_x, test_y)

    def _get_data_from_file(self, file_name: str):
        sentences = []
        labels = []
        with open(file_name) as f:
            for line in tqdm(f.readlines()):
                data = json.loads(line)
                sentence = " ".join(data['sent'])

                # add subject markup
                new_sub = self.SUB_START_CHAR + data['sub'].replace('_', '') + self.SUB_END_CHAR
                new_obj = self.OBJ_START_CHAR + data['obj'].replace('_', '') + self.OBJ_END_CHAR
                sentence = sentence.replace(data['sub'], new_sub).replace(data['obj'], new_obj)

                # tokenize sentence
                sequence = self.tokenizer.encode(sentence)
                sentences.append(self._pad_sequence(sequence))

                labels.append(data['rel'])

        return np.array(sentences), np.array(labels)
