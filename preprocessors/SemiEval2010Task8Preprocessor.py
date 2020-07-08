import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from downloaders import RAW_DATA_DIR
from .AbstractPreprocessor import AbstractPreprocessor, DATA_DIR


class SemiEval2010Task8Preprocessor(AbstractPreprocessor):
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

        lc = locals()
        for key in ['train', 'val', 'test']:
            data = {
                'x': lc[f'{key}_x'],
                'y': lc[f'{key}_y'],
            }
            file_name = os.path.join(DATA_DIR, f'semieval_2010_{key}.pkl')
            self._pickle_data(data, file_name)

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
        return self._pad_sequence(sequence, 0, 512)

    def _process_label(self, label: str):
        return label[:-8]
