import json
import os

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
        train_input_ids, train_attention, train_label = self._get_data_from_file(self.RAW_TRAIN_FILE_NAME)
        print("Processing validate data")
        val_input_ids, val_attention, val_label = self._get_data_from_file(self.RAW_VAL_FILE_NAME)
        print("Processing test data")
        test_input_ids, test_attention, test_label = self._get_data_from_file(self.RAW_TEST_FILE_NAME)

        print("Encoding labels to integers")
        le = LabelEncoder()
        le.fit(train_label)
        train_label = le.transform(train_label).tolist()
        val_label = le.transform(val_label).tolist()
        test_label = le.transform(test_label).tolist()

        lc = locals()
        for k in ['train', 'val', 'test']:
            file_name = self.get_pickle_file_name(k)
            self._pickle_data({
                'input_ids': lc[f'{k}_input_ids'],
                'attention_mask': lc[f'{k}_attention'],
                'label': lc[f'{k}_label']
            }, file_name)

    def _get_data_from_file(self, file_name: str):
        sentences = []
        attentions = []
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
                sequence = self._pad_sequence(sequence)

                sentences.append(sequence)
                attentions.append(self._get_attention_mask(sequence))
                labels.append(data['rel'])

        return sentences, attentions, labels
