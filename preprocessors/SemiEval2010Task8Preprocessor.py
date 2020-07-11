import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from downloaders import RAW_DATA_DIR
from .AbstractPreprocessor import AbstractPreprocessor


class SemiEval2010Task8Preprocessor(AbstractPreprocessor):
    DATASET_NAME = 'SemiEval2010Task8'
    RAW_TRAIN_FILE_NAME = os.path.join(RAW_DATA_DIR,
                                       'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT')
    RAW_TEST_FILE_NAME = os.path.join(RAW_DATA_DIR,
                                      'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT')
    RAW_TRAIN_DATA_SIZE = 8000
    RAW_TEST_DATA_SIZE = 2717
    RANDOM_SEED = 2020
    VAL_DATA_PROPORTION = 0.2

    def _preprocess_data(self):
        print("Processing training data")
        train_input_ids, train_attention, train_label = self._get_data_from_file(
            self.RAW_TRAIN_FILE_NAME,
            self.RAW_TRAIN_DATA_SIZE
        )

        print("Processing test data")
        test_input_ids, test_attention, test_label = self._get_data_from_file(
            self.RAW_TEST_FILE_NAME,
            self.RAW_TEST_DATA_SIZE
        )

        print("Encoding labels to integers")
        le = LabelEncoder()
        le.fit(train_label)
        train_label = le.transform(train_label).tolist()
        test_label = le.transform(test_label).tolist()

        print("Splitting train & validate data")
        train_input_ids, val_input_ids, train_attention, val_attention, train_label, val_label = train_test_split(
            train_input_ids, train_attention, train_label,
            test_size=self.VAL_DATA_PROPORTION,
            random_state=self.RANDOM_SEED
        )

        lc = locals()
        for k in ['train', 'val', 'test']:
            file_name = self.get_pickle_file_name(k)
            self._pickle_data({
                'input_ids': lc[f'{k}_input_ids'],
                'attention_mask': lc[f'{k}_attention'],
                'label': lc[f'{k}_label']
            }, file_name)

    def _get_data_from_file(self, file_name: str, dataset_size: int):
        sentences = []
        attentions = []
        labels = []
        with open(file_name) as f:
            for _ in tqdm(range(dataset_size)):
                sentence = self._process_sentence(f.readline())
                sentences.append(sentence)
                attentions.append(self._get_attention_mask(sentence))
                labels.append(self._process_label(f.readline()))
                f.readline()
                f.readline()
        return sentences, attentions, labels

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
