from transformers import BertTokenizer, DistilBertTokenizer, RobertaTokenizer

from .AbstractPreprocessor import AbstractPreprocessor
from .GIDSPreprocessor import GIDSPreprocessor
from .SemiEval2010Task8Preprocessor import SemiEval2010Task8Preprocessor


class PreprocessorFactory:
    TOKENIZER_MAPPING = {
        'bert-base-uncased': BertTokenizer,
        'distilbert-base-cased': DistilBertTokenizer,
        'roberta-base': RobertaTokenizer,
    }

    DATASET_MAPPING = {
        'SemiEval2010Task8': SemiEval2010Task8Preprocessor,
        'GIDS': GIDSPreprocessor,
    }

    def get_preprocessor(self, dataset: str, pretrain_weight: str) -> AbstractPreprocessor:
        preprocessor_class = self.DATASET_MAPPING[dataset]
        tokenizer_class = self.TOKENIZER_MAPPING[pretrain_weight]

        tokenizer = tokenizer_class.from_pretrained(pretrain_weight)
        return preprocessor_class(tokenizer)

    @classmethod
    def add_preprocess_args(cls, subparser):
        all_preprocessors = list(cls.DATASET_MAPPING.keys())
        all_pretrain_weights = list(cls.TOKENIZER_MAPPING.keys())
        parser = subparser.add_parser('preprocess')
        parser.add_argument("--dataset", type=str, default='SemiEval2010Task8',
                            choices=all_preprocessors, help="Dataset to process")
        parser.add_argument("--pretrain-weight", type=str, default='distilbert-base-cased',
                            choices=all_pretrain_weights, help="Which pretrain weights to use")
