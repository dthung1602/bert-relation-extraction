# Relation extraction with BERT

The goal of this repo is to show how to use [BERT](https://arxiv.org/abs/1810.04805)
to [extract relation](https://en.wikipedia.org/wiki/Relationship_extraction) from text.

Used libraries:
- [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [Transformers](https://huggingface.co/transformers/index.html)

Used datasets:
- [SemiEval 2010 Task 8](http://semeval2.fbk.eu/semeval2.php?location=tasks&taskid=11)

## Setup

This project uses [Python 3.7](https://www.python.org/downloads/release/python-378/)

Clone this [repository](https://github.com/dthung1602/bert-relation-extraction):
``` shell script
git clone https://github.com/dthung1602/bert-relation-extraction
```

Create a conda virtual env with:
```shell script
conda create --name bert-re python=3.7
conda activate bert-re
```

Or if you use virtualenv:
```shell script
virtualenv -p python3.7 venv
source venv/bin/activate
```

Install the requirements (inside the project folder):
```shell script
pip install -r requirements.txt
```

## Getting Started

### Download datasets
``` shell script
python download.py
```

Available arguments:
```text
optional arguments:
  -h, --help                            Show this help message and exit
  --datasets [dataset1 [ds2 ...]]       List of datasets to download
                                        Available datasets:
                                            SemiEval2010Task8
                                        If not specified, all datasets are downloaded
  --force-redownload                    Force re-download       
```

### Preprocess data
``` shell script
python preprocess.py
```

Available arguments:
```text
optional arguments:
  -h, --help                            Show this help message and exit
  --preprocessor preprocessor_name      Preprocessor to run
                                        Available preprocessors:
                                            SemiEval2010Task8
                                        Default: SemiEval2010Task8
  --pretrain-weight pretrain_weight     Which pretrain weight to use
                                        Available pretrain weights: 
                                            distilbert-base-cased
                                            bert-base-uncased
                                            roberta-base
                                        Default: distilbert-base-cased
```

### Code style
All the code follows the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide.
