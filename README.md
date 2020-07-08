# Relation extraction with BERT

The goal of this repo is to show how to use [BERT](https://arxiv.org/abs/1810.04805)
to [extract relation](https://en.wikipedia.org/wiki/Relationship_extraction) from text.

Used libraries:
- [Transformers](https://huggingface.co/transformers/index.html)
- [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)

Used datasets:
- SemiEval 2010 Task 8 - [paper](https://arxiv.org/pdf/1911.10422.pdf) - [download](https://github.com/sahitya0000/Relation-Classification/blob/master/corpus/SemEval2010_task8_all_data.zip?raw=true)
-  Google IISc Distant Supervision (GIDS) - [paper](https://arxiv.org/pdf/1804.06987.pdf) - [download](https://drive.google.com/open?id=1gTNAbv8My2QDmP-OHLFtJFlzPDoCG4aI)

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
python main.py download
```

Available arguments:
```text
optional arguments:
  -h, --help                            Show this help message and exit
  --datasets [dataset1 [ds2 ...]]       List of datasets to download
                                        Available datasets:
                                            - SemiEval2010Task8
                                        If not specified, all datasets are downloaded
  --force-redownload                    Force re-download       
```

### Preprocess data
``` shell script
python main.py preprocess
```

Available arguments:
```text
optional arguments:
  -h, --help                            Show this help message and exit
  --dataset dataset_name                Dataset to process
                                        Available datasets:
                                            - SemiEval2010Task8
                                        Default: SemiEval2010Task8
  --pretrain-weight pretrain_weight     Which pretrain weight to use
                                        Available pretrain weights: 
                                            - distilbert-base-cased
                                            - bert-base-uncased
                                            - roberta-base
                                        Default: distilbert-base-cased
```

### Code style
All the code follows the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide.
