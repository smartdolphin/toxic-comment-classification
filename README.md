# toxic-comment-classification
[Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) 
The task was to classify online comments into 6 categories: `toxic`, `severve_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. 


## Summary of approach

__Private score: 0.9826__ / __Public score: 0.9833__


## Data

FastText embeddings trained locally on the competition data
Please download word2vec binary file from [fasttext site](https://fasttext.cc/docs/en/english-vectors.html)
And copy word2vec binary file to the data folder.

    └── data
        └── crawl-300d-2M.vec
        └── sample_submission.csv
        └── test.vec
        └── train.vec


## Usage

<b>Training locally</b>

    $ python main.py --output_file_path submission_result.csv

## Requirements
- python 3
- keras
- numpy
- matplotlib
- tensorflow
- pandas
- sklearn
