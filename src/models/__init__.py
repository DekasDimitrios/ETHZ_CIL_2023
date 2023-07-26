'''
    File name: __init__.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''


from src.models.config import update_cfg
from src.models.baselines.tf_idf import train_tf_idf
from src.models.baselines.ft import train_fasttext
from src.models.baselines.glove import train_glove
from src.models.baselines.w2v import train_w2v
from src.models.baselines.lstm import train_lstm
from src.models.baselines.bert_xgb import train_bert_xgb
import argparse


def run(cfg_path):
    """
    Used to execute the code necessary to carry out the training pipeline
    as it is dictated by the config file given as argument.

    :param cfg_path: a valid system path for the config file to be used
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=cfg_path,
                        help='path to the configuration yaml file')

    args = parser.parse_args()
    cfg = update_cfg(args.cfg)
    cfg.freeze()

    print('Start Model Training.')

    if cfg.EXPERIMENT.NAME == 'TF-IDF':
        val_score = train_tf_idf(cfg)
        print(f"Training of TD-IDF model completed with {val_score} as its validation score.")
    elif cfg.EXPERIMENT.NAME == 'FastText':
        val_score = train_fasttext(cfg)
        print(f"Training of FastText model completed with {val_score} as its validation score.")
    elif cfg.EXPERIMENT.NAME == 'GloVe':
        val_score = train_glove(cfg)
        print(f"Training of GloVe model completed with {val_score} as its validation score.")
    elif cfg.EXPERIMENT.NAME == 'Word2Vec':
        val_score = train_w2v(cfg)
        print(f"Training of Word2Vec model completed with {val_score} as its validation score.")
    elif cfg.EXPERIMENT.NAME == 'LSTM':
        train_lstm(cfg)
        print(f"Training of LSTM model completed.")
    elif cfg.EXPERIMENT.NAME == 'Bert-XGB':
        val_score = train_bert_xgb(cfg)
        print(f"Training of Bert-XGB model completed with {val_score} as its validation score.")
    else:
        print('The experiment name provided by the configuration is not available. Please make sure that a valid '
              'experiment name is given.')
