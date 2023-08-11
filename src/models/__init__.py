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
from src.models.novelty.transformer import train_hugging_face_transformer
from src.models.novelty.kfold import prod_k_fold
from src.models.novelty.dictionary import create_dictionary
from src.models.novelty.stage1 import train_stage1_split
from src.models.novelty.stage2 import train_stage2_split
from src.models.novelty.ensemble import apply_ensemble
import argparse


def run(cfg_path, flag):
    """
    Used to execute the code necessary to carry out the training pipeline
    as it is dictated by the config file and flag given as arguments.

    :param cfg_path: a valid system path for the config file to be used
    :param flag: a string that specifies whether a baseline or a novelty run is to be executed
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=cfg_path,
                        help='path to the configuration yaml file')

    args = parser.parse_args()
    cfg = update_cfg(args.cfg)
    cfg.freeze()

    print(f'Start {flag} Model Training.')

    if flag == 'Baseline':
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
            print('The baseline experiment name provided by the configuration is not available. Please make sure that a valid '
                  'experiment name is given.')
    elif flag == 'Novelty':
        prod_k_fold(cfg)
        print(f"Training and validation fold for novelty runs creation completed.")
        create_dictionary(cfg)
        print(f"Creation of score dictionary completed.")
        if cfg.EXPERIMENT.NAME == 'Transformer':
            train_hugging_face_transformer(cfg)
            print(f"Training of {cfg.PARAMETERS.MODEL_NAME} model completed.")
        if cfg.EXPERIMENT.NAME == 'Two-Stage':
            logit1 = train_stage1_split(cfg, 1)
            print(f"Training Stage 1 Split 1 completed.")
            logit2 = train_stage1_split(cfg, 2)
            print(f"Training Stage 1 Split 2 completed.")
            logit3 = train_stage1_split(cfg, 3)
            print(f"Training Stage 1 Split 3 completed.")
            logit4 = train_stage2_split(cfg, 1)
            print(f"Training Stage 2 Split 1 completed.")
            logit5 = train_stage2_split(cfg, 2)
            print(f"Training Stage 2 Split 2 completed.")
            logit6 = train_stage2_split(cfg, 3)
            print(f"Training Stage 2 Split 3 completed.")
            apply_ensemble('soft', [logit1, logit2, logit3, logit4, logit5, logit6])
            print(f"Ensemble completed.")
        else:
            print('The novelty experiment name provided by the configuration is not available. Please make sure that a valid '
                  'experiment name is given.')
    else:
        print("The provided flag is not a valid option. Please select between Baseline or Novelty and execute the main function again..")
