'''
    File name: ft.py
    Author: Tsolakis Giorgos
    Code Cleaning & Integration: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''


import pandas as pd
# import fasttext
from sklearn.model_selection import train_test_split

from src.models.io_handler import load_testing_tweets_to_df, load_training_tweets_to_df, \
    write_train_ft_tweets_to_text_file, write_eval_ft_tweets_to_text_file
from src.models.utils import eval_test_model


def train_fasttext(cfg):
    """
    A function that given a config is able to train a FastText model
    and produce the respective results and logs.

    :param cfg: a yacs CfgNode object with the appropriate parameters to be used
    """

    train_df = load_training_tweets_to_df(cfg.IO.PP_POS_TWEET_FILE_PATH,
                                          cfg.IO.PP_NEG_TWEET_FILE_PATH,
                                          seed=10)

    X = list(train_df["tweet"])
    y = list(train_df["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.FASTTEXT.TRAIN_VAL_SPLIT_RATIO,
                                                      stratify=y, random_state=cfg.SYSTEM.SEED_VALUE)

    X_test_set = load_testing_tweets_to_df(cfg.IO.PP_TEST_TWEET_FILE_PATH)
    X_train_set = pd.DataFrame({'tweet': X_train, 'label': y_train})
    X_val_set = pd.DataFrame({'tweet': X_val, 'label': y_val})

    write_train_ft_tweets_to_text_file(X_train_set, cfg.IO.FT_TRAIN_DATA_FILE_PATH)
    write_eval_ft_tweets_to_text_file(X_val_set, cfg.IO.FT_VAL_DATA_FILE_PATH)
    write_eval_ft_tweets_to_text_file(X_test_set, cfg.IO.FT_TEST_DATA_FILE_PATH)

    # model = fasttext.train_supervised(input=cfg.IO.FT_TRAIN_DATA_FILE_PATH,
    #                             lr=cfg.FASTTEXT.LR,
    #                             dim=cfg.FASTTEXT.DIMENSION,
    #                             epoch=cfg.FASTTEXT.EPOCH,
    #                             seed=cfg.SYSTEM.SEED_VALUE)
    #
    # with open(cfg.IO.FT_VAL_DATA_FILE_PATH) as f:
    #     val_data = [line.strip() for line in f]
    # val_preds = [int(model.predict(text)[0][0].replace('__label__', '')) for text in val_data]
    #
    #
    # with open(cfg.IO.FT_TEST_DATA_FILE_PATH) as f:
    #     test_data = [line.strip() for line in f]
    #
    # test_preds = [int(model.predict(text)[0][0].replace('__label__', '')) for text in test_data]
    #
    # return eval_test_model(y_val, val_preds, test_preds, cfg.IO.TEST_PREDICTIONS_FILE_PATH)
