'''
    File name: __init__.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''

from src.preprocess.io_handler import load_training_tweets_to_df, load_testing_tweets_to_df, \
    write_tweets_to_text_file, load_extra_dataset
from src.preprocess.duplicates import drop_duplicates, drop_near_duplicates
from src.preprocess.ekp import apply_ekphrasis
from src.preprocess.word_level import apply_word_level
from src.preprocess.config import update_cfg
import argparse
import pandas as pd


def run(cfg_path):
    """
    Used to execute the code necessary to carry out the preprocessing pipeline
    as it is dictated by the config file given as argument.

    :param cfg_path: a valid system path for the config file to be used
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=cfg_path,
                        help='path to the configuration yaml file')

    args = parser.parse_args()
    cfg = update_cfg(args.cfg)
    cfg.freeze()
    print("Loading Dataset.")

    # Loading the tweets used for training purposes into a dataframe
    train_tweets_df = load_training_tweets_to_df(cfg.IO.POS_TWEET_FILE_PATH, cfg.IO.NEG_TWEET_FILE_PATH, cfg.SYSTEM.SEED_VALUE)
    print("Training set loaded successfully.")

    # Loading the tweets used for testing purposes
    test_tweets_df = load_testing_tweets_to_df(cfg.IO.TEST_TWEET_FILE_PATH)
    print("Testing set loaded successfully.")

    # If you decide to drop the use extra data, proceed in doing so
    if cfg.SYSTEM.USE_EXTRA_DATASET:
        extra_data_df = load_extra_dataset(cfg.IO.EXTRA_DATA_FILE_PATH, cfg.SYSTEM.SEED_VALUE)
        train_tweets_df = pd.concat([train_tweets_df, extra_data_df]).reset_index(drop=True)
        print("Additional dataset added to the training set successfully")

    # If you decide to drop the near duplicate tweets, proceed in doing so
    if cfg.PP.DROP_DUPLICATES:
        train_tweets_df = drop_duplicates(train_tweets_df, cfg.PP.AMBIGUOUS_DROP_FACTOR)

    # If you decide to drop the duplicate tweets, proceed in doing so
    if cfg.PP.DROP_NEAR_DUPLICATES:
        print("Dropping near duplicates.")
        train_tweets_df = drop_near_duplicates(train_tweets_df, cfg.SYSTEM.SEED_VALUE)

    # If you decide to use the ekphrasis library, proceed in doing so
    if cfg.PP.USE_EKPHRASIS:
        print("Applying ekphrasis.")
        train_tweets_df = apply_ekphrasis(train_tweets_df, cfg.EKPHRASIS)
        test_tweets_df = apply_ekphrasis(test_tweets_df, cfg.EKPHRASIS)

    # If you decide to use word level preprocessing transformations, proceed in doing so
    if cfg.PP.USE_WORD_LVL:
        print("Applying word level transformations.")
        train_tweets_df = apply_word_level(train_tweets_df, cfg.WORD_LVL)
        test_tweets_df = apply_word_level(test_tweets_df, cfg.WORD_LVL)

    # If there was not preprocessing set applied still save the new dataset in the same text file naming format
    if 'processed_tweet' not in train_tweets_df.columns:
        train_tweets_df['processed_tweet'] = train_tweets_df['tweet']
        test_tweets_df['processed_tweet'] = test_tweets_df['tweet']

    # Save the results of the preprocessing in newly created text files
    print("Saving Preprocessed Results")

    neg_df = train_tweets_df.loc[train_tweets_df['label'] == -1]
    pos_df = train_tweets_df.loc[train_tweets_df['label'] == 1]

    write_tweets_to_text_file(pos_df['processed_tweet'].tolist(), cfg.IO.PP_NEG_TWEET_FILE_PATH)
    write_tweets_to_text_file(neg_df['processed_tweet'].tolist(), cfg.IO.PP_POS_TWEET_FILE_PATH)
    write_tweets_to_text_file(test_tweets_df['processed_tweet'].tolist(), cfg.IO.PP_TEST_TWEET_FILE_PATH)
