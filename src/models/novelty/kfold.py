'''
    File name: kfold.py
    Author: Andreas Psaroudakis / Kostas Fanaras
    Code Cleaning & Integration: Dekas Dimitrios
    Date last modified: 07/15/2023
    Python Version: 3.9
'''


import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
import os
from datetime import datetime
from datasets import DatasetDict, Dataset

from src.models.utils import tokenization
from src.models.io_handler import load_training_tweets_to_df, save_to_pkl


def prod_k_fold(cfg):
    """
    A function that given a config is able to produce the
    train validation splits used to train our novel approach.

    :param cfg: a yacs CfgNode object with the appropriate parameters to be used
    """

    # Select GPU device if available, reset cache, set seeds
    torch.cuda.empty_cache()
    np.random.seed(cfg.SYSTEM.SEED_VALUE)
    torch.manual_seed(cfg.SYSTEM.SEED_VALUE)

    # Create filing system structure
    experiment_date_for_folder_name = "experiment-" + cfg.PARAMETERS.MODEL_NAME + '_' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    experiments_results_path = cfg.IO.EXPERIMENT_PATH + experiment_date_for_folder_name
    val_results_path = experiments_results_path + "/val_results/"
    test_results_path = experiments_results_path + "/test_results/"
    os.makedirs(experiments_results_path, exist_ok=True)
    os.makedirs(val_results_path, exist_ok=True)
    os.makedirs(test_results_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.PARAMETERS.MODEL_NAME, use_fast=True)

    # Load data into appropriate datasets and tokenize them
    data = load_training_tweets_to_df(cfg.IO.PP_POS_TWEET_FILE_PATH,
                                      cfg.IO.PP_NEG_TWEET_FILE_PATH,
                                      10)

    X = list(data["tweet"])
    y = list(data["label"])

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.SYSTEM.SEED_VALUE)
    train_val_folds = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):

        train_tweets_fold = [X[i] for i in train_index]
        train_labels_fold = [y[i] for i in train_index]

        val_tweets_fold = [X[i] for i in val_index]
        val_labels_fold = [y[i] for i in val_index]

        train_data_fold = {"tweet": train_tweets_fold, "label": train_labels_fold}
        train_dataset_fold = Dataset.from_dict(train_data_fold)

        val_data_fold = {"tweet": val_tweets_fold, "label": val_labels_fold}
        val_dataset_fold = Dataset.from_dict(val_data_fold)

        data_dict_fold = DatasetDict({"train": train_dataset_fold, "validation": val_dataset_fold})
        tokenized_dataset_fold = data_dict_fold.map(lambda examples: tokenization(tokenizer, examples, cfg.PARAMETERS.TOK_MAX_LENGTH), batched=True)
        train_val_folds.append((tokenized_dataset_fold["train"], tokenized_dataset_fold["validation"]))

    save_to_pkl(train_val_folds, cfg.IO.TRAIN_VAL_FOLDS_PATH)
