'''
    File name: bert_xgb.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from xgboost import XGBClassifier
from tqdm import tqdm

from src.models.io_handler import load_testing_tweets_to_df, load_training_tweets_to_df
from src.models.utils import eval_test_model


def embedding(tweet, model):
    return model.embeddings.word_embeddings.weight[tweet].mean(0)


def tokenize(tweet, tokenizer, max_len):
    return tokenizer.encode(tweet, max_length=max_len, truncation=True)


def train_bert_xgb(cfg):
    train_df = load_training_tweets_to_df(cfg.IO.PREPROCESSED_POS_DATA_PATH,
                                          cfg.IO.PREPROCESSED_NEG_DATA_PATH,
                                          seed=10)

    X = list(train_df["tweet"])
    y = list(train_df["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.GLOVE.TEST_VAL_SPLIT_RATIO,
                                                      stratify=y, random_state=cfg.SYSTEM.SEED_VALUE)

    X_test_set = load_testing_tweets_to_df(cfg.IO.PREPROCESSED_TEST_DATA_PATH)

    # Load the Bertweet tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=cfg.BERTXGB.MAXIMUM_LENGTH)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(cfg.BERTXGB.DEVICE)

    X_train = np.vstack([embedding(tokenize(z, tokenizer, cfg.BERTXGB.MAXIMUM_LENGTH), model).detach().cpu().numpy() for z in tqdm(X_train)])
    y_train = [0 if y == -1 else y for y in y_train]
    model_xgb = XGBClassifier(n_estimators=cfg.BERTXGB.NUM_OF_ESTIMATORS,
                              tree_method=cfg.BERTXGB.TREE_METHOD,
                              objective=cfg.BERTXGB.OBJECTIVE,
                              use_label_encoder=cfg.BERTXGB.USE_LABEL_ENCODER,
                              seed=cfg.SYSTEM.SEED_VALUE)

    model_xgb.fit(X_train, y_train)

    X_val = np.vstack([embedding(tokenize(z, tokenizer, cfg.BERTXGB.MAXIMUM_LENGTH), model).detach().cpu().numpy() for z in tqdm(X_val)])
    X_test = np.vstack([embedding(tokenize(z, tokenizer, cfg.BERTXGB.MAXIMUM_LENGTH), model).detach().cpu().numpy() for z in tqdm(X_test_set['tweet'].values)])

    val_preds = model_xgb.predict(X_val)
    test_preds = model_xgb.predict(X_test)

    return eval_test_model(y_val, val_preds, test_preds, cfg.IO.TEST_PREDICTIONS_FILE_PATH)