'''
    File name: glove.py
    Author: Dekas Dimitrios
    Date last modified: 07/15/2023
    Python Version: 3.9
'''

from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from nltk import WhitespaceTokenizer
from sklearn.model_selection import train_test_split

from src.models.io_handler import load_testing_tweets_to_df, load_training_tweets_to_df
from src.models.utils import eval_test_model


def tokenize(df):
    tokenizer = WhitespaceTokenizer()
    df['tweet'] = df['tweet'].apply(lambda tweet: tokenizer.tokenize(tweet))


def tweet_embedding(tokens, embeddings):
    embedding = np.zeros(200)
    count = 0
    for token in tokens:
        if token in embeddings:
            embedding += embeddings[token]
            count = count + 1
    return embedding


def train_glove(cfg):
    train_df = load_training_tweets_to_df(cfg.IO.PREPROCESSED_POS_DATA_PATH,
                                          cfg.IO.PREPROCESSED_NEG_DATA_PATH,
                                          seed=10)

    X = list(train_df["tweet"])
    y = list(train_df["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.GLOVE.TEST_VAL_SPLIT_RATIO,
                                                      stratify=y, random_state=cfg.SYSTEM.SEED_VALUE)

    X_test_set = load_testing_tweets_to_df(cfg.IO.PREPROCESSED_TEST_DATA_PATH)
    X_train_set = pd.DataFrame({'tweet': X_train, 'label': y_train})
    X_val_set = pd.DataFrame({'tweet': X_val, 'label': y_val})

    embeddings = {}
    with open(cfg.IO.GLOVE_EMBEDDING_FILE_PATH, "r") as glove:
        for line in glove:
            line = line.split()
            word = line[0]
            vec = np.asarray(line[1:], "float32")
            embeddings[word] = vec

    tokenize(X_train_set)
    tokenize(X_val_set)
    tokenize(X_test_set)

    x_train = np.vstack(X_train_set['tweet'].apply(lambda tokens: tweet_embedding(tokens, embeddings)))
    x_val = np.vstack(X_val_set['tweet'].apply(lambda tokens: tweet_embedding(tokens, embeddings)))
    x_test = np.vstack(X_test_set['tweet'].apply(lambda tokens: tweet_embedding(tokens, embeddings)))

    model = XGBClassifier(n_estimators=cfg.GLOVE.NUM_OF_ESTIMATORS,
                          subsample=cfg.GLOVE.SUBSAMPLE,
                          tree_method=cfg.GLOVE.TREE_METHOD,
                          objective=cfg.GLOVE.OBJECTIVE,
                          use_label_encoder=cfg.GLOVE.USE_LABEL_ENCODER,
                          seed=cfg.SYSTEM.SEED_VALUE)
    y_train = [0 if y == -1 else y for y in y_train]
    model.fit(x_train, y_train)

    val_preds = model.predict(x_val)
    test_preds = model.predict(x_test)

    return eval_test_model(y_val, val_preds, test_preds, cfg.IO.TEST_PREDICTIONS_FILE_PATH)
