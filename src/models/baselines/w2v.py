'''
    File name: w2v.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''

import numpy as np
import pandas as pd
from nltk import WhitespaceTokenizer
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from xgboost import XGBClassifier

from src.models.io_handler import load_testing_tweets_to_df, load_training_tweets_to_df
from src.models.utils import eval_test_model


def tokenize(df):
    tokenizer = WhitespaceTokenizer()
    df['tweet'] = df['tweet'].apply(lambda tweet: tokenizer.tokenize(tweet))


def tweet_embeddings(tweet, model, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tweet:
        # print(word)
        try:
            vec += model.wv[word].reshape((1, size))
            count += 1
        # throws KeyError if word not found
        except KeyError:
            continue
    # #normalize
    if count != 0:
        vec /= count
    return vec


def train_w2v(cfg):
    train_df = load_training_tweets_to_df(cfg.IO.PREPROCESSED_POS_DATA_PATH,
                                          cfg.IO.PREPROCESSED_NEG_DATA_PATH,
                                          seed=10)

    X = list(train_df["tweet"])
    y = list(train_df["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.W2V.TEST_VAL_SPLIT_RATIO,
                                                      stratify=y, random_state=cfg.SYSTEM.SEED_VALUE)

    X_test_set = load_testing_tweets_to_df(cfg.IO.PREPROCESSED_TEST_DATA_PATH)
    X_train_set = pd.DataFrame({'tweet': X_train, 'label': y_train})
    X_val_set = pd.DataFrame({'tweet': X_val, 'label': y_val})

    tokenize(X_train_set)
    tokenize(X_val_set)
    tokenize(X_test_set)

    input = X_train_set['tweet'].tolist()

    model = Word2Vec(min_count=cfg.W2V.MIN_COUNT,
                     sample=cfg.W2V.SAMPLE,
                     window=cfg.W2V.WINDOW,
                     vector_size=cfg.W2V.VECTOR_SIZE,
                     alpha=cfg.W2V.ALPHA,
                     min_alpha=cfg.W2V.MIN_ALPHA,
                     negative=cfg.W2V.NEGATIVE,
                     workers=cfg.W2V.NUM_OF_WORKERS,
                     seed=cfg.SYSTEM.SEED_VALUE)

    model.build_vocab(input)
    model.train(input, total_examples=model.corpus_count, epochs=cfg.W2V.EPOCHS)

    transformed_X_train = np.concatenate([tweet_embeddings(z, model, cfg.W2V.VECTOR_SIZE) for z in X_train_set['tweet']])
    transformed_X_val = np.concatenate([tweet_embeddings(z, model, cfg.W2V.VECTOR_SIZE) for z in X_val_set['tweet']])
    transformed_X_test = np.concatenate([tweet_embeddings(z, model, cfg.W2V.VECTOR_SIZE) for z in X_test_set['tweet']])

    model_xgb = XGBClassifier(n_estimators=cfg.W2V.NUM_OF_ESTIMATORS,
                              tree_method=cfg.W2V.TREE_METHOD,
                              objective=cfg.W2V.OBJECTIVE)

    model_xgb.fit(transformed_X_train, y_train)

    val_preds = model_xgb.predict(transformed_X_val)
    test_preds = model_xgb.predict(transformed_X_test)

    return eval_test_model(y_val, val_preds, test_preds, cfg.IO.TEST_PREDICTIONS_FILE_PATH)
