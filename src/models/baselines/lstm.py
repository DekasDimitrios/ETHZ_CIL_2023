'''
    File name: lstm.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split


from src.models.io_handler import load_testing_tweets_to_df, load_training_tweets_to_df


def train_lstm(cfg):
    tf.random.set_seed(cfg.SYSTEM.SEED_VALUE)
    train_df = load_training_tweets_to_df(cfg.IO.PREPROCESSED_POS_DATA_PATH,
                                          cfg.IO.PREPROCESSED_NEG_DATA_PATH,
                                          seed=10)

    X = list(train_df["tweet"])
    y = list(train_df["label"])

    # Load Data
    tweets = np.array(X)
    labels = np.array(y)

    X_train_set = pd.DataFrame(columns=['tweet', 'label'], data=np.array([tweets, labels]).T)

    # Tokenize tweets and pad
    tokenizer = Tokenizer(num_words=cfg.LSTM.MAXIMUM_FEATURES, split=' ')
    tokenizer.fit_on_texts(X_train_set['tweet'].values)
    X = tokenizer.texts_to_sequences(X_train_set['tweet'].values)
    X = pad_sequences(X, maxlen=cfg.LSTM.MAXIMUM_LENGTH)

    # Define Bidirectional LSTM layer
    LSTMLayer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(cfg.LSTM.UNITS,
                                                                   dropout=cfg.LSTM.DROPOUT,
                                                                   recurrent_dropout=cfg.LSTM.RECURRENT_DROPOUT))

    # Simple LSTM model (Noticed that the stacked one would perform worse)
    model = Sequential()
    model.add(Embedding(cfg.LSTM.MAXIMUM_FEATURES,
                        cfg.LSTM.EMBEDDING_DIMENSION,
                        input_length=X.shape[1]))
    model.add(SpatialDropout1D(cfg.LSTM.SPATIAL_DROPOUT))
    model.add(LSTMLayer)
    model.add(Dense(cfg.LSTM.DENSE_LAYER_SIZE, activation=cfg.LSTM.DENSE_LAYER_ACTIVATION))
    model.compile(optimizer=cfg.LSTM.MODEL_OPTIMIZER, loss=cfg.LSTM.MODEL_LOSS, metrics=['accuracy'])

    # Create Validation set
    Y = pd.get_dummies(X_train_set['label']).values
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=cfg.LSTM.TEST_VAL_SPLIT_RATIO, random_state=cfg.SYSTEM.SEED_VALUE)

    # Train Model
    history = model.fit(X_train, Y_train, validation_data=[X_val, Y_val], epochs=cfg.LSTM.EPOCHS,
                        steps_per_epoch=cfg.LSTM.STEPS_PER_EPOCH, batch_size=cfg.LSTM.BATCH_SIZE, verbose=cfg.LSTM.VERBOSE)

    X_test_set = load_testing_tweets_to_df(cfg.IO.PREPROCESSED_TEST_DATA_PATH)
    X_test = tokenizer.texts_to_sequences(X_test_set['tweet'].values)
    X_test = pad_sequences(X_test, maxlen=cfg.LSTM.MAXIMUM_LENGTH)
    y_test = model.predict(X_test, batch_size=len(X_test))
    y_preds = [-1 if np.argmax(val) == 0 else 1 for val in y_test]
    df = pd.DataFrame(y_preds, columns=["Prediction"])
    df.index.name = "Id"
    df.index += 1
    df.to_csv(cfg.IO.TEST_PREDICTIONS_FILE_PATH)
