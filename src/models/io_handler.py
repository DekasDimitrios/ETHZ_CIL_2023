'''
    File name: io_handler.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle


def load_training_tweets_from_text_file(path):
    """
    Given a system path that corresponds to a text file this function
    is able to return a list of the tweets contained in the text file.

    :param path: string representing a valid system path
    :return: list of tweets contained in the text file provided as the function parameter.
    """
    tweets = []
    with open(path, 'r', encoding='utf-8') as file:
        for tweet in tqdm(file):
            tweets.append(tweet.rstrip('\n'))
    return tweets


def load_testing_tweets_from_text_file(path):
    """
    Given a system path that corresponds to a text file this function
    is able to return a list of the tweets contained in the text file.

    It also takes into account the structure of the test file that includes
    the index of each tweet, e.g. '1, Football is amazing #LoveTheGame'

    :param path: string representing a valid system path
    :return: list of tweets contained in the text file provided as the function parameter.
    """
    tweets = []
    with open(path, 'r', encoding='utf-8') as file:
        for tweet in tqdm(file):
            tweets.append("".join(tweet.rstrip('\n').split(',')[1:]))
    return tweets


def load_training_tweets_to_df(pos_path, neg_path, seed):
    """
    Given two file paths and a random seed the function is able to return a dataframe that contains both the
    positive and negative tweets in the dataset, also providing us with the appropriate labeling scheme.

    :param pos_path: a string representing a valid system path corresponding to the file that contain the positive tweets
    :param neg_path: a string representing a valid system path corresponding to the file that contain the negative tweets
    :param seed: an integer representing the random seed that will control the shuffling of the dataset.
    :return: a dataframe that contains the tweets in the training set
    """
    # Loading of the initial training tweets as provided to us
    train_pos_tweets = load_training_tweets_from_text_file(pos_path)
    train_pos_labels = [1] * len(train_pos_tweets)
    train_neg_tweets = load_training_tweets_from_text_file(neg_path)
    train_neg_labels = [0] * len(train_neg_tweets)

    # Unify the two types of training tweets
    train_tweets = train_pos_tweets + train_neg_tweets
    train_labels = train_pos_labels + train_neg_labels

    # Shuffle the tweets
    train_tweets, train_labels = shuffle(train_tweets, train_labels, random_state=seed)

    # Use a pandas dataframe to store them for easier handling
    train_tweet_df = pd.DataFrame({'tweet': train_tweets, 'label': train_labels})

    return train_tweet_df


def load_testing_tweets_to_df(file_path):
    """
    Given a file path the function is able to return a dataframe that contains the testing tweets in the dataset.

    :param file_path: a string representing a valid system path corresponding to the file that contain the testing tweets
    :return: a dataframe that contains the tweets in the test set
    """
    # Loading of the initial testing tweets as provided to us
    test_tweets = load_testing_tweets_from_text_file(file_path)

    # Use a pandas dataframe to store them for easier handling
    test_tweets_df = pd.DataFrame({'tweet': test_tweets})

    return test_tweets_df


def write_tweets_to_text_file(tweets, path):
    """
    Given a list of tweets and a valid system path the function can be used to store
    the tweets in a text file format inside the file specified by the system path.

    :param tweets: a list containing the tweets to be saved
    :param path: a system path that indicated the file in which the text file will be saved
    """
    with open(path, 'w') as file:
        for tweet in tweets:
            file.write(f"{tweet}\n")


def write_train_ft_tweets_to_text_file(tweets, path):
    """
    Given a list of tweets and a valid system path the function can be used to store
    the tweets in a text file format inside the file specified by the system path.

    :param tweets: a list containing the tweets to be saved
    :param path: a system path that indicated the file in which the text file will be saved
    """
    with open(path, 'w') as f:
        for index, row in tweets.iterrows():
            f.write('__label__{} {}\n'.format(row['label'], row['tweet']))


def write_eval_ft_tweets_to_text_file(tweets, path):
    """
    Given a list of tweets and a valid system path the function can be used to store
    the tweets in a text file format inside the file specified by the system path.

    :param tweets: a list containing the tweets to be saved
    :param path: a system path that indicated the file in which the text file will be saved
    """
    with open(path, 'w') as f:
        for index, row in tweets.iterrows():
            f.write('{}\n'.format(row['tweet']))


def save_test_preds(preds, path):
    """
    Given a list of predictions and a valid system path the function can be used to store
    the predictions in a csv file format inside the file specified by the system path.

    :param preds: a list containing the predictions to be saved
    :param path: a system path that indicated the file in which the csv file will be saved
    """
    if 0 in preds:
        test_preds = [-1 if p == 0 else 1 for p in preds]
    else:
        test_preds = preds
    df = pd.DataFrame(test_preds, columns=["Prediction"])
    df.index.name = "Id"
    df.index += 1
    df.to_csv(path)


def save_logits(logits, path, model_name, text_name):
    """
    Given a list of logits, a valid system path, a model name, and a text name, the function can
    be used to store the logits in a text file format inside the file specified by the system path.

    :param logits: a list containing the logits to be saved
    :param path: a system path that indicated the file in which the csv file will be saved
    :param model_name: a string representing the name of the model used to produce the logits
    :param text_name: a string representing the name used to specify the experiment that produced the logits
    """
    os.makedirs(path + model_name + "-" + text_name, exist_ok=True)
    np.savetxt(path + text_name, logits, delimiter=",", header="negative,positive", comments="")


def load_from_pkl(path):
    """
    Given valid system path the function can be used to load the
    pickled data inside the file specified by the system path.

    :param path: a system path that indicated the file in which the data will be loaded from
    :return: the loaded data
    """
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def save_to_pkl(data, path):
    """
    Given any type of data and a valid system path the function can be used to store
    the data in a pickle file format inside the file specified by the system path.

    :param data: the data to be saved
    :param path: a system path that indicated the file in which the data will be saved
    """
    with open(path, 'wb') as file:
        pickle.dump(data, file)
