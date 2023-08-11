'''
    File name: utils.py
    Author: Andreas Psaroudakis / Kostas Fanaras
    Code Cleaning & Integration: Dekas Dimitrios
    Date last modified: 07/15/2023
    Python Version: 3.9
'''


import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

from src.models.io_handler import save_test_preds, load_from_pkl, save_to_pkl


def eval_test_model(y_val, val_preds, test_preds, test_out_path):
    """
    A function that produces the validation accuracy of a model and saves its test predictions

    :param y_val: a list representing the ground truth labels of validation samples given to a model
    :param val_preds: a list representing the predictions of a model on the validation set
    :param test_preds: a list representing the predictions of a model on the testing set
    :param test_out_path: a string representing a valid system path that the testing predictions are going to be saved to
    :return: a float representing the validation accuracy achieved by a model given its predicitions
    """
    accuracy = accuracy_score(y_val, val_preds)
    save_test_preds(test_preds, test_out_path)
    return accuracy


def numpy_softmax(model_preds):
    """
    A function used to convert the raw predictions from a HuggingFace model into clean logits.

    :param model_preds: a list containing the predictions give by a model
    :return: a list containing the logits produced by a softmax operation of the predictions
    """
    max = np.max(model_preds, axis=1, keepdims=True)
    e_x = np.exp(model_preds-max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    out = e_x / sum
    return out


def compute_metrics(pred):
    """
    A function that can compute the f1 and accuracy score when given to a Trainer object as parameter

    :param pred: an EvalPrediction object that represents the predictions made by a model
    :return: a dictionary containing the accuracy and f1-score of the trained model
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def tokenization(tokenizer, df, tok_max_length):
    """
    A function that is able to create a tokenized version of the strings
    contained in a 'tweet' column of a dataframe.

    :param tokenizer: a tokenized object used to produce the tokens
    :param df: a pandas dataframe containing the samples to be tokenized
    :param tok_max_length: an integer representing the maximum number of tokens allowed for a single sample
    :return:
    """
    return tokenizer(df["tweet"], truncation=True, max_length=tok_max_length, padding=True)


def get_wrongly_pred_samples(X, y_pred, y_true):
    """
    A function that produces two lists that containing the samples and labels
    that were wrongly classified by a model

    :param X: a list containing the samples given to a model
    :param y_pred: a list representing the predictions of a model on the validation set
    :param y_true: a list representing the ground truth labels of validation samples given to a model
    :return: two lists were the first one contains the samples and the second one the labels that were not classified correctly
    """
    wrong_pred_X = []
    wrong_pred_Y = []

    for i in range(len(X)):
        if y_pred[i] != y_true[i]:
            wrong_pred_X.append(X[i])
            wrong_pred_Y.append(y_true[i])

    return wrong_pred_X, wrong_pred_Y


def sort_by_difficulty(dataset, dict_path, tokenizer, tml):
    """
    A function that is able to sort a list of samples based on their individual difficulty.

    :param dataset: a Dataset object that containing the sample collection to be order by difficulty
    :param dict_path: a string representing a valid system path containing the dictionary used to rank the samples
    :param tokenizer: a tokenizer object used to produce the tokenization of individual samples
    :param tml: an integer representing the maximum length of tokens allowed
    :return: a Dataset object containing the sorted sample collection
    """
    subset_X = dataset['tweet']
    subset_y = dataset['label']
    scores_dict = load_from_pkl(dict_path)

    new_dict = {}
    for i, x in enumerate(subset_X):
      new_dict[i] = {'x': x, 'label': subset_y[i], 'score': scores_dict[x]['score']}

    new_sorted_dict = dict(sorted(new_dict.items(), key=lambda item: item[1]['score'], reverse=False))
    sorted_X = [item[1]['x'] for item in new_sorted_dict.items()]
    sorted_y = [item[1]['label'] for item in new_sorted_dict.items()]

    new_data = {"tweet": sorted_X, "label": sorted_y}
    new_dataset = Dataset.from_dict(new_data)
    new_tokenized_dataset = new_dataset.map(lambda examples: tokenization(tokenizer, examples, tml), batched=True)

    return new_tokenized_dataset


def keep_distribution(train_dataset_sorted, tokenizer, tok_max_length):
    """
    A function used to maintain the true ratio of positive and negative samples in a sorted train dataset.
    It takes the sorted train dataset and performs the following steps: it separates
    the positive and negative samples based on their labels, calculates the true ratio,
    and alternates the samples accordingly to maintain the true ratio. Finally, it
    constructs a new dataset from the alternating samples, preprocesses it, and returns
    the preprocessed tokenized dataset.

    :param train_dataset_sorted:
    :param tokenizer:
    :param tok_max_length:
    :return:
    """
    pos_tweets, pos_labels = list(zip(*filter(lambda t: t[1] == 1, zip(train_dataset_sorted['tweet'], train_dataset_sorted['label']))))
    neg_tweets, neg_labels = list(zip(*filter(lambda t: t[1] == 0, zip(train_dataset_sorted['tweet'], train_dataset_sorted['label']))))

    positive_samples, negative_samples = [pos_tweets, pos_labels], [neg_tweets, neg_labels]
    n = len(positive_samples[0]) + len(negative_samples[0])
    results = []
    for j in range(2):
        true_ratio = len(positive_samples[j]) / n
        alternating_samples = []
        pos_count, neg_count = 0, 0
        running_ratio = 0
        for i in range(n):
            if running_ratio < true_ratio:
                alternating_samples.append(positive_samples[j][pos_count])
                pos_count += 1
            else:
                alternating_samples.append(negative_samples[j][neg_count])
                neg_count += 1
            running_ratio = pos_count / (pos_count + neg_count)
        results.append(alternating_samples)

    new_data = {"tweet": results[0], "label": results[1]}
    new_dataset = Dataset.from_dict(new_data)
    new_tokenized_dataset = new_dataset.map(lambda examples: tokenization(tokenizer, examples, tok_max_length), batched=True)

    return new_tokenized_dataset


def concat_datasets(Dataset1, Dataset2):
    """
    A function that given two dataset objects is able to concatenate them and return a unified dataset object.

    :param Dataset1: a dataset object representing the first dataset that is going to participate in the union
    :param Dataset2: a dataset object representing the second dataset that is going to participate in the union
    :return: a dataset object representing the union of the two datasets.
    """
    df1 = Dataset1.to_pandas()
    df2 = Dataset2.to_pandas()
    concatenated_df = pd.concat([df1, df2], ignore_index=True)
    concatenated_dataset = Dataset.from_pandas(concatenated_df)
    return concatenated_dataset