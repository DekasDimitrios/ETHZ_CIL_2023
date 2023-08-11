'''
    File name: tf_idf.py
    Author: Tsolakis Giorgos
    Code Cleaning & Integration: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from src.models.io_handler import load_testing_tweets_to_df, load_training_tweets_to_df
from sklearn.ensemble import RandomForestClassifier

from src.models.utils import eval_test_model


def fit_linear_model(X_train, y_train, X_val, y_val, X_test, seed, cw, penalty, test_pred_fp):
    """
    A function that is used to train a Linear SVC.

    :param X_train: a list containing the training samples used to train the model
    :param y_train: a list containing the training labels used to train the model
    :param X_val: a list containing the validation samples used to evaluate the model
    :param y_val: a list containing the validation labels used to evaluate the model
    :param X_test: a list containing the testing samples used to produce the test prediction using the trained model
    :param seed: an integer representing the random seem used by the model
    :param cw: a float representing the weight attributed to each class during training
    :param penalty: a float representing the penalty used by the LinearSVC
    :param test_pred_fp: a string representing a valid system path used to store the testing predictions of the model
    :return: an integer representing the validation accuracy achieved by the trained model
    """
    model = LinearSVC(class_weight=cw,
                      random_state=seed,
                      penalty=penalty)

    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    return eval_test_model(y_val, val_preds, test_preds, test_pred_fp)


def fit_non_linear_model(X_train, y_train, X_val, y_val, X_test, seed, cw, n_estimators, max_depth, verb, test_pred_fp):
    """
    A function that is used to train a RandomForestClassifier.

    :param X_train: a list containing the training samples used to train the model
    :param y_train: a list containing the training labels used to train the model
    :param X_val: a list containing the validation samples used to evaluate the model
    :param y_val: a list containing the validation labels used to evaluate the model
    :param X_test: a list containing the testing samples used to produce the test prediction using the trained model
    :param seed: an integer representing the random seem used by the model
    :param cw: a float representing the weight attributed to each class during training
    :param n_estimators: an integer representing the number of estimators used by the RandomForestClassifier
    :param max_depth: an integer representing the maximum depth of each individual tree in the RandomForestClassifier
    :param verb: an integer representing the verbosity level while training the RandomForestClassifier
    :param test_pred_fp: a string representing a valid system path used to store the testing predictions of the model
    :return: an integer representing the validation accuracy achieved by the trained model
    """
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   random_state=seed,
                                   class_weight=cw,
                                   verbose=verb)

    model.fit(X_train, y_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    return eval_test_model(y_val, val_preds, test_preds, test_pred_fp)


def train_tf_idf(cfg):
    """
    A function that given a config is able to train both a linear and a non-linear classifier
    using TF_IDF embeddings and produce the respective results and logs.

    :param cfg: a yacs CfgNode object with the appropriate parameters to be used
    """
    train_df = load_training_tweets_to_df(cfg.IO.PP_POS_TWEET_FILE_PATH,
                                          cfg.IO.PP_NEG_TWEET_FILE_PATH,
                                          seed=10)

    X = list(train_df["tweet"])
    y = list(train_df["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.TF_IDF.TRAIN_VAL_SPLIT_RATIO,
                                                      stratify=y, random_state=cfg.SYSTEM.SEED_VALUE)

    X_test = load_testing_tweets_to_df(cfg.IO.PP_TEST_TWEET_FILE_PATH)
    X_test = list(X_test['tweet'])

    # Fit TD-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # Fit Standard Scaler
    scaler = StandardScaler(with_mean=False)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if cfg.TF_IDF.MODEL == 'linear':
        val_score = fit_linear_model(X_train, y_train, X_val, y_val, X_test,
                                     cfg.SYSTEM.SEED_VALUE, cfg.TF_IDF.CW, cfg.TF_IDF.PENALTY, cfg.IO.TEST_PREDICTIONS_FILE_PATH)
    elif cfg.TF_IDF.MODEL == 'non-linear':
        val_score = fit_non_linear_model(X_train, y_train, X_val, y_val, X_test,
                                         cfg.SYSTEM.SEED_VALUE, cfg.TF_IDF.CW, cfg.TF_IDF.NUM_OF_ESTIMATORS, cfg.TF_IDF.MAX_DEPTH, cfg.TF_IDF.VERBOSE, cfg.IO.TEST_PREDICTIONS_FILE_PATH)
    else:
        print('An unexpected model option was provided by the configuration file. Therefore, the linear model shall '
              'be used.')
        val_score = fit_linear_model(X_train, y_train, X_val, y_val, X_test,
                                     cfg.SYSTEM.SEED_VALUE, cfg.TF_IDF.CW, cfg.TF_IDF.PENALTY, cfg.IO.TEST_PREDICTIONS_FILE_PATH)

    return val_score