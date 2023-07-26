'''
    File name: config.py
    Author: Dekas Dimitrios
    Date last modified: 07/15/2023
    Python Version: 3.9
'''


from yacs.config import CfgNode as CN

# Create initial config
_C = CN()

# Create System Variables config
_C.SYSTEM = CN()
_C.SYSTEM.SEED_VALUE = 12222
_C.SYSTEM.FP16 = False

# Create Experiment Variables config
_C.EXPERIMENT = CN()
_C.EXPERIMENT.NAME = 'TF-IDF'

# Create Input/Output Variables config
_C.IO = CN()
_C.IO.PROJECT_PATH = './'
_C.IO.EXPERIMENT_PATH = './Experiments/'
_C.IO.DICTIONARY_PATH = './scores_dict.pkl'
_C.IO.LOG_PATH = './logging'
_C.IO.PREPROCESSED_POS_DATA_PATH = '../processed_pos_tweets_non_transformer.txt'
_C.IO.PREPROCESSED_NEG_DATA_PATH = '../processed_neg_tweets_non_transformer.txt'
_C.IO.PREPROCESSED_TEST_DATA_PATH = '../processed_test_tweets_non_transformer.txt'
_C.IO.FT_TRAIN_DATA_FILE_PATH = './train_data.txt'
_C.IO.FT_VAL_DATA_FILE_PATH = './val_data.txt'
_C.IO.FT_TEST_DATA_FILE_PATH = './test_data.txt'
_C.IO.TEST_PREDICTIONS_FILE_PATH = './test_predictions.csv'
_C.IO.GLOVE_EMBEDDING_FILE_PATH = './glove.twitter.27B.200d.txt'

# Create TF-IDF Model Hyperparameters config
_C.TF_IDF = CN()
_C.TF_IDF.TEST_VAL_SPLIT_RATIO = 0.05
_C.TF_IDF.MODEL = 'linear'
_C.TF_IDF.CW = 'balanced'
_C.TF_IDF.PENALTY = 'l2'
_C.TF_IDF.NUM_OF_ESTIMATORS = 200
_C.TF_IDF.MAX_DEPTH = 11
_C.TF_IDF.VERBOSE = 0

# Create FastText Model Hyperparameters config
_C.FASTTEXT = CN()
_C.FASTTEXT.TEST_VAL_SPLIT_RATIO = 0.05
_C.FASTTEXT.LR = 0.01
_C.FASTTEXT.DIMENSION = 150
_C.FASTTEXT.EPOCH = 20

# Create GloVe Model Hyperparameters config
_C.GLOVE = CN()
_C.GLOVE.TEST_VAL_SPLIT_RATIO = 0.05
_C.GLOVE.NUM_OF_ESTIMATORS = 850
_C.GLOVE.SUBSAMPLE = 0.8
_C.GLOVE.TREE_METHOD = 'gpu_hist'
_C.GLOVE.OBJECTIVE = 'binary:logistic'
_C.GLOVE.USE_LABEL_ENCODER = False

# Create Word2Vec Model Hyperparameters config
_C.W2V = CN()
_C.W2V.TEST_VAL_SPLIT_RATIO = 0.05
_C.W2V.MIN_COUNT = 5
_C.W2V.SAMPLE = 5e-5
_C.W2V.WINDOW = 3
_C.W2V.ALPHA = 0.035
_C.W2V.MIN_ALPHA = 0.00075
_C.W2V.NEGATIVE = 5
_C.W2V.NUM_OF_WORKERS = 4
_C.W2V.VECTOR_SIZE = 250
_C.W2V.EPOCHS = 25
_C.W2V.NUM_OF_ESTIMATORS = 1250
_C.W2V.TREE_METHOD = 'gpu_hist'
_C.W2V.OBJECTIVE = 'binary:logistic'

# Create LSTM Model Hyperparameters config
_C.LSTM = CN()
_C.LSTM.TEST_VAL_SPLIT_RATIO = 0.05
_C.LSTM.EMBEDDING_DIMENSION = 500
_C.LSTM.MAXIMUM_FEATURES = 15000
_C.LSTM.MAXIMUM_LENGTH = 100
_C.LSTM.UNITS = 200
_C.LSTM.DROPOUT = 0.3
_C.LSTM.RECURRENT_DROPOUT = 0.25
_C.LSTM.SPATIAL_DROPOUT = 0.35
_C.LSTM.DENSE_LAYER_SIZE = 2
_C.LSTM.DENSE_LAYER_ACTIVATION = 'softmax'
_C.LSTM.MODEL_OPTIMIZER = 'adam'
_C.LSTM.MODEL_LOSS = 'categorical_crossentropy'
_C.LSTM.BATCH_SIZE = 128
_C.LSTM.EPOCHS = 25
_C.LSTM.STEPS_PER_EPOCH = 300
_C.LSTM.VERBOSE = 1

# Create Bert-XGB Model Hyperparameters config
_C.BERTXGB = CN()
_C.BERTXGB.TEST_VAL_SPLIT_RATIO = 0.05
_C.BERTXGB.DEVICE = 'cpu'
_C.BERTXGB.MAXIMUM_LENGTH = 128
_C.BERTXGB.NUM_OF_ESTIMATORS = 850
_C.BERTXGB.TREE_METHOD = 'gpu_hist'
_C.BERTXGB.OBJECTIVE = 'binary:logistic'
_C.BERTXGB.USE_LABEL_ENCODER = False

# Create Novelty Hyperparameters config
_C.NHP = CN()
_C.NHP.MODEL_NAME = "bert-base-uncased"
_C.NHP.BATCH_SIZE = 32
_C.NHP.EPOCHS = 3
_C.NHP.LR = 1e-4
_C.NHP.WD = 0.005
_C.NHP.TOK_MAX_LENGTH = 128


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def update_cfg(cfg_file):
    """
    Given a valid system path in the form of a string this function is able to return a
    config version that incorporated the values found in the file into the default ones.

    :param cfg_file: a string representing a valid system path
    :return: the updated configuration for our preprocessing
    """
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()
