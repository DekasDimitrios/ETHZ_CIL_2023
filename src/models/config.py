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
_C.IO.EXPERIMENT_PATH = '../runs/Experiments/'
_C.IO.SCORES_DICTIONARY_PATH = '../data/scores_dict.pkl'
_C.IO.TRAIN_VAL_FOLDS_PATH = '../data/train_val_folds.pkl'
_C.IO.LOG_PATH = '../runs/logging'
_C.IO.PP_NEG_TWEET_FILE_PATH = '../data/preprocessed/pp_train_neg_full.txt'
_C.IO.PP_POS_TWEET_FILE_PATH = '../data/preprocessed/pp_train_pos_full.txt'
_C.IO.PP_TEST_TWEET_FILE_PATH = '../data/preprocessed/pp_test_data.txt'
_C.IO.FT_TRAIN_DATA_FILE_PATH = './train_data.txt'
_C.IO.FT_VAL_DATA_FILE_PATH = './val_data.txt'
_C.IO.FT_TEST_DATA_FILE_PATH = './test_data.txt'
_C.IO.TEST_PREDICTIONS_FILE_PATH = './test_predictions.csv'
_C.IO.GLOVE_EMBEDDING_FILE_PATH = './glove.twitter.27B.200d.txt'

# Create TF-IDF Model Hyperparameters config
_C.TF_IDF = CN()
_C.TF_IDF.TRAIN_VAL_SPLIT_RATIO = 0.25
_C.TF_IDF.MODEL = 'linear'
_C.TF_IDF.CW = 'balanced'
_C.TF_IDF.PENALTY = 'l2'
_C.TF_IDF.NUM_OF_ESTIMATORS = 200
_C.TF_IDF.MAX_DEPTH = 11
_C.TF_IDF.VERBOSE = 0

# Create FastText Model Hyperparameters config
_C.FASTTEXT = CN()
_C.FASTTEXT.TRAIN_VAL_SPLIT_RATIO = 0.25
_C.FASTTEXT.LR = 0.01
_C.FASTTEXT.DIMENSION = 150
_C.FASTTEXT.EPOCH = 20

# Create GloVe Model Hyperparameters config
_C.GLOVE = CN()
_C.GLOVE.TRAIN_VAL_SPLIT_RATIO = 0.25
_C.GLOVE.NUM_OF_ESTIMATORS = 850
_C.GLOVE.SUBSAMPLE = 0.8
_C.GLOVE.TREE_METHOD = 'gpu_hist'
_C.GLOVE.OBJECTIVE = 'binary:logistic'
_C.GLOVE.USE_LABEL_ENCODER = False

# Create Word2Vec Model Hyperparameters config
_C.W2V = CN()
_C.W2V.TRAIN_VAL_SPLIT_RATIO = 0.25
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
_C.LSTM.TRAIN_VAL_SPLIT_RATIO = 0.25
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
_C.BERTXGB.TRAIN_VAL_SPLIT_RATIO = 0.25
_C.BERTXGB.DEVICE = 'cpu'
_C.BERTXGB.MAXIMUM_LENGTH = 128
_C.BERTXGB.NUM_OF_ESTIMATORS = 850
_C.BERTXGB.TREE_METHOD = 'gpu_hist'
_C.BERTXGB.OBJECTIVE = 'binary:logistic'
_C.BERTXGB.USE_LABEL_ENCODER = False

# Create Novelty Hyperparameters config
_C.PARAMETERS = CN()
_C.PARAMETERS.TRAIN_VAL_SPLIT_RATIO = 0.25
_C.PARAMETERS.MODEL_NAME = "vinai/bertweet-base"
_C.PARAMETERS.NUM_OF_LABELS = 2
_C.PARAMETERS.BATCH_SIZE = 32
_C.PARAMETERS.EPOCHS = 2
_C.PARAMETERS.LR = 1e-4
_C.PARAMETERS.WD = 0.005
_C.PARAMETERS.TOK_MAX_LENGTH = 128
_C.PARAMETERS.LOGGING_STEPS = 4000
_C.PARAMETERS.WARMUP_STEPS = 500
_C.PARAMETERS.SAVE_TOTAL_LIMIT = 2
_C.PARAMETERS.EVAL_STRATEGY = 'epoch'
_C.PARAMETERS.LOGGING_STRATEGY = 'epoch'
_C.PARAMETERS.SAVE_STRATEGY = 'epoch'
_C.PARAMETERS.GRAD_ACCUM_STEPS = 4
_C.PARAMETERS.DISABLE_TQDM = False
_C.PARAMETERS.LOAD_BEST_MODEL_AT_END = True


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
