'''
    File name: config.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''


from yacs.config import CfgNode as CN

# Create initial config
_C = CN()

# Create System Variables config
_C.SYSTEM = CN()
_C.SYSTEM.SEED_VALUE = 42
_C.SYSTEM.USE_EXTRA_DATASET = False

# Create Input/Output Variables config
_C.IO = CN()
_C.IO.NEG_TWEET_FILE_PATH = '..data/train_neg_full.txt'
_C.IO.POS_TWEET_FILE_PATH = '../data/train_pos_full.txt'
_C.IO.TEST_TWEET_FILE_PATH = '../data/test_data.txt'
_C.IO.PP_NEG_TWEET_FILE_PATH = '../data/preprocessed/pp_train_neg_full.txt'
_C.IO.PP_POS_TWEET_FILE_PATH = '../data/preprocessed/pp_train_pos_full.txt'
_C.IO.PP_TEST_TWEET_FILE_PATH = '../data/preprocessed/pp_test_data.txt'
_C.IO.EXTRA_DATA_FILE_PATH = '../data/additional/training.1600000.processed.noemoticon.csv'

# Create Preprocessing Steps Variables config
_C.PP = CN()
_C.PP.DROP_DUPLICATES = False
_C.PP.AMBIGUOUS_DROP_FACTOR = 0.51    # Only relevant if DROP_DUPLICATES = True
_C.PP.DROP_NEAR_DUPLICATES = False
_C.PP.USE_EKPHRASIS = False
_C.PP.USE_WORD_LVL = False

# Create Ekphrasis Variables config
_C.EKPHRASIS = CN()
_C.EKPHRASIS.OMIT = ['email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'hashtag']
_C.EKPHRASIS.NORMALIZE = ['email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'hashtag']
_C.EKPHRASIS.UNPACK_CONTRACTIONS = False
_C.EKPHRASIS.UNPACK_HASHTAGS = False
_C.EKPHRASIS.ANNOTATE = ['hashtag', 'allcaps', 'elongated', 'repeated']
_C.EKPHRASIS.SEGMENTOR = "twitter"
_C.EKPHRASIS.CORRECTOR = "twitter"
_C.EKPHRASIS.ALL_CAPS_TAG = "wrap"
_C.EKPHRASIS.SPELL_CORRECT_ELONGATED = False
_C.EKPHRASIS.SPELL_CORRECTION = False
_C.EKPHRASIS.FIX_TEXT = False
_C.EKPHRASIS.FIX_HTML = False
_C.EKPHRASIS.REMOVE_TAGS = False
_C.EKPHRASIS.DICTIONARIES = ['emoticons', 'slang_dict']
_C.EKPHRASIS.TOKENIZER = 'Whitespace'

# Create Word Level Variables config
_C.WORD_LVL = CN()
_C.WORD_LVL.LOWER_CASE = False
_C.WORD_LVL.REMOVE_USER = False
_C.WORD_LVL.CHANGE_USER = True
_C.WORD_LVL.USER_REPRESENTATION = '@USER'
_C.WORD_LVL.REMOVE_URL = False
_C.WORD_LVL.CHANGE_URL = True
_C.WORD_LVL.URL_REPRESENTATION = 'HTTPURL'
_C.WORD_LVL.REMOVE_ELONGATED = True
_C.WORD_LVL.REMOVE_STOPWORDS = False
_C.WORD_LVL.REMOVE_PUNCTUATIONS = False
_C.WORD_LVL.REMOVE_RETWEETS = False
_C.WORD_LVL.NORMALIZE_LAUGHTER = False
_C.WORD_LVL.REMOVE_SINGLE_CHARACTERS = False
_C.WORD_LVL.REMOVE_ANGLE_BRACKETS = False
_C.WORD_LVL.LEMMATIZE = False
_C.WORD_LVL.NORMALIZE_REPEATING_CHARACTERS = False
_C.WORD_LVL.REMOVE_NUMERIC = False
_C.WORD_LVL.REMOVE_EXTRA_SPACES = False
_C.WORD_LVL.REMOVE_TRAILING_URL = False
_C.WORD_LVL.MANUAL_SPELL_CORRECTION = False
_C.WORD_LVL.CLARIFY_EMPTY = False


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
