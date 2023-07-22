'''
    File name: config.py
    Author: Fanaras Konstantinos
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


# Create Input/Output Variables config
_C.IO = CN()
_C.IO.PROJECT_PATH = './'
_C.IO.EXPERIMENT_PATH = './Experiments/'
_C.IO.DICTIONARY_PATH = './scores_dict.pkl'
_C.IO.LOG_PATH = './logging'

# Create Preprocessing Steps Variables config
_C.HP = CN()
_C.HP.MODEL_NAME = "bert-base-uncased"
_C.HP.BATCH_SIZE = 32
_C.HP.EPOCHS = 3
_C.HP.LR = 1e-4
_C.HP.WD = 0.005
_C.HP.TOK_MAX_LENGTH = 128


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
