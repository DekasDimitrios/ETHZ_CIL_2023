'''
    File name: utils.py
    Author: Dekas Dimitrios
    Date last modified: 07/15/2023
    Python Version: 3.9
'''

import pandas as pd
from sklearn.metrics import accuracy_score


def eval_test_model(y_val, val_preds, test_preds, test_out_path):
    accuracy = accuracy_score(y_val, val_preds)
    df = pd.DataFrame(test_preds, columns=["Prediction"])
    df.index.name = "Id"
    df.index += 1
    df.to_csv(test_out_path)
    return accuracy