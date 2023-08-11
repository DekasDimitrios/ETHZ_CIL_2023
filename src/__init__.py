'''
    File name: __init__.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''

import preprocess
import models

if __name__ == '__main__':
    # Preprocess the dataset
    preprocess.run('../configs/preprocess/best_preprocess_cfg.yaml')

    # Train the desired model
    # models.run('../configs/training/baseline_cfg.yaml', flag='Baseline')
    models.run('../configs/training/novelty_cfg.yaml', flag='Novelty')
