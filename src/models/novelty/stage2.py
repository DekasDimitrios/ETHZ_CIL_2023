'''
    File name: stage2.py
    Author: Andreas Psaroudakis / Kostas Fanaras
    Code Cleaning & Integration: Dekas Dimitrios
    Date last modified: 07/15/2023
    Python Version: 3.9
'''


import torch
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
from datasets import Dataset
from datetime import datetime

from src.models.io_handler import load_testing_tweets_to_df, load_from_pkl, save_test_preds, save_logits
from src.models.utils import concat_datasets, sort_by_difficulty, keep_distribution, compute_metrics, tokenization, numpy_softmax


def train_stage2_split(cfg, split_number):
    """
    A function that given a config is able to train the split number run of the first stage
    of our novel approach and produce the respective results and logs.

    :param cfg: a yacs CfgNode object with the appropriate parameters to be used
    :param split_number: an integer denoting the split number to be trained
    :return: a string representing the path that the logits of the model are stored in
    """

    # Select GPU device if available, reset cache, set seeds
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    np.random.seed(cfg.SYSTEM.SEED_VALUE)
    torch.manual_seed(cfg.SYSTEM.SEED_VALUE)

    # Create filing system structure
    experiment_date_for_folder_name = "experiment-" + cfg.PARAMETERS.MODEL_NAME + '_' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    experiments_results_path = cfg.IO.EXPERIMENT_PATH + experiment_date_for_folder_name
    checkpoints_path = experiments_results_path + "/checkpoints/"
    val_results_path = experiments_results_path + "/val_results/"
    test_results_path = experiments_results_path + "/test_results/"
    os.makedirs(experiments_results_path, exist_ok=True)
    os.makedirs(val_results_path, exist_ok=True)
    os.makedirs(test_results_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.PARAMETERS.MODEL_NAME, use_fast=True)

    train_val_folds = load_from_pkl(cfg.IO.TRAIN_VAL_FOLDS_PATH)

    data_test = load_testing_tweets_to_df(cfg.IO.PP_TEST_TWEET_FILE_PATH)
    test_dataset = Dataset.from_dict(data_test)
    test_dataset = test_dataset.map(lambda examples: tokenization(tokenizer, examples, cfg.PARAMETERS.TOK_MAX_LENGTH), batched=True)

    ds_options = [1, 2, 3]
    ds_options.remove(split_number)
    val_false_datasets = []

    for opt in ds_options:
        val_false_datasets.append(load_from_pkl(f'val_false_tokenized_dataset_stage1_split_{opt}.pkl'))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.PARAMETERS.MODEL_NAME,
                                                               num_labels=cfg.PARAMETERS.NUM_OF_LABELS,
                                                               ignore_mismatched_sizes=True).to(device)

    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=cfg.PARAMETERS.LR,
        per_device_train_batch_size=cfg.PARAMETERS.BATCH_SIZE,
        per_device_eval_batch_size=cfg.PARAMETERS.BATCH_SIZE,
        num_train_epochs=cfg.PARAMETERS.EPOCHS,
        save_total_limit=cfg.PARAMETERS.SAVE_TOTAL_LIMIT,
        seed=cfg.SYSTEM.SEED_VALUE,
        weight_decay=cfg.PARAMETERS.WD,
        evaluation_strategy=cfg.PARAMETERS.EVAL_STRATEGY,
        gradient_accumulation_steps=cfg.PARAMETERS.GRAD_ACCUM_STEPS,
        disable_tqdm=cfg.PARAMETERS.DISABLE_TQDM,
        fp16=cfg.SYSTEM.FP16,
        logging_steps=cfg.PARAMETERS.LOGGING_STEPS,
        logging_strategy=cfg.PARAMETERS.LOGGING_STRATEGY,
        save_strategy=cfg.PARAMETERS.SAVE_STRATEGY,
        logging_dir=cfg.IO.LOG_PATH,
        load_best_model_at_end=cfg.PARAMETERS.LOAD_BEST_MODEL_AT_END,
        warmup_steps=cfg.PARAMETERS.WARMUP_STEPS
    )

    train_dataset = concat_datasets(train_val_folds[2][0], val_false_datasets[0])
    train_dataset = concat_datasets(train_dataset, val_false_datasets[1])
    train_dataset_sorted = sort_by_difficulty(train_dataset, cfg.IO.SCORES_DICTIONARY_PATH, tokenizer, cfg.PARAMETERS.TOK_MAX_LENGTH)
    train_dataset_same_dist = keep_distribution(train_dataset_sorted, tokenizer, cfg.PARAMETERS.TOK_MAX_LENGTH)

    evaluation_dataset_sorted = sort_by_difficulty(train_val_folds[2][1], cfg.IO.SCORES_DICTIONARY_PATH, tokenizer, cfg.PARAMETERS.TOK_MAX_LENGTH)
    evaluation_dataset_same_dist = keep_distribution(evaluation_dataset_sorted, tokenizer, cfg.PARAMETERS.TOK_MAX_LENGTH)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_same_dist,
        eval_dataset=evaluation_dataset_same_dist,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results_test = trainer.predict(test_dataset)
    y_preds_test = np.argmax(results_test.predictions, axis=1)

    results_val = trainer.predict(train_val_folds[2][1])

    save_test_preds(y_preds_test, test_results_path + f"test_data_stage2_split_{split_number}.csv")
    save_logits(numpy_softmax(results_val.predictions), test_results_path, cfg.PARAMETERS.MODEL_NAME, f'logits_val_stage2_split_{split_number}.txt')
    save_logits(numpy_softmax(results_test.predictions), test_results_path, cfg.PARAMETERS.MODEL_NAME, f'logits_test_stage2_split_{split_number}.txt')
    return test_results_path + f'logits_test_stage2_split_{split_number}.txt'
