'''
    File name: transformer.py
    Author: Andreas Psaroudakis / Kostas Fanaras
    Code Cleaning & Integration: Dekas Dimitrios
    Date last modified: 07/15/2023
    Python Version: 3.9
'''


import torch
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, \
    Trainer
import os
from datasets import DatasetDict, Dataset

from src.models.utils import numpy_softmax, compute_metrics, tokenization
from src.models.io_handler import load_training_tweets_to_df, load_testing_tweets_to_df, save_logits,  save_test_preds


def train_hugging_face_transformer(cfg):
    """
    A function that given a config is able to train a hugging face transformer and produce the respective results and logs.

    :param cfg: a yacs CfgNode object with the appropriate parameters to be used
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
    test_results_path = experiments_results_path + "/test_results/"
    os.makedirs(experiments_results_path, exist_ok=True)
    os.makedirs(test_results_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.PARAMETERS.MODEL_NAME, use_fast=True)

    # Load data into appropriate datasets and tokenize them
    data = load_training_tweets_to_df(cfg.IO.PP_POS_TWEET_FILE_PATH,
                                      cfg.IO.PP_NEG_TWEET_FILE_PATH,
                                      10)

    X = list(data["tweet"])
    y = list(data["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.PARAMETERS.TRAIN_VAL_SPLIT_RATIO, stratify=y, random_state=cfg.SYSTEM.SEED_VALUE)

    train_data = {"tweet": X_train, "label": y_train}
    train_dataset = Dataset.from_dict(train_data)

    val_data = {"tweet": X_val, "label": y_val}
    val_dataset = Dataset.from_dict(val_data)

    data_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})
    tokenized_dataset = data_dict.map(lambda examples: tokenization(tokenizer, examples, cfg.PARAMETERS.TOK_MAX_LENGTH), batched=True)

    data_test = load_testing_tweets_to_df(cfg.IO.PP_TEST_TWEET_FILE_PATH)
    test_dataset = Dataset.from_dict(data_test)
    test_dataset = test_dataset.map(lambda examples: tokenization(tokenizer, examples, cfg.PARAMETERS.TOK_MAX_LENGTH),
                                    batched=True)

    # DataCollator for efficient batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.PARAMETERS.MODEL_NAME, num_labels=cfg.PARAMETERS.NUM_OF_LABELS,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.predict(test_dataset)

    y_preds = np.argmax(results.predictions, axis=1)
    y_preds = [-1 if val == 0 else 1 for val in y_preds]
    save_test_preds(y_preds, path=test_results_path + "test_data.csv")
    save_logits(numpy_softmax(results.predictions), test_results_path, cfg.PARAMETERS.MODEL_NAME, 'logits.txt')
