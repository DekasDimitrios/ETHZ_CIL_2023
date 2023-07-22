import logging
import torch
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import sys
import os
from sklearn.utils import shuffle
from datasets import DatasetDict, Dataset
from sklearn.metrics import accuracy_score, f1_score
from loguru import logger
import pickle

from modules import *
from config import update_cfg
import argparse

sys.path.append('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../../configs/training/stage1.yaml',
                        help='path to the configuration yaml file')

    args = parser.parse_args()
    cfg = update_cfg(args.cfg)
    cfg.freeze()

    # Set default values for the variables
    model_name = cfg.HP.MODEL_NAME
    batch_size = cfg.HP.BATCH_SIZE
    seed = cfg.SYSTEM.SEED_VALUE
    fp16 = cfg.SYSTEM.FP16
    out = cfg.IO.LOG_PATH
    epochs = cfg.HP.EPOCHS
    lr = cfg.HP.LR
    wd = cfg.HP.WD
    tok_max_length = cfg.HP.TOK_MAX_LENGTH

    torch.cuda.empty_cache()
    time_run = time.time()

    project_path = cfg.IO.PROJECT_PATH
    experiment_path = cfg.IO.EXPERIMENT_PATH

    experiment_date_for_folder_name = "experiment-" + model_name + "_" + "default"

    experiments_results_path = experiment_path + experiment_date_for_folder_name
    os.makedirs(experiments_results_path, exist_ok=True)
    checkpoints_path = experiments_results_path + "/checkpoints/"
    print("The project path is: ", project_path)
    print("The experiment path is: ", experiment_path)
    print("The model checkpoints will be saved at: ", checkpoints_path, "\n")

    # for the submission
    test_results_path = experiments_results_path + "/test_results/"
    os.makedirs(test_results_path, exist_ok=True)

    # for validation results
    val_results_path = experiments_results_path + "/val_results/"
    os.makedirs(val_results_path, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    test_tweets = load_tweets('/content/test_data.txt')

    with open('/content/train_val_folds.pkl', 'rb') as file:
        train_val_folds = pickle.load(file)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True).to(device)

    logging_steps = 4000
    training_args = TrainingArguments(
        output_dir=out,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        save_total_limit=2,
        seed=seed,
        weight_decay=wd,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=4,
        disable_tqdm=False,
        fp16=fp16,
        logging_steps=logging_steps,
        logging_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        load_best_model_at_end=True,
        warmup_steps=500
    )

    dictionary_path = cfg.IO.DICTIONARY_PATH
    train_dataset_sorted = sort_by_difficulty(train_val_folds[0][0], dictionary_path)
    train_interleave_dataset = interleave(train_dataset_sorted)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_interleave_dataset,
        eval_dataset=train_val_folds[0][1],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Started training")
    trainer.train()
    logger.info(f"Ended training")

    data_test = pd.DataFrame({'tweet': test_tweets})
    test_dataset = Dataset.from_dict(data_test)
    test_dataset = test_dataset.map(lambda examples: preprocess_function(examples, tok_max_length, tokenizer), batched=True)

    results_test = trainer.predict(test_dataset)
    y_preds_test = np.argmax(results_test.predictions, axis=1)

    results_val = trainer.predict(train_val_folds[0][1])
    y_preds_val = np.argmax(results_val.predictions, axis=1)

    y_preds_test = [-1 if test == 0 else 1 for test in y_preds_test]

    X_val_false, y_val_false = get_mispredicted_samples(train_val_folds[0][1]['tweet'],y_preds_val,train_val_folds[0][1]['label'])

    val_data_false = {"tweet": X_val_false, "label": y_val_false}
    # Convert the dictionary to a Dataset object
    val_false_dataset = Dataset.from_dict(val_data_false)

    #Tokenization using map
    val_false_tokenized_dataset = val_false_dataset.map(lambda examples: preprocess_function(examples, tok_max_length, tokenizer), batched=True)

    # Save val_false locally
    with open('/content/val_false_tokenized_dataset1.pkl', 'wb') as file:
        pickle.dump(val_false_tokenized_dataset, file)

    # Save val_false on Google Drive
    with open('/content/drive/MyDrive/val_false_tokenized_dataset1.pkl', 'wb') as file:
        pickle.dump(val_false_tokenized_dataset, file)

    df = pd.DataFrame(y_preds_test, columns=["Prediction"])
    df.index.name = "Id"
    df.index += 1
    df.to_csv(test_results_path + f"test_data1.csv")

    logits_val = numpy_softmax(results_val.predictions)
    logits_test = numpy_softmax(results_test.predictions)

    os.makedirs(test_results_path + model_name + "-" + 'logits_test1.txt', exist_ok=True)
    np.savetxt(test_results_path + f"logits_test1.txt", logits_test, delimiter=",", header="negative,positive", comments="")

    os.makedirs(val_results_path + model_name + "-" + 'logits_val1.txt', exist_ok=True)
    np.savetxt(val_results_path + f"logits_val1.txt", logits_val, delimiter=",", header="negative,positive", comments="")

    # Save val_false on Google Drive
    with open('/content/drive/MyDrive/logits_val1.txt', 'wb') as file:
        pickle.dump(val_results_path + f"logits_val1.txt", file)

    with open('/content/drive/MyDrive/logits_test1.txt', 'wb') as file:
        pickle.dump(test_results_path + f"logits_test1.txt", file)

    time_total = time.time() - time_run
    print(f"The program took {str(time_total/60/60)[:6]} Hours or {str(time_total/60)[:6]} minutes to run.")
