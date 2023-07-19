import logging
from loguru import logger
import torch
import time
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import argparse
import sys
import os
from sklearn.utils import shuffle
from datasets import DatasetDict, Dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd




sys.path.append('')

def numpy_softmax(model_preds):
    '''Converts the raw predictions from a HuggingFace model into clean logits.'''
    max = np.max(model_preds, axis=1, keepdims=True)
    e_x = np.exp(model_preds-max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    out = e_x / sum 
    return out


def load_tweets(file_path):    
    tweets = list()
    with open(file_path, 'r', encoding='utf-8') as preprocessed_tweets:
        for tweet in preprocessed_tweets :
            tweets.append(tweet.rstrip('\n'))          
    return tweets

def preprocess_function(examples):
    return tokenizer(examples["tweet"], truncation=True, max_length=args.tok_max_length, padding=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    #F1 score
    f1 = f1_score(labels, preds, average="weighted")
    #Accuracy 
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    dir = '/mnt/scratch/gtsolakis/cil_sentiment/CIL-Sentiment-Analysis-Project/src/8 - bert_torch/dataset_directory'
    out = '/mnt/scratch/gtsolakis/cil_sentiment/CIL-Sentiment-Analysis-Project/src/8 - bert_torch/logging'

    parser.add_argument('-m', '--model_name', type=str, help='bert model name', required=True)
    parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('-s', '--seed', type=int, help='random seed', default=12222)
    parser.add_argument("--fp16", default=True,  help='fp16  training')
 
  

    parser.add_argument('-o', '--out', type=str, help='output directory', default=out)
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs', default=3)
    parser.add_argument('-l', '--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('-w', '--wd', type=float, help='weight decay', default=0.005)
    parser.add_argument('--tag', default='', help='track experiments')
    parser.add_argument('--load_m', default=None, help='in case of wanting to load model')
    parser.add_argument("--test", default=False, help='load&evaluate')
    parser.add_argument('--tok_max_length', type=int, default=128, help='define tokenizer maximum length')
    parser.add_argument('--train_val_ratio', type=float, default=0.99, help='The training/validation ratio to use for the given dataset')

    
    
    args = parser.parse_args()

    torch.cuda.empty_cache()    
    time_run = time.time()


    project_path = "./"
    experiment_path = "./" + "Experiments/"



    experiment_date_for_folder_name = "experiment-" + args.model_name + "_" + args.tag

    experiments_results_path = experiment_path + experiment_date_for_folder_name
    os.makedirs(experiments_results_path, exist_ok=True)    # create the experiment folder(s) needed
    checkpoints_path = experiments_results_path + "/checkpoints/"
    print("The project path is: ", project_path)
    print("The experiment path is: ", experiment_path)
    print("The model checkpoints will be saved at: ", checkpoints_path, "\n")

    # for the submission
    test_results_path = experiments_results_path + "/test_results/"
    os.makedirs(test_results_path, exist_ok=True)   


    #Specify transformer model
    model_name = args.model_name


    #Select GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device {device}')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/bernice", model_max_length=128)






    #Load positive and negative preprocessed tweets
    train_pos_tweets = load_tweets('/mnt/scratch/gtsolakis/cil_sentiment/processed_pos_tweets_08.txt')
    train_neg_tweets = load_tweets('/mnt/scratch/gtsolakis/cil_sentiment/processed_neg_tweets_08.txt')
 

    test_tweets = load_tweets('/mnt/scratch/gtsolakis/cil_sentiment/processed_test_tweets_08.txt')



    #Create labels
    train_neg_labels = [0] * len(train_neg_tweets)
    train_pos_labels = [1] * len(train_pos_tweets)

    train_tweets = train_pos_tweets + train_neg_tweets
    train_labels = train_pos_labels + train_neg_labels 
    #Shuffle
    train_tweets, train_labels = shuffle(train_tweets, train_labels, random_state=10)
    data = pd.DataFrame({'tweet': train_tweets, 'label': train_labels})

    X = list(data["tweet"])
    y = list(data["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05,stratify=y,random_state=args.seed)


    # Combine X_train and y_train into a single dictionary
    train_data = {"tweet": X_train, "label": y_train}
    # Convert the dictionary to a Dataset object
    train_dataset = Dataset.from_dict(train_data)

    # Combine X_val and y_val into a single dictionary
    val_data = {"tweet": X_val, "label": y_val}
    # Convert the dictionary to a Dataset object
    val_dataset = Dataset.from_dict(val_data)

    # Combine the train and validation datasets into a DatasetDict
    data_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})



    #Tokenization using map
    tokenized_dataset = data_dict.map(preprocess_function, batched=True)
    #DataCollator for efficient batches
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,ignore_mismatched_sizes=True).to(device)



    logging_steps = 4000
    print(logging_steps)
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        seed = args.seed,
        weight_decay=args.wd,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=4,
        disable_tqdm=False,
        fp16=args.fp16,
        logging_steps=logging_steps,
        logging_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        load_best_model_at_end=True,
        warmup_steps=500
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


    logger.info('Started training')
    trainer.train()
    logger.info('Ended training')
    print(time.time())

    data_test = pd.DataFrame({'tweet': test_tweets})
    test_dataset = Dataset.from_dict(data_test)
    test_dataset = test_dataset.map(preprocess_function, batched=True)




    results = trainer.predict(test_dataset)

    y_preds = np.argmax(results.predictions, axis=1)

    y_preds = [-1 if val == 0 else 1 for val in y_preds]
    
    df = pd.DataFrame(y_preds, columns=["Prediction"])
    df.index.name = "Id"
    df.index += 1
    df.to_csv(test_results_path+"test_data.csv")

    logits = numpy_softmax(results.predictions)    


    os.makedirs(test_results_path + args.model_name + "-" + 'logits.txt', exist_ok=True)
    np.savetxt(test_results_path  + 'logits.txt', logits, delimiter=",", header = "negative,positive", comments = "") 


    exit(0)