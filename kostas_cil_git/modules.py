def load_model_from_checkpoint(path_to_checkpoint):
    ''' Helper function, to load the model from a checkpoint.
    takes as input a path to the checkpoint (from the "experiment-[...]" )
     '''
    full_path_to_model_checkpoint = experiment_path + path_to_checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(full_path_to_model_checkpoint, num_labels=2, local_files_only=False, ignore_mismatched_sizes=True)
    print(f"Loaded model from: {full_path_to_model_checkpoint}")
    return model

def numpy_softmax(model_preds):
    '''Converts the raw predictions from a HuggingFace model into clean logits.'''
    max = np.max(model_preds, axis=1, keepdims=True)
    e_x = np.exp(model_preds-max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    out = e_x / sum
    return out

def load_tweets(file_path):
    '''Reads a file containing preprocessed tweets and returns them as 
    a list of strings. It opens the file, reads each line, and appends the 
    stripped line (without the newline character) to the list of tweets.
    '''
    tweets = list()
    with open(file_path, 'r', encoding='utf-8') as preprocessed_tweets:
        for tweet in preprocessed_tweets:
            tweets.append(tweet.rstrip('\n'))
    return tweets

def preprocess_function(examples, tok_max_length, tokenizer):
    '''Preprocesses examples using a tokenizer. It takes a dictionary of examples with a "tweet" 
    key and applies tokenization with truncation, maximum length, and padding using the provided 
    tokenizer. It returns the preprocessed examples.
    '''
    return tokenizer(examples["tweet"], truncation=True, max_length=tok_max_length, padding=True)


def save_dictionary_as_pickle(dictionary, filename):
    ''' Saves a dictionary as a pickle file. 
    It takes a dictionary and a filename as input, 
    opens the file in write-binary mode, and uses 
    the pickle.dump() function to save the dictionary.
    '''
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def load_pickle_as_dictionary(filename):
    '''loads a pickle file as a dictionary. 
    It takes a filename as input, opens the 
    file in read-binary mode, and uses the 
    pickle.load() function to load the dictionary 
    from the file.
    '''
    with open(filename, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary


def compute_metrics(pred):
    '''computes evaluation metrics (accuracy and F1 score) 
    based on predicted labels and true labels.
    '''
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def get_mispredicted_samples(X, y_pred, y_true):
    '''retrieves mispredicted samples from given inputs. 
    It takes input samples X, predicted labels y_pred, 
    and true labels y_true
    '''
    mispredicted_X = []
    mispredicted_Y = []

    for i in range(len(X)):
        if y_pred[i] != y_true[i]:
            mispredicted_X.append(X[i])
            mispredicted_Y.append(y_true[i])

    return mispredicted_X, mispredicted_Y

def sort_by_difficulty(dataset, dict_path):
    '''sorts a dataset based on a difficulty score stored in a pickle file. 
    It takes a dataset dictionary, a path to a scores dictionary stored as 
    a pickle file, and performs the following steps: it creates a new dictionary 
    with indices as keys and includes the original tweet, label, and score. 
    Then, it sorts the new dictionary based on the scores in ascending order. 
    Finally, it constructs a new dataset from the sorted dictionary, preprocesses 
    it, and returns the preprocessed tokenized dataset.
    '''
    subset_X = dataset['tweet']
    subset_y = dataset['label']
    scores_dict = load_pickle_as_dictionary(dict_path)

    new_dict = {}
    for i, x in enumerate(subset_X):
      new_dict[i] = {'x': x, 'label': subset_y[i], 'score': scores_dict[x]['score']}

    new_sorted_dict = dict(sorted(new_dict.items(), key=lambda item: item[1]['score'], reverse=False))
    sorted_X = [item[1]['x'] for item in new_sorted_dict.items()]
    sorted_y = [item[1]['label'] for item in new_sorted_dict.items()]

    new_data = {"tweet": sorted_X, "label": sorted_y}
    new_dataset = Dataset.from_dict(new_data)
    new_tokenized_dataset = new_dataset.map(lambda examples: preprocess_function(examples, tok_max_length), batched=True)

    return new_tokenized_dataset

def interleave(train_dataset_sorted):
    ''' interleaves positive and negative samples in a sorted train dataset. 
    It takes the sorted train dataset and performs the following steps: it separates 
    the positive and negative samples based on their labels, calculates the true ratio, 
    and interleaves the samples accordingly to maintain the true ratio. Finally, it 
    constructs a new dataset from the interleaved samples, preprocesses it, and returns 
    the preprocessed tokenized dataset.
    '''
    X_pos_train, y_pos_train = list(zip(*filter(lambda t: t[1] == 1, zip(train_dataset_sorted['tweet'], train_dataset_sorted['label']))))
    X_neg_train, y_neg_train = list(zip(*filter(lambda t: t[1] == 0, zip(train_dataset_sorted['tweet'], train_dataset_sorted['label']))))

    a, b = [X_pos_train, y_pos_train], [X_neg_train, y_neg_train]
    n = len(a[0]) + len(b[0])
    results = []
    for j in range(2):
      true_ratio = len(a[j]) / n
      c = []
      a_count, b_count = 0, 0
      running_ratio = 0
      for i in range(n):
          if running_ratio < true_ratio:
              c.append(a[j][a_count])
              a_count += 1
          else:
              c.append(b[j][b_count])
              b_count += 1
          running_ratio = a_count / (a_count + b_count)
      results.append(c)

    new_data = {"tweet": results[0], "label": results[1]}
    new_dataset = Dataset.from_dict(new_data)
    new_tokenized_dataset = new_dataset.map(lambda examples: preprocess_function(examples, tok_max_length), batched=True)

    return new_tokenized_dataset