import numpy as np
import pandas as pd
from collections import Counter


def soft_voting(file_paths):
    # Initialize empty lists to store positive and negative logits
    positive_logits = []
    negative_logits = []

    # Read each file and extract positive and negative logits
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            # Skip the first line
            next(file)
            # Extract positive and negative logits from each line and append to the lists
            logits = [line.split(',') for line in file]
            positive_logits.append([float(line[1]) for line in logits])
            negative_logits.append([float(line[0]) for line in logits])

    # Convert the lists to numpy arrays
    positive_logits_array = np.array(positive_logits).T
    negative_logits_array = np.array(negative_logits).T

    average_positive_logits = np.mean(positive_logits_array, axis=1)
    average_negative_logits = np.mean(negative_logits_array, axis=1)

    average_logits = np.column_stack((average_negative_logits, average_positive_logits))

    np.savetxt("average_logits.txt", average_logits, delimiter=",", header="negative,positive", comments="")

    predictions = np.argmax(average_logits, axis=1)
    predictions = [-1 if test == 0 else 1 for test in predictions]

    df = pd.DataFrame(predictions, columns=["Prediction"])
    df.index.name = "Id"
    df.index += 1
    df.to_csv("soft_voting_predictions.csv")


def hard_voting(file_paths):
    # Read and combine predictions from each file
    predictions = []
    for file in file_paths:
        df = pd.read_csv(file, index_col="Id")
        predictions.append(df["Prediction"].values)

    # Perform majority voting
    majority_voted_predictions = []
    for i in range(len(predictions[0])):
        votes = [prediction[i] for prediction in predictions]
        majority_vote = Counter(votes).most_common(1)[0][0]
        majority_voted_predictions.append(majority_vote)

    # Create a DataFrame with the majority voted predictions
    df_majority_voted = pd.DataFrame(majority_voted_predictions, columns=["Prediction"])
    df_majority_voted.index.name = "Id"
    df_majority_voted.index += 1

    # Save the majority voted predictions to a CSV file
    df_majority_voted.to_csv("majority_voting_predictions.csv")
