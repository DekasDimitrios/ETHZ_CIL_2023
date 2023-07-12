'''
    File name: duplicates.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''


import pandas as pd
from nlp_dedup import Deduper
import jsonlines


def drop_duplicates(tweets_df, drop_factor):
    """
    Given a dataframe and a float factor the function is able to drop the duplicates that are present in the dataframe.
    In order to do so, the labeling of tweets that have been labeled with both positive and negative sentiment is
    determined as follows. Tweets that have the same number of negative and positive labeling instances are dropped
    when the drop factor is any value below 0.51. Then, and for higher values of drop_factor, any tweet that has both
    positive and negative labels we opt to keep the label that is the most frequent, only if the
    number of occurrences (noc) of the most frequent label satisfies the following equation:

                            noc / total number of occurrences of tweet >= drop_factor

    :param tweets_df: a dataframe containing the training tweets
    :param drop_factor: a float that represents the percentage needed for a label selection to be definite
    :return: a dataframe that contains the remaining tweets, after the dropping of duplicates is carried out
    """
    # Group the dataframe by tweet text
    grouped = tweets_df.groupby('tweet')

    # Count the number of positive and negative labels for each tweet
    counts_df = grouped['label'].value_counts().unstack(fill_value=0)
    pos_count = counts_df[1].rename('pos_count')
    neg_count = counts_df[-1].rename('neg_count')
    counts_df = pd.concat([pos_count, neg_count], axis=1)

    # Find the count of the most frequent label for each tweet and the label itself
    counts_df['most_frequent_count'] = counts_df[['pos_count', 'neg_count']].max(axis=1)
    counts_df['most_frequent_label'] = counts_df[['pos_count', 'neg_count']].idxmax(axis=1)

    # Calculate the fraction between the most frequent label and the other one
    counts_df['fraction'] = counts_df['most_frequent_count'] / (counts_df['pos_count'] + counts_df['neg_count'])

    # Determine if fraction is above a certain threshold factor
    counts_df['keep'] = counts_df['fraction'] >= drop_factor

    # Keep positive tweets
    pos_tweets = ((counts_df['keep'] == True) & (counts_df['most_frequent_label'] == 'pos_count'))
    pos_tweets = pos_tweets.iloc[pos_tweets.to_numpy()].index.to_numpy()

    # Keep negative tweets
    neg_tweets = ((counts_df['keep'] == True) & (counts_df['most_frequent_label'] == 'neg_count'))
    neg_tweets = neg_tweets.iloc[neg_tweets.to_numpy()].index.to_numpy()

    # Drop the ambiguous ones based on the factor given as parameter
    other_duplicates = (counts_df['keep'] == False)
    other_duplicates = other_duplicates.iloc[other_duplicates.to_numpy()].index.to_numpy()

    # Assign the appropriate label to each tweet in the dataset
    tweets_df.loc[tweets_df['tweet'].isin(pos_tweets), 'label'] = 1
    tweets_df.loc[tweets_df['tweet'].isin(neg_tweets), 'label'] = -1
    tweets_df.loc[tweets_df['tweet'].isin(other_duplicates), 'label'] = 0

    # Drop the tweets that are assigned with the zero label
    tweets_df = tweets_df.drop(tweets_df[tweets_df['label'] == 0].index)
    tweets_df.drop_duplicates(inplace=True)
    tweets_df = tweets_df.reset_index()

    return tweets_df


def drop_near_duplicates(tweets_df, seed):
    """
    Given a dataframe the function is able to drop the near duplicates that are present in the dataframe.
    In order to do so, the use of the nlp_dedup library, implementing a MinHash algorithm, is being made.

    :param tweets_df: a dataframe containing the training tweets
    :return: a dataframe that contains the remaining tweets, after the dropping of near duplicates is carried out
    """
    # Initialize the Deduper class instance
    deduper = Deduper(random_seed=seed)

    # Extract the corpus given by the tweets in the dataframe
    corpus = tweets_df['tweet']

    # Extract the deduplicated version of our corpus in a JSON file
    x = deduper.deduplicate(corpus=corpus, output_dir='../../data/deduplicated/', overwrite=True)

    # Create an empty list to hold the JSON objects
    data = []

    # Open the jsonl file and append each object to the list
    with jsonlines.open('../../data/deduplicated/deduplicated_corpus.jsonl') as reader:
        for obj in reader:
            data.append(obj)

    # Create a DataFrame from the list of JSON objects
    df_deduplicated = pd.DataFrame(data)
    df_deduplicated.rename(columns={'text': 'tweet'}, inplace=True)

    # Return the tweets that belong to the deduplicated instances
    return tweets_df.loc[df_deduplicated['id']]
