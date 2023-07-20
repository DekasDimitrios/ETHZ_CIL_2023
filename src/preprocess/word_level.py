"""
    File name: word_level.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
"""


import re
import string
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

tqdm.pandas()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
punctuations = string.punctuation
lemmatizer = WordNetLemmatizer()


# Hard coded spelling corrections derived by inspecting our data
spell_dict = {"i'm": "i am",
              "u": "you",
              "don't": "do not",
              "it's": "it is",
              "its": "it is",
              "im": "i am",
              "i'll": "i will",
              "that's": "that is",
              "you're": "you are",
              "dont": "do not",
              "i've": "i have",
              "didn't": "did not",
              "ur": "you are",
              "ill": "i will",
              "won't": "will not",
              "he's": "he is",
              "thats": "that is",
              "TRUE": "true",
              "she's": "she is",
              "haven't": "have not",
              "doesn't": "does not",
              "what's": "what is",
              "i'd": "i would",
              "you'll": "you will",
              "there's": "there is",
              "we're": "we are",
              "isn't": "is not",
              "let's": "let us",
              "they're": "they are",
              "wasn't": "was not",
              "didnt": "did not",
              "couldn't": "could not",
              "you've": "you have",
              "we'll": "we will"}


def to_lower_case(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to only contain sentences with lowercase characters.

    :param df: a dataframe that the lower-casing procedure is going to be applied to
    """
    df['to_lower'] = df['processed_tweet'].progress_apply(lambda x: x.lower())
    df['processed_tweet'] = df['to_lower'].copy()


def remove_user(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove the <user> tokens from its sentences.

    :param df: a dataframe that the removal of the <user> token is going to be applied to
    """
    df['remove_user'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([w for w in x.split() if w != '<user>']))
    df['processed_tweet'] = df['remove_user'].copy()


def remove_url(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove the <url> tokens from its sentences.

    :param df: a dataframe that the removal of the <url> token is going to be applied to

    """
    df['remove_url'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([w for w in x.split() if w != '<url>']))
    df['processed_tweet'] = df['remove_url'].copy()


def change_user(df, user_rep):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to change the <user> tokens from its sentences to an alternative
    representation given as a function argument.

    :param df: a dataframe that the change of the <user> token is going to be applied to
    :param user_rep: a string representing the alternative representation used to transform the <user> token.
    """
    df['change_user'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([user_rep if w == '<user>' else w for w in x.split()]))
    df['processed_tweet'] = df['change_user'].copy()


def change_url(df, url_rep):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to change the <url> tokens from its sentences to an alternative
    representation given as a function argument.

    :param df: a dataframe that the change of the <url> token is going to be applied to
    :param user_rep: a string representing the alternative representation used to transform the <url> token.
    """
    df['change_url'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([url_rep if w == '<url>' else w for w in x.split()]))
    df['processed_tweet'] = df['change_url'].copy()


def remove_elongated(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove the <elongated> tokens from its sentences.

    :param df: a dataframe that the removal of the <url> token is going to be applied to
    """
    df['remove_elongated'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([w if w != '<elongated>' else ' ' for w in x.split()]))
    df['processed_tweet'] = df['remove_elongated'].copy()


def remove_stopwords(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove the stopword tokens from its sentences.

    :param df: a dataframe that the removal of stopwords token is going to be applied to
    """
    df['remove_stopwords'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in stop_words]))
    df['processed_tweet'] = df['remove_stopwords'].copy()


def remove_punctuations(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove the punctuation tokens from its sentences.

    :param df: a dataframe that the removal of stopwords token is going to be applied to
    """
    df['remove_punctuations'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([w for w in x.split() if w.lower() not in punctuations]))
    df['processed_tweet'] = df['remove_punctuations'].copy()


def remove_retweets(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove the retweet tags from its sentences.

    :param df: a dataframe that the removal of stopwords token is going to be applied to
    """
    df['remove_retweets'] = df['processed_tweet'].progress_apply(lambda x: re.sub(r'\brt\b', '', x))
    df['processed_tweet'] = df['remove_retweets'].copy()


def normalize_laughter(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to change the laughing patterns from its sentences to an alternative representation.

    :param df: a dataframe that the change of the laughing patterns is going to be applied to
    """
    laughter_pattern = re.compile(r'.*(h+a+h+a+|a+h+a+h+).*', re.IGNORECASE)
    df['normalize_laughter'] = df['processed_tweet'].progress_apply(lambda x: ' '.join(['haha' if laughter_pattern.match(w) else w for w in x.split()]))
    df['processed_tweet'] = df['normalize_laughter'].copy()


def remove_single_characters(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to  remove single character words from its sentences.

    :param df: a dataframe that the change of the laughing patterns is going to be applied to
    """
    df['remove_single_characters'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
    df['processed_tweet'] = df['remove_single_characters'].copy()


def remove_angle_brackets(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove angle brackets surrounding words in its sentences.

    :param df: a dataframe that the change of the laughing patterns is going to be applied to
    """
    df['remove_angle_brackets'] = df['processed_tweet'].progress_apply(lambda x: re.sub(r'<([^<>]+)>', r'\1', x))
    df['processed_tweet'] = df['remove_angle_brackets'].copy()


def lemmatize(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to lemmatize words in its sentences.

    :param df: a dataframe that the change of the laughing patterns is going to be applied to
    """
    df['lemmatize'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in x.split()]))
    df['processed_tweet'] = df['lemmatize'].copy()


def remove_numeric(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove words that include numeric values in its sentences.

    :param df: a dataframe that the change of the laughing patterns is going to be applied to
    """
    digit_pattern = r'\d'
    df['remove_numeric'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([w for w in x.split() if not re.search(digit_pattern, w)]))
    df['processed_tweet'] = df['remove_numeric'].copy()


def remove_extra_spaces(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove extra spaces found between words in its sentences.

    :param df: a dataframe that the change of the laughing patterns is going to be applied to
    """
    df['remove_extra_spaces'] = df['processed_tweet'].progress_apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    df['processed_tweet'] = df['remove_extra_spaces'].copy()


def remove_trailing_urls(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to remove the trailing '... url' instances in its sentences.

    :param df: a dataframe that the change of the laughing patterns is going to be applied to
    """
    pattern = ' ... <url>'
    df['remove_trailing_urls'] = df['processed_tweet'].progress_apply(lambda s: s[:-len(pattern)] if s.endswith(pattern) else s)
    df['processed_tweet'] = df['remove_trailing_urls'].copy()


def manual_spell_correction(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to manually correct spelling in its sentences.

    :param df: a dataframe that the change of the laughing patterns is going to be applied to
    """
    df['spell_correction'] = df['processed_tweet'].progress_apply(lambda x: ' '.join([spell_dict[w] if w in spell_dict else w for w in x.split()]))
    df['processed_tweet'] = df['spell_correction'].copy()


def clarify_empty(df):
    """
    Function that when applied to a dataframe transforms its 'processed_tweet' column
    to manually label its empty sentences.

    :param df: a dataframe that the change of the laughing patterns is going to be applied to
    """
    empty_string_placeholder = "$!@EMPTYSTRING@!$"
    df['clarify_empty'] = df['processed_tweet'].progress_apply(lambda x: x if x != '' and x != ' ' else empty_string_placeholder)
    df['processed_tweet'] = df['clarify_empty'].copy()

    mask = df['processed_tweet'] == empty_string_placeholder
    df.loc[mask, 'processed_tweet'] = df.loc[mask, 'tweet']


def apply_word_level(df, cfg):
    """
    Function that when applied to a dataframe transforms a 'tweet' column to
    produce a new 'processed_tweet' column where the transformations designated
    by the config are performed to produce its content.

    :param df: a dataframe that the transformations are going to be applied to
    :param cfg: a config that indicated the transformations to be applied to the df argument
    :return: the dataframe obtained by applying the desired transformations
    """
    # Create a column named "processed_tweet" if no previous preprocessing step did so.
    if 'processed_tweet' not in df.columns:
        df['processed_tweet'] = df['tweet'].copy()

    if cfg.LOWER_CASE:
        print(">>> Applying lower casing")
        time.sleep(2)  # Used for the prints and tqdm co-existence to not produce weird visual effects
        to_lower_case(df)
    if cfg.REMOVE_TRAILING_URL:
        print(">>> Removing trailing urls")
        time.sleep(2)
        remove_trailing_urls(df)
    if cfg.REMOVE_USER:
        print(">>> Removing user tags")
        time.sleep(2)
        remove_user(df)
    if cfg.CHANGE_USER:
        print(">>> Changing user tags")
        time.sleep(2)
        change_user(df, cfg.USER_REPRESENTATION)
    if cfg.REMOVE_URL:
        print(">>> Removing url tags")
        time.sleep(2)
        remove_url(df)
    if cfg.CHANGE_URL:
        print(">>> Changing url tags")
        time.sleep(2)
        change_url(df, cfg.URL_REPRESENTATION)
    if cfg.NORMALIZE_LAUGHTER:
        print(">>> Normalizing laughter")
        time.sleep(2)
        normalize_laughter(df)
    if cfg.REMOVE_ELONGATED:
        print(">>> Removing elongated tags")
        time.sleep(2)
        remove_elongated(df)
    if cfg.REMOVE_RETWEETS:
        print(">>> Removing retweet tags")
        time.sleep(2)
        remove_retweets(df)
    if cfg.REMOVE_SINGLE_CHARACTERS:
        print(">>> Removing single characters")
        time.sleep(2)
        remove_single_characters(df)
    if cfg.REMOVE_ANGLE_BRACKETS:
        print(">>> Removing Angle Brackets")
        time.sleep(2)
        remove_angle_brackets(df)
    if cfg.LEMMATIZE:
        print(">>> Lemmatizing")
        time.sleep(2)
        lemmatize(df)
    if cfg.REMOVE_NUMERIC:
        print(">>> Removing numeric words")
        time.sleep(2)
        remove_numeric(df)
    if cfg.REMOVE_EXTRA_SPACES:
        print(">>> Removing extra spaces")
        time.sleep(2)
        remove_extra_spaces(df)
    if cfg.MANUAL_SPELL_CORRECTION:
        print(">>> Manual spell correction")
        time.sleep(2)
        manual_spell_correction(df)
    if cfg.REMOVE_STOPWORDS:
        print(">>> Removing stopwords")
        time.sleep(2)
        remove_stopwords(df)
    if cfg.REMOVE_PUNCTUATIONS:
        print(">>> Removing punctuations")
        time.sleep(2)
        remove_punctuations(df)
    if cfg.CLARIFY_EMPTY:
        print('>>> Clarifying empty tweets')
        time.sleep(2)
        clarify_empty(df)
    return df
