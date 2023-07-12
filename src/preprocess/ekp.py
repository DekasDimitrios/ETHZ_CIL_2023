'''
    File name: ekp.py
    Author: Dekas Dimitrios
    Date last modified: 07/08/2023
    Python Version: 3.9
'''


from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang import slangdict
from nltk.tokenize import WhitespaceTokenizer
from tqdm import tqdm


def apply_ekphrasis(df, cfg):
    """
    Given a dataframe containing a "tweet" column and a config that determines the
    ekphrasis hyperparameters the function is able to return the dataframe after
    applying the desired transformation producing a new column containing them.

    :param df: a dataframe containing the tweets that will be preprocessed using ekphrasis
    :param cfg: a config that contains the necessary parameters needed for ekphrasis to execute properly
    :return: a dataframe containing the tweets preprocessed using ekphrasis
    """

    tqdm.pandas()
    dicts = []
    if 'emoticons' in cfg.DICTIONARIES:
        dicts.append(emoticons)
    if 'slang_dict' in cfg.DICTIONARIES:
        dicts.append(slangdict)

    tokenizer = None
    if cfg.TOKENIZER == 'Whitespace':
        tokenizer = WhitespaceTokenizer().tokenize
    elif cfg.TOKENIZER == 'Social':
        tokenizer = SocialTokenizer(lowercase=True).tokenize
    else:
        print("The provided tokenizer value is not an expected one. Therefore, the Whitespace one is going to be used.")
        tokenizer = WhitespaceTokenizer().tokenize

    text_processor = TextPreProcessor(
        # terms that will be omitted | myaddress@mysite.com ->
        omit=cfg.OMIT,
        # terms that will be normalized | myaddress@mysite.com -> <email>
        normalize=cfg.NORMALIZE,
        # terms that will be annotated | myaddress@mysite.com -> myaddress@mysite.com <email>
        annotate=cfg.ANNOTATE,
        # how to wrap the capitalized words
        all_caps_tag=cfg.ALL_CAPS_TAG,

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter=cfg.SEGMENTOR,

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector=cfg.CORRECTOR,

        unpack_hashtags=cfg.UNPACK_HASHTAGS,  # perform word segmentation on hashtags | #ilikedogs -> i like dogs
        unpack_contractions=cfg.UNPACK_CONTRACTIONS,  # Unpack contractions | can't -> can not, wouldn't -> would not
        spell_correct_elong=cfg.SPELL_CORRECT_ELONGATED,  # spell correction for elongated words
        spell_correction=cfg.SPELL_CORRECTION,  # choose if you want to perform spell correction to the text

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=tokenizer,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionary.
        dicts=dicts,

        # Chooses if you want to fix bad unicode terms and html entities.
        fix_text=cfg.FIX_TEXT,
        fix_html=cfg.FIX_HTML,

        # Removes tags after processing
        remove_tags=cfg.REMOVE_TAGS
    )

    df['ekphrasis_results'] = df['tweet'].progress_apply(lambda x: ' '.join(text_processor.pre_process_doc(x)))
    df['processed_tweet'] = df['ekphrasis_results'].copy()
    return df


def apply_ekphrasis_extra(df):
    """
    Given a dataframe containing a "tweet" column the function is able to
    return the dataframe after applying the desired transformation.

    :param df: a dataframe containing the tweets that will be preprocessed using ekphrasis
    :return: a dataframe containing the tweets preprocessed using ekphrasis
    """

    tqdm.pandas()
    text_processor = TextPreProcessor(
        # terms that will be normalized | myaddress@mysite.com -> <email>
        normalize=['user', 'url'],

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        # corpus from which the word statistics are going to be used
        # for spell correction
        corrector="twitter",

        unpack_hashtags=False,  # perform word segmentation on hashtags | #ilikedogs -> i like dogs
        unpack_contractions=False,  # Unpack contractions | can't -> can not, wouldn't -> would not
        spell_correct_elong=False,  # spell correction for elongated words
        spell_correction=False,  # choose if you want to perform spell correction to the text

        # select a tokenizer. You can use SocialTokenizer, or pass your own
        # the tokenizer, should take as input a string and return a list of tokens
        tokenizer=WhitespaceTokenizer().tokenize,

        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionary.
        dicts=[],

        # Chooses if you want to fix bad unicode terms and html entities.
        fix_html=False,
    )

    df['tweet'] = df['tweet'].progress_apply(lambda x: ' '.join(text_processor.pre_process_doc(x)))
    return df