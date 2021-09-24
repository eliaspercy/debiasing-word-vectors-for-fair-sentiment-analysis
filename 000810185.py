"""
Pre-requisites

The codebase has been entirely written with Python 3.9. Therefore, please run
the program using this version to ensure all will work correctly.

Moreover, for the duration of the project I have only had access to a Windows
10 operating system, meaning that the program has been thoroughly tested on
a Windows 10 OS, but not any other. If possible, it would be best if this
program is run on Windows 10, although I have put effort into ensuring that
the program should theoretically work on other operating systems.


DATA
-------------------------------------------------------------------------------

BY DEFAULT, ALL DATA WILL BE IN THE RIGHT PLACE WITH THE SUBMISSION.


The program makes use of the following data(sets):
(note that opening any of the following links will instantiate a download)

1. Stanford Sentiment Treebank
LINK: nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip

2. Equity Evaluation Corpus
LINK: https://saifmohammad.com/WebDocs/EEC/Equity-Evaluation-Corpus.zip

3. Word2vec, pretrained on the Google News corpus
LINK: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

4. Opinion Lexicon (English)
LINK: www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar


Each of these datasets must be unzipped (with the exception of the word2vec,
which must remain zipped), and stored in a folder entitled 'data', which must
be in the same directoy as the Python file. By default, the submission will
contain all data files in the correct place, so none of the above should need
to be downloaded. The links are there merely for reference.



-------------------------------------------------------------------------------

EXTERNAL LIBRARIES
-------------------------------------------------------------------------------

The following external libraries have been used in the program, which must be
installed in order for the program to successfully run.

    wordcloud==1.8.1
    pandas==1.2.2
    numpy==1.20.1
    seaborn==0.11.1
    torch==1.8.1
    nltk==3.6.2
    matplotlib==3.3.4
    gensim==4.0.1
    scikit_learn==0.24.2

Note the inclusion of the versions, these are the versions that I used when
developing and testing the program, so for safety purposes please install the
modules with the versions specified.

Provided with the submission is a file 'requirements.txt', which can be used to
quickly install all of the above dependencies, i.e.:

    - pip install -r requirements.txt


Optionally, these can all be installed in the standard way, via pip. E.g.,

    - pip install wordcloud==1.8.1


Once these libraries have been installed, the program can be run in the usual
way. That is as follows:

    - python 000810185.py



Please note that the results in the visualisations may differ somewhat from
those that appear in the report. This is not a bug, but purely due to the
random nature of some parts (for instance, the splitting of the trian/test
sets, the operating system used, the words from the word2vec embeddings used,
and so on). Generally, however, they should remain mostly consistent.

-------------------------------------------------------------------------------


Note on run-time:

Typically, the program will take around 8 to 10 minutes to complete when the
initial bias analysis is set to false. When the initial bias analysis is
included then the program will take around 15 to 20 minutes.

"""
# ------- ADJUSTABLE PARAMETERS ------- #
INITIALISE_DATA: bool = False
INITIAL_ANALYSIS: bool = True
USE_APP: bool = False


# ------------------------------------- #

# Main Imports
import pandas as pd
import numpy as np
import time
import os
from typing import (
    Dict,
    Tuple,
    Any,
    Union,
    List,
    Set
)
# Main constants
LINE_LENGTH: int = 50
DOWNLOAD_ROOT: str = "http://nlp.stanford.edu/~socherr/" \
                     "stanfordSentimentTreebank.zip"
SOS_TXT: str = "SOStr.txt"
SENTENCES_TXT: str = "datasetSentences.txt"
SPLIT_TXT: str = "datasetSplit.txt"
DICTIONARY_TXT: str = "dictionary.txt"
SENT_LABELS_TXT: str = "sentiment_labels.txt"
STANFORD_FOLDER: str = "stanfordSentimentTreebank"
DATASETS_PATH: str = os.path.join("data")
STANFORD_PATH: str = os.path.join("data", STANFORD_FOLDER)
READ_IN_FORMAT: str = 'utf-8'

# Disable unrequired warnings
pd.options.mode.chained_assignment = None

# Set random seed
np.random.seed(13)


# --------------------------------------------------------------------------- #
# ------------------------- Part 1: Initialisation -------------------------- #

# ---- Dataset Initialisation (i.e., Data Collection)
from urllib.request import urlopen
from zipfile import ZipFile
from numpy import inf
from io import BytesIO


def download_files(url: str = DOWNLOAD_ROOT,
                   path: str = DATASETS_PATH
                   ) -> None:
    """
    Download the contatiner zip file and extract to current directory
    """
    with urlopen(url) as file:
        with ZipFile(BytesIO(file.read())) as file_zip:
            file_zip.extractall(path)


def get_sostr(path: str = STANFORD_PATH,
              sos_txt: str = SOS_TXT,
              form: str = READ_IN_FORMAT
              ) -> pd.DataFrame:
    """
    Read in the sentences from the SOStr.txt
    """
    sostr = pd.read_csv(f'{path}/{sos_txt}', sep='\t', index_col=False,
                        header=None, names=['sentences'], encoding=form)
    sostr["sentence_index"] = sostr.index + 1
    sostr["sentences"] = sostr["sentences"] \
        .apply(lambda s: ' '.join(str(s).split('|')))
    return sostr


def get_sentences_and_split(path: str = STANFORD_PATH,
                            sentences_txt: str = SENTENCES_TXT,
                            split_txt: str = SPLIT_TXT,
                            form: str = READ_IN_FORMAT
                            ) -> pd.DataFrame:
    """
    Read in and the sentences and merge with dataset split
    """
    sentences = get_sostr()
    split = pd.read_csv(f'{path}/{split_txt}', sep=',', encoding=form)
    sentence_split = pd.merge(sentences, split, on=['sentence_index'])
    return sentence_split


def get_sentiment_labels(path: str = STANFORD_PATH,
                         labels_txt: str = SENT_LABELS_TXT,
                         form: str = READ_IN_FORMAT
                         ) -> pd.DataFrame:
    """
    Read in the sentiment labels
    """
    labels = pd.read_csv(f'{path}/{labels_txt}', sep='|', encoding=form)
    return labels


def get_phrase_sentiment_dictionary(path: str = STANFORD_PATH,
                                    dictionary_txt: str = DICTIONARY_TXT,
                                    form: str = READ_IN_FORMAT
                                    ) -> Dict[int, str]:
    """
    Get the phrases and their corresponding ID from the dictionary.txt
    Store this in a Python dict
    """
    id2phrase = dict()
    with open(f"{path}/{dictionary_txt}", 'r', encoding=form) as d:
        for line in d.read().split('\n'):
            phrase_id = line.split('|')
            try:
                id2phrase[int(phrase_id[1])] = phrase_id[0]
            except IndexError:
                pass
    labels = get_sentiment_labels()
    labels["phrases"] = labels["phrase ids"].apply(lambda s: id2phrase[s])
    phrase_sentiment_dict = dict(zip(labels["phrases"],
                                     labels["sentiment values"]))
    return phrase_sentiment_dict


def write_csv(dataset: pd.DataFrame,
              ds_name: str,
              data_path: str = DATASETS_PATH
              ) -> None:
    """
    Drop the unneeded columns and save dataframe as csv
    """
    # Creeate the dataset directory if it doesn't exist
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    to_drop = ["sentence_index", "splitset_label"]
    dataset.drop(labels=to_drop, axis=1, inplace=True)
    dataset.to_csv(f'{data_path}/{ds_name}', index=False)


def initialise_datasets(path: str = STANFORD_PATH,
                        split: bool = False
                        ) -> None:
    """
    Initialise the data for the project. Note that if split is set to true,
    then the data will be split into a train/test/val set corresponding with
    the reccomended split in the article. However, for the sake of the
    coursework specification, I will set this to False by default.
    """
    # Download the required files, if necessary
    if not os.path.exists(path):
        download_files()

    # Retrieve the dataset and the phrase-to-sentiment value dictionary
    data = get_sentences_and_split()
    phrase2sentiment = get_phrase_sentiment_dictionary()

    # Create sentiment value column to the dataset
    data["sentiments"] = data["sentences"].apply(lambda s: phrase2sentiment[s])

    if split:
        # Split the dataset into the test, train, and development sets (and
        # write these to csv format)
        dataset_names = ("train_set.csv", "test_set.csv", "dev_set.csv")
        for i in range(3):
            write_csv(data[data["splitset_label"] == i + 1], dataset_names[i])
    else:
        # Write out the entire data into one file
        write_csv(data, "stanford_dataset.csv")


def read_in_datasets(ds_names: List[str],
                     path: str = DATASETS_PATH,
                     form: str = READ_IN_FORMAT,
                     ) -> tuple[Any, ...]:
    """
    Read in the the csv dataset(s) as pandas dataframes
    """
    return tuple(pd.read_csv(f'{path}/{name}', encoding=form)
                 for name in ds_names)


# --------------------------------------------------------------------------- #
# --------------------- Part 2: Data Analysis ------------------------------- #

# ----- Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

plt.style.use("seaborn-muted")
sns.set()
sns.set_palette("muted")
sns.set_style("whitegrid")


def get_statistics(data: pd.DataFrame
                   ) -> None:
    """
    Get dataset statistics: i.e., info, standard deviation, mean, etc.
    """
    print("Dataset head.")
    print(data.head())
    print()
    print("Dataset Info.")
    print(data.info())
    print()
    print("Sentiment statistics")
    print(data.describe())
    print()


def visualise_sentiment_distribution(data: pd.DataFrame
                                     ) -> None:
    """
    Illustrate the distribution of sentiment values in the data
    """

    sns.displot(x="sentiments", data=data)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Count')
    plt.show()


def visualise_num_words(data: pd.DataFrame
                        ) -> None:
    """
    Illustrate the number of words "sentence" via a histogram
    """
    words_per_sentence = data["sentences"].apply(
        lambda s: len(s.split(' ')) if not isinstance(s, list) else len(s)
    )
    words_per_sentence.hist(bins=100)
    print("Sentence length statistics.")
    print(words_per_sentence.describe())
    print()
    plt.xlabel('Number of Words in Sentence')
    plt.ylabel('Count')
    plt.show()


def display_word_cloud(data: pd.DataFrame
                       ) -> None:
    """
    Create word cloud to visualise the frequency of specific words or symbols
    """
    all_text = ''.join(data['sentences'])
    plt.figure(figsize=(10, 7))
    word_cloud = WordCloud(
        max_font_size=100,
        max_words=200,
        scale=10,
        width=500,
        height=400,
        # background_color="white"
    ).generate(all_text)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def data_visualisation(data: pd.DataFrame
                       ) -> None:
    """
    Container function for visualisation methods
    """
    get_statistics(data)
    visualise_sentiment_distribution(data)
    visualise_num_words(data)
    display_word_cloud(data)


# ----- Data Preprocessing (i.e., Data Cleaning/cleansing)
"""
Five generic preprocessing procedures of NLP ML programs:
    1. Conversion of strings to lower case
    2. Removal of invalid (or "special") symbols
    3. Removal of "stop words" (such as "almost", "always", "the"; i.e., high
       freq. words that provide little lexical content, we may want to remove.)
    4. "Stemming" or "lemmatisation" (i.e., talk, talking, talked -> talk)
    5. Tokenisation

(Influenced by NLP With Python, by Bird,Klein&Loper - chapters 2 to 4)
"""
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk import word_tokenize


def handle_sentiments(data: pd.DataFrame,
                      make_classification: bool = True,
                      make_binary: bool = False,
                      delete_neutral: bool = False
                      ) -> pd.DataFrame:
    """
    Function for handling the sentiment labels. This function allows for
    turning the problem into a classification one, by placing the sentiment
    values into bins. By default, a binary classifcation problem will be
    established (i.e., sentiments equal one or zero). But, by setting the
    'make_binary' parameter to False, we will have a five-class problem.
    """
    def map2binary(sentiment: int
                   ) -> Union[int, None]:
        if sentiment < 2:
            return 0
        if sentiment > 2:
            return 1
        else:
            return np.nan

    if make_classification:
        data["sentiments"] = pd.cut(data["sentiments"],
                                    bins=[-inf, 0.2, 0.4, 0.6, 0.8, inf],
                                    labels=[0, 1, 2, 3, 4],
                                    ordered=True).astype(int)
        if make_binary:
            data["sentiments"] = data["sentiments"].apply(map2binary)
        elif delete_neutral:
            data = data[data["sentiments"] != 2]

    elif delete_neutral:
        data = data[data["sentiments"].apply(lambda x: x < 0.4 or x > 0.6)]

    data.dropna(inplace=True)
    return data


def convert_lower_case(data: pd.DataFrame
                       ) -> pd.DataFrame:
    """
    Convert all characters in the data to lower case, which will reduce the
    number of unique words (thus increasing accuracy)
    """
    data["sentences"] = data["sentences"].apply(lambda s: s.lower())
    return data


def handle_special_symbols(data: pd.DataFrame,
                           keep_some: bool = False
                           ) -> pd.DataFrame:
    """
    Remove all "special" characters that *do not* provide any insight; for
    instance, punctuation such as exlamantion marks (!) may be used to express
    sentiment, so we may not want to remove them. Conversely, removing commas
    is unlikely to harm the accuracy.
    """
    special_symbols = set()

    # Helper function for collecting special symbols
    def collect_special_symbols(s: str
                                ) -> List[str]:
        [special_symbols.add(c) for c in list(s)
         if c not in special_symbols and not c.isalnum() and c != ' ']

    data["sentences"].apply(collect_special_symbols)

    # Remove special symbols from the set that may provide some insight
    if keep_some:
        for symbol in ('!', '?'):
            if symbol in special_symbols:
                special_symbols.remove(symbol)

    # Remove the remaining special symbols
    def map_char(c: str
                 ) -> str:
        return c if c not in special_symbols else ''

    def remove_symbols(s: str
                       ) -> str:
        return ''.join([map_char(c) for c in list(s)])

    data["sentences"] = data["sentences"].apply(remove_symbols)

    return data


def handle_stop_words(data: pd.DataFrame
                      ) -> pd.DataFrame:
    """
    Remove high frequency words that contain minimal semantic value
    """
    stopws = set(stopwords.words('english'))

    def sans_stopwords(s: str
                       ) -> str:
        return ' '.join([w for w in word_tokenize(s) if w not in stopws])

    data["sentences"] = data["sentences"].apply(sans_stopwords)
    return data


def stemming(data: pd.DataFrame
             ) -> pd.DataFrame:
    """
    Treat all "inflections" identically for the sake of the model
    """
    stemmer = LancasterStemmer()

    def apply_stemmer(s: str
                      ) -> str:
        return " ".join(stemmer.stem(w) for w in word_tokenize(s))

    data["sentences"] = data["sentences"].apply(apply_stemmer)
    return data


def lemmatisation(data: pd.DataFrame
                  ) -> pd.DataFrame:
    """
    Similar to stemming, except deriving semantic root instead of simple prefix
    """
    lemmatiser = WordNetLemmatizer()

    def apply_lemmatiser(s: str
                         ) -> str:
        return " ".join(lemmatiser.lemmatize(w) for w in word_tokenize(s))

    data["sentences"] = data["sentences"].apply(apply_lemmatiser)
    return data


def tokenisation(data: pd.DataFrame
                 ) -> pd.DataFrame:
    """
    Tokenise the sentences
    """
    data["sentences"] = data["sentences"].apply(word_tokenize)
    return data


def visualise_num_words_again(data: pd.DataFrame
                              ) -> None:
    """
    Compare the number of words in the sentence before and after the removal
    of outliers (i.e., overly long or short sentences)
    """
    words_per_sentence = data["sentences"].apply(len)
    words_per_sentence.hist(bins=100)
    print()
    print(words_per_sentence.describe())
    plt.xlabel('Number of Words in Phrase')
    plt.ylabel('Count')
    plt.show()


def remove_length_outliers(data: pd.DataFrame
                           ) -> pd.DataFrame:
    """
    Remove sentences whose lengths are disproportionately long
    """
    # visualise_num_words_again(data)
    data["sentence_length"] = data["sentences"].apply(len)
    lim = data["sentence_length"].quantile(0.99)
    data = data[data["sentence_length"] < lim]
    data.drop(labels=["sentence_length"], axis=1, inplace=True)
    # visualise_num_words_again(data)
    return data


def tidy_data(data: pd.DataFrame,
              lower_case: bool = False,
              special_symbols: bool = True,
              stop_words: bool = True,
              stem: bool = False,
              lemmatise: bool = True,
              tokenise: bool = True,
              length: bool = True
              ) -> pd.DataFrame:
    """
    Container method for data cleaning
    """
    if lower_case:
        data = convert_lower_case(data)
    if special_symbols:
        data = handle_special_symbols(data)
    if stop_words:
        data = handle_stop_words(data)
    if stem:
        data = stemming(data)
    if lemmatise:
        data = lemmatisation(data)
    if tokenise:
        data = tokenisation(data)
    if length:
        data = remove_length_outliers(data)
    return data


# ------- Analyse the Stanford Sentiment dataset in order to locate potential
# ------- sources of bias here
EEC_LOCATION: str = os.path.join(
    "data", "Equity-Evaluation-Corpus", "Equity-Evaluation-Corpus.csv"
)


def obtain_eec_df(eec_location: str = EEC_LOCATION
                  ) -> None:
    """
    Reusable function for obtaining a dataframe containing the EEC dataset
    """
    eec = pd.read_csv(eec_location)
    return eec


def get_demographic_statistics(data: pd.DataFrame,
                               demographic: str
                               ) -> None:

    print(f'\nStatistics for the "{demographic}" demographic '
          f'in the Stanford sentiment treebank dataset.')
    print('----------------------------------------------')
    print(f" - Demographic size:        {data['sentiments'].size}")
    print(f" - Mean sentiment:          {data['sentiments'].mean()}")
    print(f" - Modal sentiment (class): {data['class'].mode()[0]}")
    print(f" - Sentiment variance:      {data['sentiments'].var()}")
    print(f" - 3 most frequent sentiment scores (class): \n"
          f"   -> {data['class'].value_counts()[:3].index.tolist()}")


def get_synonyms(words: Tuple[str]
                 ) -> set[Union[str, Any]]:
    """
    Function for obtaining a set of synonyms for words from a list
    """
    syns = set()
    for word in words:
        syns.add(word)
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                syns.add(lemma.name())
    return syns


def get_demographic_data(data: pd.DataFrame,
                         words: Tuple[str],
                         visualise: bool = True
                         ) -> pd.DataFrame:
    """
    Function for extracting demographic data from the dataset, and placing this
    in a new dataframe.
    """
    def words_filter(lst: Union[list, str],
                     ) -> bool:
        if isinstance(lst, str):
            return any(w in word_tokenize(lst) for w in ws)
        else:
            return any(w in lst for w in ws)

    ws = get_synonyms(words)
    demog_data = data[data["sentences"].apply(words_filter)]

    if visualise:
        display_word_cloud(demog_data)

    demog_data.drop(["sentences"], axis=1, inplace=True)
    return demog_data


def compare_demographics(data: pd.DataFrame,
                         demographics: Tuple[str, ...],
                         words: Tuple[Tuple[str, ...], ...]
                         ) -> None:
    """
    Methods for comparing two demographic groups.
    """
    demographic_data = [get_demographic_data(data, ws) for ws in words]

    for i, d in enumerate(demographic_data):
        d['demographic'] = d['sentiments'].apply(
            lambda x: demographics[i]
        )
        get_demographic_statistics(d, demographics[i])

    full_df = pd.concat(demographic_data)

    # Visualise the variations in sentiment scores with a displot
    plt.figure(figsize=(9, 7.5))
    sns.displot(x="sentiments", hue="demographic", data=full_df, kind='kde')
    plt.show()

    # Visualise the variations in sentiment classes with a countplot
    plt.figure(figsize=(9, 7.5))
    sns.countplot(x="class", hue="demographic", data=full_df)
    plt.show()

    # Visualise the differing sentiment distributions using a boxplot
    plt.figure(figsize=(9, 7.5))
    sns.boxplot(x="demographic", y='sentiments', data=full_df)
    plt.show()


def analyse_demographics(data: pd.DataFrame,
                         additional: bool = False
                         ) -> None:
    """
    Function for comparing the statistics and gaining visualisations for the
    various demographic-related sentences in the dataset. This was no trivial
    procedure, as the dataset by defualt contains no explicit demographic
    features. Therefore, the following function comprises a method I personally
    devised, wherein lists of demographic-related vocabulary are created from
    the Equity Evaluation Corpus, which are augmented with custom terms as well
    as synonyms generated from the NLTK library (specifically, via the wordnet
    corpus). For each list, the dataset is crawled to locate any examples with
    sentences containing a word from that list, and these examples are
    extracted from the dataframe into a new one, and statistics are generated
    from there. Additionaly methodology contained here includes the
    construction of a 'class' feature, where the sentement scores are
    interpolated into one of 5-classes, allowing us to obtain statistics
    regarding which classes are most common amongst demographics.
    Visualisations, including wordclouds, are generated for each demographic,
    and barplots and boxplots are constructed to directly compare demographics.
    """
    # Create a sentiment "class" feature
    def cvt2cat(sentiment: float
                ) -> int:
        x = int((10*sentiment)/2)
        return x if x != 5 else 4
    data['class'] = data['sentiments'].apply(cvt2cat)

    # Obtain EEC and extract some demographic features from there
    eec = obtain_eec_df()
    aa_eec = eec[eec["Race"] == "African-American"]
    euro_eec = eec[eec["Race"] == "European"]
    male_eec = eec[eec["Gender"] == "male"]
    female_eec = eec[eec["Gender"] == "female"]

    # Create lists of demographic terms
    male = [x.split(" ")[-1] for x in male_eec['Person'].unique()]
    male += [
        "masculine", "boy", "boys", "man", "male", "men", "males"
    ]
    female = [x.split(" ")[-1] for x in female_eec['Person'].unique()]
    female += [
        "feminine", "girl", "girls", "woman", "female", "women", "females"
    ]
    african_american = [x.split(" ")[-1] for x in aa_eec['Person'].unique()]
    african_american += [
        "black", "blacks", "africa", "african"
    ]
    european = [x.split(" ")[-1] for x in euro_eec['Person'].unique()]
    european += [
        "white", "whites", "europe", "european"
    ]

    # Compare the gender demographics
    compare_demographics(
        data, ("Male", "Female"), (male, female)
    )

    # Compare the ethnic demographics
    compare_demographics(
        data, ("African-American", "European"), (african_american, european)
    )

    # Additional demographics to compare if toggled
    if additional:
        young = ("young", "youth", "new", "modern", "younger")
        old = ("old", "elderly", "aged", "vintage", "older")
        compare_demographics(data, ("young", "old"), (young, old))

        asian = ("asian", "chinese", "eastern", "japanese")
        compare_demographics(
            data, ("Asian", "European", "African-American"),
            (asian, european, african_american)
        )


def initial_data_analysis(data: pd.DataFrame) -> None:
    """
    Container function for the initial data analysis methods.
    """

    data_visualisation(data)
    print("Generating visualisations and statistics for demographic sentences"
          " in the dataset...")
    analyse_demographics(data)


# --------------------------------------------------------------------------- #
# ------------------ Part 3: Conventional Implementation -------------------- #
"""
The following section of code entails the conventional implementation of a
sentiment analysis machine learning model. Specifically, the ML algorithm used
is Support Vector Classification, taken directly from the scikit-learn library.
The model was trained using the training set (comprising 75% of the dataset),
and cross-fold validation was used (via GridSearchCV) to optimise the hyper-
parameters.

Emphasis is placed on hyperparameter tuning here: the purpose of this program
is to detect and mitigate bias in sentiment analysis, so it's very important
that any bias detected is the result of more general NLP-related
insufficiencies, and *not* simply a poorly configured model (i.e., if the
sentiment predictions vary significantly by differing demographic groups purely
because the model is making poor predictions, then we can't say that this is
down to 'bias' rather than simply bad predictions).

The model requires the vocabulary/sentences to be 'vectorised'. This is a key
aspect of the model, and a likely source of any potential bias. The word
embeddings selected are those from the word2vec model, which effectively
encapsulate word semantics, pretrained on a large Google news corpus.

Immediately following the model is methods for analysing the sentiment bias
in the predictions made by the trained model, using the EEC to input sentences
where the only difference will be in the demographic identity term present (for
instance, "The boy is angry.", and "The girl is angry." - the search for bias
will entail finding out how different the sentiment predictions are between
inputted sentences such as these, which in theory should have identical
sentiment scores as the demographic identity term presente should not
contribute to the final prediction).

Furthermore, to gain a more intuitive insight into the presence of any bias in
a semi-realistic sentiment analysis application, I wrote a very simple app that
takes user-inputted sentences and returns a sentiment score, predicted by the
model.
"""
# Support Vector Classification model,
# trained firstly using the orginal (i.e., biased) embeddings

from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import gensim.models.keyedvectors as Word2Vec 
GOOGLE_NEWS_W2V: str = os.path.join(
    'data', 'GoogleNews-vectors-negative300.bin.gz'
)


def obtain_w2v_model(limit: int) -> Word2Vec:
    """
    Retrieve the locally stored pretrained word2vec model
    """
    w2v = Word2Vec.KeyedVectors.load_word2vec_format(
        GOOGLE_NEWS_W2V, binary=True, limit=limit
    )
    return w2v


def normalise_embedding(vector: List[float]
                        ) -> np.ndarray:
    """
    Function for normalising the vectors
    """
    norm = np.linalg.norm(vector)
    return vector/norm


def naive_train_test_split(data: pd.DataFrame
                           ) -> Tuple[pd.DataFrame, ...]:
    """
    Naively split the data into a train and test set
    """
    train_data, test_data = train_test_split(
        data,
        train_size=0.75,
        test_size=0.25,
        random_state=13
    )
    return train_data, test_data


def split_x_y(data: pd.DataFrame
              ) -> Tuple[List[str], List[int]]:
    """
    Split the dataset into the features and outcome
    """
    return list(data["sentences"]), list(data["sentiments"])


def establish_training_data(data: pd.DataFrame,
                            embeddings: Word2Vec,
                            ) -> Tuple[List[str], List[int],
                                       List[str], List[int]]:
    """
    Final training preparations
    """

    # Remove any columns such that the sentences contain no words in the
    # Word2Vec model
    data = data[
        data['sentences'].map(
            lambda s: any(w in embeddings for w in s)
        )
    ]

    # Separate test and train data, then separate the x and y columns
    train_data, test_data = naive_train_test_split(data)

    return split_x_y(train_data), split_x_y(test_data)


def sentences2vectors(sentences: List[str],
                      embeddings: Word2Vec
                      ) -> pd.DataFrame:
    """
    Convert all sentences into normalised vector form, then take the mean
    of each: that is, all sentences are now lists of the means of the
    vector representations of the words that comprise them
    """

    # Helper function for mapping each word in a sentence into its vector
    # embedding, from the word2vec embeddings, normalising these vectors,
    # then calculating the mean of each
    def map2vec(sentence: List[str]
                ) -> List[np.ndarray]:
        return np.mean([
            normalise_embedding(embeddings[word])
            for word in sentence if word in embeddings
        ], axis=0)

    return list(map(map2vec, sentences))


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Used for cleaning data when raw strings are used as inputs
    """

    def fit(self,
            X: List[str],
            y: List[int] = None
            ) -> 'W2vVectoriser':
        return self

    def transform(self,
                  X: List[str]
                  ) -> List[float]:
        if isinstance(X[0], list) or isinstance(X[0], np.ndarray):
            return X
        data = pd.DataFrame([{"sentences": sentence} for sentence in X])
        data = tidy_data(data, length=False)
        return list(data['sentences'])


class W2vVectoriser(BaseEstimator, TransformerMixin):
    """
    Class for use in the machine learning pipeline; serves the purpose of
    converting natural language into their corresponding word2vec word vector.
    """

    def __init__(self, w2v: Word2Vec) -> None:
        self.w2v = w2v

    @staticmethod
    def sentences2vectors(sentences: List[str],
                          embeddings: Word2Vec
                          ) -> pd.DataFrame:
        """
        Convert all sentences into normalised vector form, then take the mean
        of each: that is, all sentences are now lists of the means of the
        vector representations of the words that comprise them
        """

        def map2vec(sentence: List[str]
                    ) -> List[np.ndarray]:
            """
            Helper function for mapping each word in a sentence into its vector
            embedding, from the word2vec embeddings, normalising these vectors,
            then calculating the mean of each
            """
            return np.mean([
                normalise_embedding(embeddings[word])
                for word in sentence if word in embeddings
            ], axis=0)

        return list(map(map2vec, sentences))

    def fit(self,
            X: List[str],
            y: List[int] = None
            ) -> 'W2vVectoriser':
        return self

    def transform(self,
                  X: List[str]
                  ) -> List[float]:
        if isinstance(X[0], np.ndarray):
            return X
        return self.sentences2vectors(X, self.w2v)


def init_support_vector_machine_pipeline(embeddings: Word2Vec
                                         ) -> Pipeline:
    """
    Initialise the support vector classifier pipeline, using parameters
    obtained via hyperparameter tuning with cross-fold validation.
    """
    data_cleaner = DataCleaner()
    vectoriser = W2vVectoriser(embeddings)
    truncated_svd = TruncatedSVD(n_components=250)
    std_scaler = StandardScaler()
    svc = SVC(C=0.1, probability=True, kernel='linear')
    return Pipeline([
        ('cleaner', data_cleaner),
        ('vec', vectoriser),
        ('svd', truncated_svd),
        ('scaler', std_scaler),
        ('svc', svc)
    ])


def linear_svc_grid_search(embeddings: Word2Vec
                           ) -> GridSearchCV:
    """
    Tune the hyperparameters of the support vector classifier. It's important
    that the algorithm itself is as good as it can be, such that the source of
    any bias isn't purely due to a poorly implemented model.
    """
    data_cleaner = DataCleaner()
    vectoriser = W2vVectoriser(embeddings)
    truncated_svd = TruncatedSVD()
    std_scaler = StandardScaler()
    svc = SVC(max_iter=5000)
    hyperparams = {
        "svd__n_components": [2, 50, 100, 200, 250],
        "svc__C": [0.001, 0.01, 0.1, 1, 10],
    }
    grid = Pipeline([
        ('cleaner', data_cleaner),
        ('vec', vectoriser),
        ('scaler', std_scaler),
        ('cv', GridSearchCV(
            estimator=Pipeline(
                [('svd', truncated_svd),
                 ('svc', svc)]
            ),
            param_grid=hyperparams,
            scoring="accuracy",
            n_jobs=-1,
            verbose=3
        ))
    ])
    return grid


def support_vector_classification(data: Tuple[List[str], List[int]],
                                  embeddings: Word2Vec,
                                  best_params: bool = False
                                  ) -> SVC:
    """
    Container function for Support Vector classification
    """
    train_X, train_y = data
    if best_params:
        model = linear_svc_grid_search(embeddings)
        model.fit(train_X, train_y)
        print(model['cv'].best_params_)
    else:
        model = init_support_vector_machine_pipeline(embeddings)
        model.fit(train_X, train_y)
    return model


# --------- Model evaluation
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def evaluate(model: Pipeline,
             test_data: Tuple[pd.DataFrame, ...],
             ) -> None:

    x_test, y_test = test_data
    predictions = model.predict(x_test)

    print(f"Accuracy score:  {accuracy_score(y_test, predictions)}.")
    print(f"F1 score:        {f1_score(y_test, predictions)}.")
    print(f"Precision score: {precision_score(y_test, predictions)}.")
    print(f"Recall score:    {recall_score(y_test, predictions)}.")
    print("Classification report: ")
    print(classification_report(y_test, predictions))
    print("Confustion matrix: ")
    print(confusion_matrix(y_test, predictions))


def test_inputs(model: Pipeline,
                embeddings: Word2Vec,
                form: str,
                proba: bool = True
                ) -> None:
    """
    Trial run of a sentiment-analysis application in order to gain insight into
    the model in action, and to gain a more intuitive sense of the
    effectiveness of the debiasing.
    """
    print("\n--------------------------------------------")
    print("Now predicting sentence sentiment. Input ':quit' to exit.")
    while True:
        inp = input("Type in a sentence: ")
        if inp == ":quit":
            break
        elif len(inp) == 0:
            print("Sentence not recognised, sorry.\n")
            continue
        else:
            if not any(w in embeddings for w in inp.split(" ")):
                print("Sentence not recognised, sorry.\n")
                continue

        try:
            if not proba or form == "regression":
                score = model.predict([inp])[0]
            else:
                score = model.decision_function([inp])[0]
        except ValueError:
            print("Sentence not recognised, sorry.\n")
            continue

        if form == "five class" and not proba:
            if score == 0:
                print("This sentence is very negative.\n")
            elif score == 1:
                print("This sentence is negative.\n")
            elif score == 2:
                print("This sentence is neutral.\n")
            elif score == 3:
                print("This sentence is positive.\n")
            elif score == 4:
                print("This sentence is very positive.\n")

        elif form == "two class" and not proba:
            if score == 1:
                print("This sentence is positive.\n")
            else:
                print("This sentence is negative.\n")

        else:
            print(f"Sentiment level: {score}\n")

    print("\nThank you for your time.")
    print("--------------------------------------------\n")


# ----- Conventional Main

def conventional_implementation(data: pd.DataFrame,
                                w2v_embeddings: Word2Vec,
                                regression: bool = False,
                                binary: bool = True,
                                test_model: bool = False,
                                encode_sentiments: bool = True
                                ) -> Pipeline:
    """
    Container function for the conventional sentiment analysis implementation,
    without any bias-mitigation measures applied.
    """
    print("Commencing pre-processing...")
    if encode_sentiments:
        data = handle_sentiments(
            data, make_classification=(not regression), make_binary=binary
        )
    data = tidy_data(data)
    print("Pre-processing finished!")

    print("Commencing machine learning...")
    train_data, test_data = establish_training_data(
        data, w2v_embeddings
    )

    model = support_vector_classification(train_data, w2v_embeddings, False)
    print("Support Vector classifier learned!")

    print("Evaluating model...")
    evaluate(model, test_data)

    if test_model:
        if binary and not regression:
            test_inputs(model, w2v_embeddings, "two class")
        elif not regression:
            test_inputs(model, w2v_embeddings, "five class")
        else:
            test_inputs(model, w2v_embeddings, "regression")

    return model, test_data


# -------- Bias analysis in machine learning model
"""
The following functions will be used to analyse/attempt to observe any bias in
the trained machine learning models. These functions will be reusable, as in
they will be used to analyse and compare the bias present in both the
conventional and debiased models.
"""


def analyse_demographic_predictions(model: Pipeline,
                                    to_predict: List[str],
                                    demographic: str,
                                    emotion: str,
                                    status: str,
                                    do_print: bool = False
                                    ) -> Dict[str, Dict[str, float]]:
    """
    Gather statistics for the predictions.
    """
    if do_print:
        print(f"\nBias statistics for the \"{demographic}\" demographic. "
              f"(Emotion: '{emotion}', Status = {status}.)")
        print("-"*LINE_LENGTH)

    predictions = model.decision_function(to_predict)
    n = len(predictions)
    num_neg = sum(1 for x in predictions if x < 0)
    avg_pred = sum(predictions)/n
    max_pred = max(predictions)
    min_pred = min(predictions)
    variance = np.var(predictions)

    if do_print:
        print(f"Percentage of negative predictions: {(num_neg/n)*100}")
        print(f"Average prediction value:           {avg_pred}")
        print(f"Most positive prediction:           {max_pred}")
        print(f"Most negative prediction:           {min_pred}")
        print(f"Variance of predictions:            {variance}")

    return {
        'Demographic': demographic, 'Emotion': emotion,
        'Average': avg_pred, 'Max': max_pred, 'Min': min_pred,
        'Predictions': predictions, 'Variance': variance
    }


def emotion_based_statistics(data: pd.DataFrame,
                             emotion: str,
                             status: str
                             ) -> None:
    """
    Compare the results for all demographics
    """
    print(f"\nBias statistics for the all demographcs. "
          f"(Emotion: '{emotion}', Status = {status}.)")
    print("-"*LINE_LENGTH)

    averages = data["Average"]
    max_s = max(averages)
    min_s = min(averages)

    print(f"Variance of average sentiments scores: {np.var(averages)}")
    print(f"Range of average sentiment scores:     {np.abs(max_s - min_s)}")


def model_bias_analysis_visualisations(data: pd.DataFrame,
                                       emotion: str,
                                       stats_df: pd.DataFrame,
                                       status: str,
                                       boxplots: bool = False
                                       ) -> None:
    """
    Visualisations for the model bias analysis.
    """
    plt.figure(figsize=(7, 7))
    plt.title(f"Emotion = '{emotion}', Status = {status}.")
    sns.barplot(x="Demographic", y="Average", data=stats_df)
    plt.show()

    # Optionally, display some boxplots for the data (turned off by default as
    # they are difficult to reason about)
    if boxplots:
        plt.figure(figsize=(7, 7))
        plt.title(f"Emotion = '{emotion}', Status = {status}.")
        sns.boxplot(x="Race", y="Prediction", data=data)
        plt.show()

        plt.figure(figsize=(7, 7))
        plt.title(f"Emotion = '{emotion}', Status = {status}.")
        sns.boxplot(x="Gender", y="Prediction", data=data)
        plt.show()


def model_bias_analysis(model: Pipeline,
                        embeddings: Word2Vec,
                        status: str,
                        display_indiv: bool = False
                        ) -> Tuple[pd.DataFrame, ...]:
    """
    Analyse bias in the sentiment analysis predictions by making predictions
    for various sentences with variations in demographic.
    """

    def predict_sentence_sentiment(s: str
                                   ) -> float:
        return model.decision_function([s])[0]

    # Obtain the EEC, and use the trained model to predict the sentiment of the
    # sentences in the EEC
    eec = obtain_eec_df()
    eec["Prediction"] = eec["Sentence"].apply(predict_sentence_sentiment)

    # Separate EEC by demographic
    aa = eec[eec["Race"] == "African-American"]
    euro = eec[eec["Race"] == "European"]
    male = eec[eec["Gender"] == "male"]
    female = eec[eec["Gender"] == "female"]

    # To store the minimum, maximum, and average prediction values for each
    # demographic
    stats_dicts = []

    # Make predictions and analyse
    for df, demog in (
        (aa, "African-American"),
        (euro, "European"),
        (male, "Male"),
        (female, "Female"),
    ):
        for tup in (
            (list(df["Sentence"]), demog, "all"),
            (list(df[df["Emotion"]=='anger']["Sentence"]), demog, "anger"),
            (list(df[df["Emotion"]=='sadness']["Sentence"]), demog, "sadness"),
            (list(df[df["Emotion"]=='fear']["Sentence"]), demog, "fear"),
            (list(df[df["Emotion"]=='joy']["Sentence"]), demog, "joy"),
        ):
            stats_dicts.append(
                analyse_demographic_predictions(
                    model, tup[0], tup[1], tup[2], status
                )
            )

    # Convert the stats dictionaries to a DataFrame for visulaisations
    stats_df = pd.DataFrame(stats_dicts)

    # Separate EEC by emotions
    anger = eec[eec["Emotion"] == "anger"]
    sadness = eec[eec["Emotion"] == "sadness"]
    fear = eec[eec["Emotion"] == "fear"]
    joy = eec[eec["Emotion"] == "joy"]
    for df, emotion in (
        (eec, "all"),
        (anger, "anger"),
        (sadness, "sadness"),
        (fear, "fear"),
        (joy, "joy"),
    ):
        emotion_based_statistics(
            stats_df[stats_df['Emotion'] == emotion], emotion, status
        )
        if display_indiv:
            model_bias_analysis_visualisations(
                df, emotion, stats_df[stats_df['Emotion'] == emotion], status
            )

    # Plot a bar chart for all demographics and emotions at once
    plt.figure(figsize=(6, 5))
    plt.title(f"Status = {status}.")
    plt.ylim(-1.6, 1.6)
    sns.barplot(x="Emotion", y="Average", hue="Demographic", data=stats_df)
    plt.show()

    stats_df['Status'] = [status for _ in range(len(stats_df))]
    return eec, stats_df


# -- Stratified debiased subsample
"""
The following two functions comprise the 'unbiased subsample' requirement of
the coursework. The subsampling was done by first creating some new features
on the dataset, namely 'Male', 'Female', 'African-American', and 'European'
(these were selected as these are the demographics found in the Equity
Evaluation Corpus). Then, the number of positive instances and the number of
negative instances in the dataset for each gender was found, and the smallest
of these two values was taken, let's call it n. Then, the number of positive
and negative instances of each gender was reduced to size n (i.e., they were
equalised, meaning there was the same number of positive and negative instances
in the dataset for each gender). The same process was done for the 'ethnicity'
demographic (i.e., African-American and European). Care has been taken to
ensure that this procedure genuinely lead to a fairer subsample, and the
approach has been taken in a very logical manner, as can be seen in the
following functions.
"""


def create_demographic_features(data: pd.DataFrame
                                ) -> pd.DataFrame:
    """
    Helper function for augmenting the dataset to include one-hot columns for
    various specified demographic features.
    """

    # Function for applying the one-hot columns
    def construct_feature(sentence: str,
                          demographic: set
                          ) -> int:
        return 1 if any(
            w in demographic for w in word_tokenize(sentence)
        ) else 0

    # Obtain EEC and extract some demographic features from there
    eec = obtain_eec_df()
    aa_eec = eec[eec["Race"] == "African-American"]
    euro_eec = eec[eec["Race"] == "European"]
    male_eec = eec[eec["Gender"] == "male"]
    female_eec = eec[eec["Gender"] == "female"]

    # Create lists of demographic terms
    male = [x.split(" ")[-1] for x in male_eec['Person'].unique()]
    male += [
        "masculine", "boy", "boys", "man", "male", "men", "males"
    ]
    female = [x.split(" ")[-1] for x in female_eec['Person'].unique()]
    female += [
        "feminine", "girl", "girls", "woman", "female", "women", "females"
    ]
    african_american = [x.split(" ")[-1] for x in aa_eec['Person'].unique()]
    african_american += [
        "black", "blacks", "africa", "african"
    ]
    european = [x.split(" ")[-1] for x in euro_eec['Person'].unique()]
    european += [
        "white", "whites", "europe", "european"
    ]

    # Augment the demographic lists with synonyms
    male = get_synonyms(male)
    female = get_synonyms(female)
    african_american = get_synonyms(african_american)
    european = get_synonyms(european)

    # Apply the demographic features
    data["Male"] = data["sentences"].apply(
        lambda s: construct_feature(s, male)
    )
    data["Female"] = data["sentences"].apply(
        lambda s: construct_feature(s, female)
    )
    data["African-American"] = data["sentences"].apply(
        lambda s: construct_feature(s, african_american)
    )
    data["European"] = data["sentences"].apply(
        lambda s: construct_feature(s, european)
    )

    return data


def debiased_subsample(data: pd.DataFrame
                       ) -> Tuple[List[str], List[int]]:
    """
    For the sake of complexity, only the gender and two ethnic attributes are
    considered (due to their presence in the EEC), but this process may be
    augmented to include more. That is to say, this process should work without
    loss of generality.
    """

    # Append an index column to the dataset to distinguish b/t examples
    data["id"] = list(range(len(data)))

    # Augment the dataset to include demographic features
    data = create_demographic_features(data)

    # Apply data tidying
    data = handle_sentiments(data, make_binary=True)

    # Establish predicates
    pred_m = (data["Male"] == 1)
    pred_f = (data["Female"] == 1)
    pred_a = (data["African-American"] == 1)
    pred_e = (data["European"] == 1)

    # Separate the non-demographic examples (note: ~ = "not")
    safe_data = data[~pred_m & ~pred_f & ~pred_a & ~pred_e]

    # If a training example contains both M and F, or AA and E, and is not
    safe_data.append(data[~pred_m & ~pred_f & pred_a & pred_e])
    safe_data.append(data[pred_m & pred_f & ~pred_a & ~pred_e])

    # Separate the dataframe by demographics
    data_m = data[pred_m & ~pred_f & ~pred_a & ~pred_e]  # Male
    data_f = data[~pred_m & pred_f & ~pred_a & ~pred_e]  # Female
    data_a = data[~pred_m & ~pred_f & pred_a & ~pred_e]  # African-American
    data_e = data[~pred_m & ~pred_f & ~pred_a & pred_e]  # European

    data_ma = data[pred_m & ~pred_f & pred_a & ~pred_e]  # Male + AA
    data_fa = data[~pred_m & pred_f & pred_a & ~pred_e]  # Female + AA
    data_me = data[pred_m & ~pred_f & ~pred_a & pred_e]  # Male + E
    data_fe = data[~pred_m & pred_f & ~pred_a & pred_e]  # Female + E

    # Extract sentiment scores
    data_m_0 = data_m[data_m["sentiments"] == 0]  # Male + 0
    data_m_1 = data_m[data_m["sentiments"] == 1]  # Male + 1
    data_f_0 = data_f[data_f["sentiments"] == 0]  # Female + 0
    data_f_1 = data_f[data_f["sentiments"] == 1]  # Female + 1
    data_a_0 = data_a[data_a["sentiments"] == 0]  # AA + 0
    data_a_1 = data_a[data_a["sentiments"] == 1]  # AA + 1
    data_e_0 = data_e[data_e["sentiments"] == 0]  # E + 0
    data_e_1 = data_e[data_e["sentiments"] == 1]  # E + 1
    data_ma_0 = data_ma[data_ma["sentiments"] == 0]  # Male + AA + 0
    data_ma_1 = data_ma[data_ma["sentiments"] == 1]  # Male + AA + 1
    data_fa_0 = data_fa[data_fa["sentiments"] == 0]  # Female + AA + 0
    data_fa_1 = data_fa[data_fa["sentiments"] == 1]  # Female + AA + 1
    data_me_0 = data_me[data_me["sentiments"] == 0]  # Male + E + 0
    data_me_1 = data_me[data_me["sentiments"] == 1]  # Male + E + 1
    data_fe_0 = data_fe[data_fe["sentiments"] == 0]  # Female + E + 0
    data_fe_1 = data_fe[data_fe["sentiments"] == 1]  # Female + E + 1

    # Get the shortest length for each gender/race per sentiment score
    n_g_0 = min(len(data_m_0), len(data_f_0))  # Male/female + 0
    n_g_1 = min(len(data_m_1), len(data_f_1))  # Male/female + 1
    n_r_0 = min(len(data_a_0), len(data_e_0))  # AA/E + 0
    n_r_1 = min(len(data_a_1), len(data_e_1))  # AA/E + 1
    n_mr_0 = min(len(data_ma_0), len(data_me_0))  # Male + AA/E + 0
    n_mr_1 = min(len(data_ma_1), len(data_me_1))  # Male + AA/E + 1
    n_fr_0 = min(len(data_fa_0), len(data_fe_0))  # Female + AA/E + 0
    n_fr_1 = min(len(data_fa_1), len(data_fe_1))  # Female + AA/E + 1

    # Get shortest overall for each of gender and race
    n_g = min(n_g_0, n_g_1)  # Minimum length of gender terms
    n_r = min(n_r_0, n_r_1)  # Minimum length of race terms
    n_mr = min(n_mr_0, n_mr_1)
    n_fr = min(n_fr_0, n_fr_1)
    n_gr = min(n_mr, n_fr)  # Minimum length of gender/race terms

    # Reduce datasets
    subsample = safe_data.append(
        pd.concat([
            # Gender
            data_m_0[:n_g],
            data_m_1[:n_g],
            data_f_0[:n_g],
            data_f_1[:n_g],

            # Race
            data_a_0[:n_r],
            data_a_1[:n_r],
            data_e_0[:n_r],
            data_e_1[:n_r],

            # Both
            data_ma_0[:n_gr],
            data_ma_1[:n_gr],
            data_me_0[:n_gr],
            data_me_1[:n_gr],
            data_fa_0[:n_gr],
            data_fa_1[:n_gr],
            data_fe_0[:n_gr],
            data_fe_1[:n_gr],
        ])
    )

    print("Unbiased subsample stats\n" + "-"*LINE_LENGTH)
    print(subsample.info())
    print(subsample["Male"].value_counts())
    print(subsample["Female"].value_counts())
    print(subsample["African-American"].value_counts())
    print(subsample["European"].value_counts())
    print("-"*LINE_LENGTH + '\n')

    subsample.drop(
        labels=["Male", "Female", "African-American", "European"], 
        axis=1, inplace=True
    )

    return subsample


# ----


# --------------------------------------------------------------------------- #
# -------- Part 4: Enhanced Machine Learning Model, Minimising Bias --------- #

# DEBIASING THE WORD VECTORS
"""
The following will be my implementation of the method outlined in the article
pertaining to the debiasing of word vectors. As with the conventional model
from above, I will use the word2vec embeddings pretrained on the Google news
data, but before the machine learning takes place this time, the vectors will
be debiased in order to reduce unfairness present in the downstream sentiment
analysis.

The aforementioned debiasing method utilises adversarial learning to
decorrelate certain vectors with sentiment. More specifically, any demographic
identity term (for instance, 'Jewish', 'African', etc.) whose vector resides in
an established negative sentiment subspace will be shifted towards a more
sentimentally neutral position, and the same for demographic identitiy term
vectors that reside in an established positive sentiment subspace (i.e., they
will be shifted in the opposite direction as those before, still torwards the
neutral equilibrium).

To undertake the adversarial learning method, first the positive and negative
sentiment subspaces, mentioned above, need to be established. This presents a
minor challenge as, by default, the Word2Vec word embeddings are not assigned
any sentiment value. To resolve this, a common solution (which I shall use in
my implementation) is to decompose a selection of the original multi-
dimensional word vectors - this selection corresponding with some 'ground
truth' positive and negative words (for instance, 'bad', 'awful', etc. for
negative, and 'good', 'happy', etc. for positive). Then, two matrices are made,
one for the positive and one for the negative decomposed vectors. We take the
most significant component of each, then the signed difference between these
principle components. This signed difference will be classified as the
'directional sentiment vector', connecting the positive and negative subspaces.
Using this directional sentiment vector, we are now able to project any given
word vector onto it to gain some insight into the sentiment polarity of that
word vector (and thus the word it represents).

To obtain the aforementioned 'ground truth' positive and negative words, I
utilise the Opinion Lexicon dataset found with the codebase.

This algorithm is trained on a very large corpus of words (i.e., 4,000,000
words from the Google news dataset, from the pretrained Word2Vec model made
available by the `gensim` library), however it will only be used to re-embed
specified (i.e., demographic identity terms). This is where the central
downside of this model lies: the words that should be debiased must be
specified by the user before the training takes place, as the model has no way
of detecting what word vectors contain bias otherwise.

In order to analyse the effect (and success) of this bias-mitigating method,
the 'Equity Evaluation Corpus (EEC)' is used. This dataset was specifically
designed to manifest bias specifically regarding *race* and *gender*.
Originally, it was utilised in response to the 2018 SemEval task 1 valency
analysis task by examining the bias associated with over 200 sentiment analysis
algorithms.
"""
NEGATIVE_WORDS: str = os.path.join(
    "data", "opinion-lexicon-English", "negative-words.txt"
)
POSITIVE_WORDS: str = os.path.join(
    "data", "opinion-lexicon-English", "positive-words.txt"
)


# --- Unveil the negative and positive subspaces in the embeddings,
# --- and obtain the corresponding directional sentiment vector

def collect_ground_truth_words(word_vectors: Word2Vec,
                               positives_loc: str = POSITIVE_WORDS,
                               negatives_loc: str = NEGATIVE_WORDS
                               ) -> Dict[str, List[str]]:
    """
    Read in all the positive and negative words from the Sentiment Opinion
    Lexicon (so long as the word exists in the word2vec embeddings) for use as
    the ground truth postive and negative words. These words shall be used to
    establish the positive and negative vector subspaces.
    """
    def criteria(l: str) -> bool:
        return l and not l.startswith(';') and l in word_vectors

    with open(negatives_loc) as neg:
        negatives = [line for line in neg.read().split('\n') if criteria(line)]
    with open(positives_loc) as pos:
        positives = [line for line in pos.read().split('\n') if criteria(line)]

    return {'negative': negatives, 'positive': positives}


def collect_ground_truth_vectors(word_vectors: Word2Vec,
                                 ground_truths: Dict[str, List[str]]
                                 ) -> Dict[str, np.ndarray]:
    """
    Function for converting the 'ground truth' positive and negative words
    collected in the prior function into their vector equivalents using the
    word2vec word embeddings model. These vectors are saved each in a positive
    and negative matrix, collectively stored in a Python dictionary.
    """
    ground_truth_matrices = {}
    for sentiment in ('negative', 'positive'):
        ground_truth_matrices[sentiment] = np.array([
            word_vectors[word] for word in ground_truths[sentiment]
        ])
    return ground_truth_matrices


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
def verify_ground_truth_vectors(gt_matrices: Dict[str, np.ndarray],
                                word_vectors: Word2Vec
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is simply for verifying that the word vectors for the 'ground
    truth' positive and negative words are consistent by checking that they can
    be used within a basic Logistic Regression model for predicting sentiment
    of individual words. This model can also be used briefly evaluate bias in
    word embeddings by predicting the mostl likely outcome for provided
    keywords.
    """

    # Helper function: establish binary classification
    def label(s: str) -> int:
        return 0 if s == 'negative' else 1

    # Create a dataframe for the positive and negative ground truth words
    to_df = {}
    for sentiment in ('negative', 'positive'):
        to_df[sentiment] = [
            {'vector': vector, 'sentiment': label(sentiment)}
            for vector in gt_matrices[sentiment]
        ]
    ground_truth_df = pd.DataFrame(to_df['positive'] + to_df['negative'])

    # Use stratified sampling to split the dataframe, so an even proportion of
    # positive and negative words reside in the test and train sets
    strat = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.3,
        random_state=42
    )
    strat_split = strat.split(
        ground_truth_df, ground_truth_df['sentiment']
    )
    for train_index, test_index in strat_split:
        train = ground_truth_df.iloc[train_index]
        test = ground_truth_df.iloc[test_index]
    x_train, y_train = list(train['vector']), train['sentiment']
    x_test, y_test = list(test['vector']), test['sentiment']

    # Initialise, fit, and evaluate a basic logistic regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    evaluate(model, (x_test, y_test))
    """
    Accuracy: 95%
    """
    return model


def illustrate_principle_components(eigenvalues: np.ndarray
                                    ) -> None:
    """
    Function for visualising the percentages of varianced explained by the
    principle components, mainly implemented in order to verify that the top
    principle component alone captures the sentiment subspace (as is suggested
    to be the case in the literature)
    """
    plt.figure(figsize=(7.5, 5))
    num_display = 20
    eigenvalues_df = pd.DataFrame(
        [{'Principle Component': idx+1, 'Eigen Value': val}
         for idx, val in enumerate(eigenvalues[:num_display])])
    sns.barplot(x='Principle Component',
                y='Eigen Value',
                data=eigenvalues_df,
                palette=sns.color_palette("crest"))
    plt.show()


def get_principle_components(matrix: np.ndarray,
                             illustrate: bool = True
                             ) -> np.ndarray:
    """
    To avoid repetion, general function for obtaining the principle PCA
    component of a matrix (will be applied to both the negative and positive
    ground truth matrices)
    """

    # Obtain the covariance matrix from the ground truth matrix, and extract
    # the eigenvalues from there
    covariance = matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # Display a barchart of the principle component values
    if illustrate:
        illustrate_principle_components(eigenvalues)

    # Obtain the maximum-valued eigen vector (i.e., the most principle
    # component)
    max_idx = np.argmax(eigenvalues)
    principle_component = np.real(eigenvectors[:, max_idx])

    # Normalise and return the principle component
    normalised_principle_component = normalise_embedding(principle_component)
    return normalised_principle_component


def create_directional_sentiment_vector(gt_matrices: Dict[str, np.ndarray]
                                        ) -> np.ndarray:
    """
    Using the ground truth positive and negative sentiment vectors, construct
    the directional sentiment word vector, which the program will go on to use
    as a tool for detecting the overarching sentiment of various other word
    vectors. This vector is constructed by taking the most significant
    components from the PCAs of the negative and positive ground truth
    matrices, and computing the signed difference of these.
    """

    # Ensure the positive and negative word matrices are of the same size
    # (i.e., reduce the size of the larger list to the size of the smaller one)
    lengths = {
        'negative': len(gt_matrices['negative']),
        'positive': len(gt_matrices['positive'])
    }
    n_min = min(lengths['negative'], lengths['positive'])
    for sentiment in ('negative', 'positive'):
        gt_matrices[sentiment] = (
            gt_matrices[sentiment][np.random.choice(
                lengths[sentiment], n_min, replace=False
            ), :]
        )

    # Get the principle PCA components for the negative and positive ground
    # truth words
    principle_components = {
        sentiment: get_principle_components(gt_matrices[sentiment])
        for sentiment in ('negative', 'positive')
    }

    # Compute the signed difference between the two principle components, and
    # normalise: and voila, we have the directional sentiment vector
    directional_sentiment_vector = (
        normalise_embedding(
            principle_components['positive'] - principle_components['negative']
        )
    )
    return directional_sentiment_vector


def sanity_check_dsv(w2v_embeddings: Word2Vec,
                     dsv: np.array
                     ) -> None:
    """
    Method for verifying that the directional sentiment vector obtained fufills
    its purpose of being capable of projecting negative words onto the negative
    subspace, and vice versa with positive words. This is done by finding the
    most most negative and most positive words present in the word embeddings
    according to the DSV.
    """
    df = pd.DataFrame(
        [{"word": word,
          "sentiment": normalise_embedding(w2v_embeddings[word]).dot(dsv)}
         for word in w2v_embeddings.index_to_key]
    )
    df.sort_values(by="sentiment", inplace=True)
    print("\nMost negative words, according to the DSV:")
    print(df.head(10))
    print("\nMost positive words, according to the DSV:")
    print(df.tail(10))
    print()


def sanity_check_gt(dsv: np.ndarray,
                    gt_matrices: Dict[str, np.ndarray]
                    ) -> None:
    """
    Additional function for checking that the directional sentiment vector
    works as expected (i.e., the signs for the dot product between the dsv and
    words of negative sentiment are all the same, and opposite to those of
    positive sentiment)
    """
    num_examples = 50
    n, m = len(gt_matrices['positive']), len(gt_matrices['negative'])

    print("\nThe DSV")
    print(dsv)

    print("\nPOSITIVE")
    for vec in gt_matrices['positive'][
            np.random.choice(n, num_examples, replace=False), :]:
        print(normalise_embedding(vec).dot(dsv))  # Project onto dsv

    print("\nNEGATIVE")
    for vec in gt_matrices['negative'][
            np.random.choice(m, num_examples, replace=False), :]:
        print(normalise_embedding(vec).dot(dsv))  # Project onto dsv


def measure_dsv_accuracy(dsv: np.ndarray,
                         gt_matrices: Dict[str, np.ndarray]
                         ) -> None:
    """
    Measure the accuracy of the directional sentiment vector by obtaining the
    proportion of correctly projected words (i.e., negative words onto the
    negative subspace and vice versa).
    """
    print("\nMeasuring accuracy of the directional sentiment vector: ")
    print("-"*LINE_LENGTH)

    # Get number of embeddings for each sentiment
    n_neg = len(gt_matrices['negative'])
    n_pos = len(gt_matrices['positive'])

    # Compute the number of accurate projections
    neg_sum = sum(
        1 for vec in gt_matrices['negative']
        if normalise_embedding(vec).dot(dsv) < 0
    )
    pos_sum = sum(
        1 for vec in gt_matrices['positive']
        if normalise_embedding(vec).dot(dsv) > 0
    )
    neg_acc = 100*neg_sum/n_neg
    pos_acc = 100*pos_sum/n_pos
    tot_acc = 100*(neg_sum + pos_sum)/(n_neg + n_pos)

    # Display results
    print(f"Percentage of accurate negative projections:  {neg_acc:.3f}%")
    print(f"Percentage of accurate negative projections:  {pos_acc:.3f}%")
    print(f"Overall percentage of accurate projections:   {tot_acc:.3f}%")
    print("-"*LINE_LENGTH, '\n')


def obtain_directional_sentiment_vector(w2v_embeddings: Word2Vec
                                        ) -> np.ndarray:
    """
    Container function for the collection of the directional sentiment vector,
    an essential part of the adversarial debiasing.
    """
    ground_truth_words = collect_ground_truth_words(w2v_embeddings)
    ground_truth_matrices = collect_ground_truth_vectors(
        w2v_embeddings, ground_truth_words
    )
    # verify_ground_truth_vectors(ground_truth_matrices, w2v_embeddings)
    dsv = create_directional_sentiment_vector(ground_truth_matrices)
    # sanity_check_dsv(w2v_embeddings, dsv)
    measure_dsv_accuracy(dsv, ground_truth_matrices)

    print("The DSV has been obtained!")

    return dsv


# ----- Use the EEC (Equity Evaluation Corpus) to search for bias in the word
# ----- embeddings (from word2vec, pretrained on Google news)


# ------- Define protected groups
"""
I was unable to locate any one dataset of demographic identity terms, or even
a list of them, so I had to extract them from various sources, and  I placed
them in some Python lists. The main source was the following:
    - https://www.ons.gov.uk/methodology/classificationsandstandards/ \
        measuringequality/ethnicgroupnationalidentityandreligion
"""

DEMOGRAPHIC_IDENTITY_TERMS: List[str] = [
    "English", "Welsh", "Scottish", "Irish", "British", "Gypsy", "White",
    "Caribbean", "African", "Asian", "Mixed", "Indian", "Pakistani",
    "Bangladeshi", "Chinese", "Black", "Arab", "Polish", "Northern",
    "Southern", "Traveller",
]

RELIGIONS_GLOSSARY: List[str] = [
    "Christian", "Catholic", "Protestant", "Buddhist", "Hindu", "Jewish",
    "Muslim", "Sikh", "Presbyterian", "Methodist", "Baptist", "Brethren",
    "Taoist", "Daoist", "Atheist", "Monotheist"
]

ETHNICITY_AND_RACE_GLOSSARY: List[str] = [
    "Asian", "Indian", "African", "Caribbean", "Afro-Caribbean", "Bangladeshi",
    "Black", "Caucasian", "Chinese", "European", "Hindu", "Hispanic",
    "Indigenous", "Irish", "Majority", "Minority", "Native", "Occidental",
    "Oriental", "Pakistani", "South", "Southern", "West", "Western", "North",
    "Northern", "East", "Eastern", "White"
]

GENDER_AND_SEXUALITY: List[str] = [
    "Heterosexual", "Homosexual", "Lesbian", "Gay", "Bisexual", "Transexual",
    "Male", "Female", "Mother", "Father", "Brother", "Sister"
]

# Names taken directly from the EEC
PEOPLE: List[str] = [
    'Alonzo', 'Jamel', 'Alphonse', 'Jerome', 'Leroy', 'Torrance', 'Darnell',
    'Lamar', 'Malik', 'Terrence', 'Adam', 'Harry', 'Josh', 'Roger', 'Alan',
    'Frank', 'Justin', 'Ryan', 'Andrew', 'Jack', 'he', 'man', 'boy', 'brother',
    'son', 'husband', 'boyfriend', 'father', 'uncle', 'dad', 'Nichelle',
    'Shereen', 'Ebony', 'Latisha', 'Shaniqua', 'Jasmine', 'Tanisha', 'Tia',
    'Lakisha', 'Latoya', 'Amanda', 'Courtney', 'Heather', 'Melanie', 'Katie',
    'Betsy', 'Kristin', 'Nancy', 'Stephanie', 'Ellen', 'she', 'woman', 'girl',
    'sister', 'daughter', 'wife', 'girlfriend', 'mother', 'aunt', 'mom', 'him',
    'her', 'they', 'them', 'their'
]

TERMS: List[str] = (
    DEMOGRAPHIC_IDENTITY_TERMS +
    RELIGIONS_GLOSSARY +
    ETHNICITY_AND_RACE_GLOSSARY +
    GENDER_AND_SEXUALITY +
    PEOPLE
)


def get_group_embeddings(embeddings: Word2Vec,
                         group: pd.Series
                         ) -> List[np.ndarray]:
    """
    Obtain a DataFrame of word vectors corresponding with a list of words.
    """
    group_embeddings = []
    for w in group:
        word = w.split(' ')[-1]
        if word in embeddings:
            group_embeddings.append(
                {'word': word, 'vector': embeddings[word]}
            )
        if word.lower() != word and word.lower() in embeddings:
            group_embeddings.append(
                {'word': word.lower(), 'vector': embeddings[word.lower()]}
            )
    return pd.DataFrame(group_embeddings)


def apply_dsv_bulk(df: pd.DataFrame,
                   dsv: np.ndarray
                   ) -> pd.DataFrame:
    """
    Create a 'sentiment' column, which entails projecting the DSV onto the
    word vectors
    """
    df['sentiments'] = df['vector'].apply(
        lambda v: dsv.dot(normalise_embedding(v))
    )
    return df


def demographic_embeddings_bias_stats(df: pd.DataFrame,
                                      demographic: str
                                      ) -> None:
    """
    Collect and display some statistics associated with the sentiment scores
    of the corresponding demographic group word vectors
    """
    print(f"\nAnalysing bias in the Word2Vec word embeddings "
          f"for the \"{demographic}\" demographic."
          f"\n---------------------------------------------------------------")
    print(f"Mean sentiment score:    {df['sentiments'].mean()}")
    print(f"Most positive sentiment: {max(df['sentiments'])}")
    print(f"Most negative sentiment: {min(df['sentiments'])}")
    print(f"Sentiment variance:      {df['sentiments'].var()}")


def analyse_word2vec_bias(embeddings: Word2Vec,
                          dsv: np.ndarray
                          ) -> None:
    """
    Using the directional sentiment vector (obtained below, as a part of the
    'debiased implementation' section), the word embeddings themselves will
    be analysed to search for sentiment-related bias associated with different
    demographics. In particular, the EEC (Equity Evaluation Corpus) will be
    utilised in order to extract demographic related names (for European and
    African-American names, in particular), as well as gender terms.
    """
    eec = obtain_eec_df()
    african_american_names = get_group_embeddings(
        embeddings, eec[eec["Race"] == "African-American"]["Person"].unique()
    )
    european_names = get_group_embeddings(
        embeddings, eec[eec["Race"] == "European"]["Person"].unique()
    )
    male_names = get_group_embeddings(
        embeddings, eec[eec["Gender"] == "male"]["Person"].unique()
    )
    female_names = get_group_embeddings(
        embeddings, eec[eec["Gender"] == "female"]["Person"].unique()
    )

    african_american_names, european_names, male_names, female_names = [
        apply_dsv_bulk(df, dsv) for df in (
            african_american_names, european_names, male_names, female_names
        )
    ]

    for demog in (
            ("African-American names", african_american_names),
            ("European names", european_names),
            ("male names", male_names),
            ("female names", female_names)
        ):
        demographic_embeddings_bias_stats(demog[1], demog[0])


# ---- Adversarial Learning for Mitigating Sentiment Bias in Word Vectors
"""
The functions of this section will collectively comprise the central debiasing
procedure. This procedure follows an adversarial learning approach that is
designed especially for reducing sentiment bias embedded in word vectors.
Although this particular implementation specifically utilises Word2Vec as the
word embedding model of choice, the article from which this method originates
claims that it has been tested also with GloVe - thus, if this implementation
is successful, then it can be deduced that this method works for various word
embedding models.

To begin, we first define the depolaraised sentiment vector as:
    - y_hat = y - w * w.T * y
where w are learned weights, and y is a biased (or rather, a non-debiased) word
vector. I specified non-debiased because it is possible that y doesn't actually
suffer from bias, but is simply being debiased due to it's nature as the vector
representative of a demographic identity term.
The weights w are trained via a pair of competing objectives. These objectives
abide by two key considerations:
    1. The debiasing process should not lead to distortion in the semantic
       essence of the word vector - this would invalidate the entire endeavour
       and would indeed even breed alternative biases from the now obscured
       semantics in the word vectors.
    2. To be free from semantic bias, the word vector should reside as far
       from both of the positive and negative subspaces - that is, it should be
       pulled/pushed as much as it possibly can towards the equilibirium
       between these subspaces. Note that if the debiasing methodology focused
       merely on pulling demographic word vectors out of the negative subspace,
       disregarding the ones residing in the positive subspace, then this does
       little to mitigate bias. The idea here is that demographic identity
       terms should be treated as neutral by downstream machine learning
       programs.

Regarding the former point, devising a loss function that encodes word vector
distortion is in itself a challenge due to the lack of interpretability in the
vector space of the word embeddings. As an alternative, the article presents a
a simpler loss function that serves to minimise the mean-squared distance
between the debiased word vector for a word and its original, non-debiased one.
This loss function is formally defined as:
    - L_p = (y - y_hat)^2
where, as before, y is the original word vector and y_hat is the vector with
debiasing applied.

Regarding the latter point from above, this consideration underlies the
adversarial objective. The method presents an adversary, and it aims to
minimise the ability that this adversary has of predicting the sentiment
polarity of the word vector being debiased. The mechanism for defining the
sentiment polarity of a word was outlined in the section prior: that is, the
result of the projection of a word vector onto the directional sentiment vector
(the construction of which is the purpose of the previous section of code). By
the terminology of the article, I shall denote this directional sentiment
vector as k. Moreover, let the sentiment polarity of a word be z, and as before
the input word vector will be y. Therefore, the adversary will attempt to
predict z from y as follows:
    - z = k.t * y
The adversary prediction problem, then, is:
    - z_hat = w_a^T * y_hat
where w_a are the adversarial weights. The adversarial weights are learned via
mean squared distance loss, which is our adversarial loss function:
    - L_a = (z - z_hat)^2

To obtain the overall loss function, L_p and L_a are combined, and this
combination dictates the gradient updates for the weights.
"""

# ---- The adversarial debiasing model!
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Parameter as TorchParameter
from torch.nn.init import xavier_uniform_, normal_
from torch.nn.functional import mse_loss, linear
torch.manual_seed(0)


def normalise_tensor(vec: torch.tensor
                     ) -> torch.tensor:
    """
    Function to return a normalised version of an inputted pytorch tensor
    """
    return vec/(torch.norm(vec) + np.finfo(np.float32).tiny)


def classifier_prediction(y: torch.tensor,
                          w: torch.tensor
                          ) -> torch.tensor:
    """
    Can be thought of as the "semantic predictor", which aims to map the input
    word vector to a word vector similar to said input.
    Defined in the literature by:
        y_hat = y - ww^Ty
    which the pytoch linear method below imitates.
    """
    return linear(y, w @ w.transpose(0, 1))


def adversary_prediction(y_hat: torch.tensor,
                         w_a: torch.tensor 
                         ) -> float:
    """
    Adversary's prediction method, which aims to predict the sentiment of the
    input (modified) word vector.
    Defined in the literature by:
        z^hat = (w_a^T)(y^hat)
    """
    return linear(y_hat, w_a.transpose(0, 1))


def do_sgd(word_vectors: List[np.ndarray],
           dsv_scores: List[float],
           dsv: np.ndarray,
           opt_p: Adam, opt_a: Adam,
           w_p: torch.Tensor, w_a: torch.Tensor,
           loss_log_p: List[float], loss_log_a: List[float],
           uber_w_p: torch.Tensor, uber_w_a: torch.Tensor, uber_score: float,
           adversary_momentum: float,
           n: int, epoch_num: int, iter_num: int,
           verbose: bool
           ) -> tuple:
    """
    Function for training the adversarial learning model via stochastic
    gradient descent. Entails updating the classifer/predictor's and the
    adversary's weights seperately, as well as updating the classifier's
    weights distinctly to mitigate bias. The models aims to maximise the
    classifier's ability to translate the word vector into one similar to the
    original word vector (to prevent semantic distortion), whilst minimising
    the adversary's ability to predict the sentiment of the vector.
    """

    # Create features and labels, where the labels are merely a copy of the
    # features (for the similarity aspect of the classifer)

    # The batch features (the same as the input words - will also be used as
    # the batch labels for the classifer loss function)
    ys = torch.tensor(word_vectors)

    # The protected attributes, i.e. the sentiment for each word vector. To be
    # used for the adversarial loss function.
    zs = torch.tensor(dsv_scores)

    # Predict classifier and obtain predictor loss
    y_hats = classifier_prediction(ys, w_p)
    loss_p = mse_loss(y_hats, ys)
    """
    [NOTE ON PREDICTOR LOSS FUNCTION]
    Denoted L_p in the literature; this loss function serves to ensure that the
    meaning represented by the word vector is not distorted to anything near an
    unusuable extent. Specifically, it aims to minimise the mean squared
    distance between the input (i.e., original) word vector and its "debiased"
    counterpart (which is engineered by the model)
    """

    # Adversarially predict and obtain adversary loss
    # Note: must use .squeeze() to remove all size 1 vectors
    z_hats = adversary_prediction(y_hats, w_a).squeeze()
    loss_a = mse_loss(zs, z_hats)
    """
    [NOTE ON ADVERSARY LOSS FUNCTION]
    Denoted L_a in the literature; this loss function describes the ability of
    an adversary in predicting the sentiment polarity of the word vector being
    debiased. Using the directional dentiment vector defined earlier, the
    actual sentiment polarity of a word vector is defined by projecting the
    vector being debiased onto this.
    """

    # Compute adversary gradients; obtain adversary gradient with respect to
    # the classifer gradient
    loss_a.backward(retain_graph=True)
    grad_a = w_p.grad.clone()

    # Learn the classifer and adversary, directly utilising the update formula
    # from the article
    loss_p.backward()

    # -------- #
    # The objective function - found in page 361 of FAT* ’20
    unit_grad_a = normalise_tensor(grad_a)
    # Middle projection term, for ensuring loss_p doesn't help the adversary
    w_p.grad += torch.sum(w_p.grad*unit_grad_a)
    # Adversary momemtum handles trade-off b/t semantic distortion w/ debiasing
    w_p.grad -= adversary_momentum*w_a.grad
    # -------- #

    opt_p.step(), opt_a.step()

    # Reset the gradients for the optimisers
    opt_a.zero_grad(), opt_p.zero_grad()

    # Append losses to their respective logs
    loss_log_p.append(loss_p.item())
    loss_log_a.append(loss_a.item())

    # Print some metrics for the current iteration
    if verbose and iter_num % 10 == 0:
        print(f"epoch {epoch_num}\n"
              f"iter: {iter_num}\n"
              f"batch classifier loss:  {loss_p.item()}\n"
              f"batch adversarial loss: {loss_a.item()}")

    return (
        (uber_w_p, uber_w_a, uber_score),
        (w_p, w_a, opt_p, opt_a, loss_log_p, loss_log_a)
    )


def init_adversarial_debiasing(embeddings: List[np.ndarray],
                               directional_sentiment_vector: np.ndarray,
                               embedding_depth: int,
                               num_epochs: int,
                               batch_size: int,
                               adversary_momentum: float,  # Cor. w/ alpha
                               learn_rate_p: float,
                               learn_rate_a: float,
                               lr_decay: float,
                               verbose: bool = True
                               ) -> None:
    """
    This function commences and contains the adversarial debiasing model. The
    model abides by the following procedure:
        1. Initialise the labels and protected attributes by converting their
           respective numpy arrays into pytorch tensors, so as to make them
           compatable with the model.
        2. Initialise the other essential elements of the model, e.g. the
           weights (for both the classifer and the adversary), the optimisers,
           and the schedules. All of these are duely commented in the following
           code.
        3. For each iteration, establish a list of batch indices corresponding
           with the batch size for each round of Stochastic Gradient Descent,
           which is how the model will learn.
        4. Do stocahstic gradient descent (seperate function) for the specified
           number of iterations (aka "epochs"), wherein the model will learn.
           The central update formula for this adversarial learning model can
           be found on page 3 of the article (or, rather, page 361 of FAT20)
        5. When trained, the model will return the weight which can be used to
           translate the word2vec embeddings to their debiased equivalent. This
           will be the best weight the model found from any iteration.
    """

    # Convert labels (word vectors) and protected (sentiment scores) to numpy
    # arrays to speed up certain processes
    ys = embeddings
    zs = np.array([normalise_embedding(label).dot(directional_sentiment_vector)
                   for label in ys])

    dsv_tens = torch.tensor(directional_sentiment_vector)
    a_vector = torch.tensor(embeddings[0])

    # Get the number of training examples and ensure batch size does not exceed
    n = len(ys)
    if batch_size > n:
        batch_size = n

    # Classifier (aka predictor) weights
    w_p = TorchParameter(
        xavier_uniform_(torch.Tensor(embedding_depth, 1))
    )

    # Adversary weights
    # Initialise as extremely small so as to ensure the classifier does not 
    # overfit against a particular sub-optimal adversary.
    w_a = TorchParameter(
        normal_(torch.Tensor(embedding_depth, 1), mean=1e-5, std=1e-4)
    )

    # Establish initial best parameters
    uber_score = 1
    uber_w_p, uber_w_a = w_p.clone(), w_a.clone()

    # Maintain two lists, for each of the respective losses
    loss_log_p, loss_log_a = [], []

    # Initialise optimisers (i.e. Adam optimisers, as most suitable)
    opt_p = Adam([w_p], lr=learn_rate_p)
    opt_a = Adam([w_a], lr=learn_rate_a)

    # Initialise schedulers (to adjust learning rate based on epochs)
    sched_p = ExponentialLR(opt_a, gamma=lr_decay)
    sched_a = ExponentialLR(opt_p, gamma=lr_decay)

    # commence training
    start_time = time.time()
    for epoch in range(num_epochs):

        if verbose:
            print(f"Epoch number {epoch+1}")
            t = time.time()

        # Shufle the indices (i.e., for mini-batch gd)
        batch_indices = np.array_split(
            np.random.choice(n, n).astype(int), n//batch_size
        )

        # Iterate through the batches and perform stochastic gradient descent
        for iteration, batch in enumerate(batch_indices):
            this_batch = np.take(ys, batch, axis=0)
            this_batch_protect = np.take(zs, batch, axis=0)
            metrics, items = do_sgd(
                this_batch, this_batch_protect,
                directional_sentiment_vector,
                opt_p, opt_a,
                w_p, w_a,
                loss_log_p, loss_log_a,
                uber_w_p, uber_w_a, uber_score,
                adversary_momentum,
                n, epoch, iteration,
                verbose,
            )
            uber_w_p, uber_w_a, uber_score = metrics
            w_p, w_a, opt_p, opt_a, loss_log_p, loss_log_a = items

            if a_vector is not None:
                pred = classifier_prediction(a_vector, w_p)
                sentiment = torch.dot(
                    dsv_tens,
                    normalise_tensor(pred)
                )

                if abs(sentiment) < uber_score and epoch > 10:
                    uber_score = abs(sentiment)
                    uber_w_p = w_p.clone()

        if verbose and epoch % 100 == 0 and a_vector is not None:
            print()
            print(f"Epoch: {epoch}")
            print(f"Sentiment projection: {sentiment}.")
            print(f"Difference:           {torch.mean(a_vector - pred)}")
            print(f"Loss_p:               {loss_log_p[-1]}.")
            print(f"Loss_a:               {loss_log_a[-1]}")

        sched_p.step(), sched_a.step()
        if verbose:
            print(f"Epoch time taken: {time.time() - t:.3f} seconds.")

    print(f"Algorithm time taken: {time.time() - start_time:.3f} seconds.")

    # Return weights, now suitable for debiasing word vectors
    return uber_w_p if a_vector is not None else w_p


def get_demographic_identity_terms(embeddings: Word2Vec,
                                   terms: List[str] = TERMS
                                   ) -> Set[str]:
    """
    Helper function for obtaining a set of unique demographic identity terms
    that exist in the word2vec embeddings. The word2vec embeddings contain both
    capitalised and non-capitalised words, so it's best to obtain both.
    """
    return set(
        [w for w in terms if w in embeddings] +
        [w.lower() for w in terms if w.lower() in embeddings]
    )


def apply_debiasing(word_vector: List[np.array],
                    weight: torch.Tensor
                    ) -> List[np.array]:
    """
    Use the trained classifier weights to obtain a debiased word vector.
    """
    word_tensor = torch.tensor(word_vector)
    db_tensor = classifier_prediction(word_tensor, weight)
    db_array = db_tensor.detach().numpy()
    return db_array


def debias_selected_embeddings(embeddings: Word2Vec,
                               trained_weight: torch.Tensor,
                               to_debias: Set[str]
                               ) -> None:
    """
    For a given list of "sensitive" words (i.e., demographic identity terms),
    the word vectors for them present in the embeddings will be replaced by
    their debiased equivalent, found via the adversarial learning model.
    """

    # Obtain array of the word vectors to be debiased, storing the indices in
    # a dictionary for easy retrieval
    one_hot = {}
    i = 0
    to_debias_arr = []
    for word in to_debias:
        if word in embeddings:
            to_debias_arr.append(embeddings[word])
            one_hot[word] = i
            i += 1
    debiased_arr = apply_debiasing(
        np.array(to_debias_arr), trained_weight
    )

    # Replace the demographic word vectors in the original word embeddings with
    # the new, debiased ones
    for word in one_hot:
        embeddings[word] = debiased_arr[one_hot[word]]

    return embeddings


def analyse_demographic_sentiments(embeddings: Word2Vec,
                                   dsv: np.array,
                                   identity_terms: Set[str],
                                   debiased: bool
                                   ) -> pd.DataFrame:
    """
    Function akin to the one from earlier, 'sanity_check_dsv()', except instead
    of analysising the sentiment of all words in the embeddings, only analyse
    the sentiment of the selected demographic identity terms. This is provides
    a convenient way of briefly confirming the success of the debiasing.
    """
    s = "prior to debiasing" if not debiased else "post debiasing"
    df = pd.DataFrame(
        [{"word": word,
          "sentiment": normalise_embedding(embeddings[word]).dot(dsv)}
         for word in identity_terms if word in embeddings]
    )
    df.sort_values(by="sentiment", inplace=True)
    print(f"\nMost negative words, according to the DSV ({s}):")
    print(df.head(3))
    print(f"\nMost positive words, according to the DSV ({s}):")
    print(df.tail(3))
    print()
    return df


def stripplot_compare(pre: pd.DataFrame,
                      post: pd.DataFrame
                      ) -> None:
    """
    Stripplot to compare the sentiment projections of demographic word vectors
    prior to and following the adversarial debiasing.
    """
    pre['status'] = ['Biased' for _ in range(len(pre))]
    post['status'] = ['Debiased' for _ in range(len(post))]
    together = pd.concat([pre, post])
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=together, x="sentiment", y="status", orient='h')
    plt.show()


def take_specific(embeddings: Word2Vec,
                  terms: List[str]
                  ) -> List[np.ndarray]:
    """
    Helper function for returning some specific word embeddings
    """
    return np.array([embeddings[word] for word in terms if word in embeddings])


def debias_word_vectors(dsv: np.ndarray,
                        params: dict,
                        num_embeddings: int,
                        verbose: bool,
                        display_strip: bool = False
                        ) -> Word2Vec:
    """
    The central method for debiaising the word embeddings. First, the word2vec
    embeddings are collected as usual, the number of which is specified by an
    adjustable parameter (generally, 100,000 is used for the sake of memory),
    then a "directional sentiment vector" is obtained which allows us to
    project word vectors onto a sentiment subspace, which unveils the
    underlying sentiment of this word vector. After this is obtained, the
    adversarial debiasing model is applied to the embeddings in order to
    neutralise the sentiment of demographc identity terms, which may
    potentially be biased (by having an underlying positive or negative
    sentiment).
    """

    print("Debiasing the word vectors.")

    print("Collecting word2vec embeddings...")
    w2v_embeddings = obtain_w2v_model(num_embeddings)
    print("Embeddings retrieved!")

    # Get the demographic identity terms and an numpy array of the word vectors
    # corresponding with them
    demographic_identity_terms = get_demographic_identity_terms(w2v_embeddings)
    to_debias = take_specific(
        w2v_embeddings, demographic_identity_terms
    )

    print(f'Number of word vectors being debiased: {len(to_debias)}')

    # The adversarial debiasing, at last!
    w_trained = init_adversarial_debiasing(
        to_debias, dsv, verbose=verbose,
        embedding_depth=params['embedding_depth'],
        num_epochs=params['num_epochs'],
        batch_size=params['batch_size'],
        adversary_momentum=params['adversary_momentum'],
        learn_rate_p=params['learn_rate_p'],
        learn_rate_a=params['learn_rate_a'],
        lr_decay=params['lr_decay'],
    )

    # For the sake of comparison, unveil the underlying sentiments of the
    # specified demographic term word vectors before and after debiasing to
    # first uncover bias and secondly verify that the debiasing worked
    pre_db = analyse_demographic_sentiments(
        w2v_embeddings, dsv, demographic_identity_terms, False
    )

    # Configure the word embeddings by replacing the potentially biased
    # demographic word embeddings with their now debiased counterparts
    debiased_embeddings = debias_selected_embeddings(
        w2v_embeddings, w_trained, demographic_identity_terms
    )

    # Analyse the demographic sentiments post-debiasing
    post_db = analyse_demographic_sentiments(
        debiased_embeddings, dsv, demographic_identity_terms, True
    )

    # Visualise the sentiment distributions before and after debiasing for the
    # demographic identity terms
    if display_strip:
        stripplot_compare(pre_db, post_db)

    return debiased_embeddings


def bias_mitigating_approach(data: pd.DataFrame,
                             dsv: np.ndarray,
                             params: dict,
                             num_embeddings,
                             regression: bool = False,
                             binary: bool = True,
                             test_model: bool = False,
                             verbose: bool = False
                             ) -> Pipeline:
    """
    Container function for the bias mitigating machine learning approach.
    """

    print("Commencing data pre-processing...")
    data = handle_sentiments(
        data, make_classification=(not regression), make_binary=binary
    )
    data = tidy_data(data)
    print("Pre-processing finished!")

    print("Obtaining and debiasing word2vec embeddings...\n" + "-"*LINE_LENGTH)
    debiased_embeddings = debias_word_vectors(
        dsv, params, num_embeddings, verbose=verbose
    )
    print("-"*LINE_LENGTH + "\nEmbeddings obtained and debiasing complete!")

    print("Commencing machine learning...")
    train_data, test_data = establish_training_data(data, debiased_embeddings)

    model = support_vector_classification(
        train_data, debiased_embeddings, False
    )
    print("Support Vector classifier learned!")

    print("Evaluating model...")
    evaluate(model, test_data)

    if test_model:
        if binary and not regression:
            test_inputs(model, debiased_embeddings, "two class")
        elif not regression:
            test_inputs(model, debiased_embeddings, "five class")
        else:
            test_inputs(model, debiased_embeddings, "regression")

    return debiased_embeddings, model, test_data


# ------- Compare biased and debiased models

def compare_models(model_stats: List[pd.DataFrame],
                   eecs: List[pd.DataFrame],
                   verbose: bool = False
                   ) -> None:
    """
    Function for comparing the statistics of each model (i.e., the original,
    'conventional' model, the model after subsampling the dataset, and the
    model after debiasing the word embeddings)
    """

    to_df_var = []
    for stats, status in model_stats:
        to_df_var += [{
            "Emotion": emotion,
            "Status": status,
            "Variance": np.var(stats[stats["Emotion"] == emotion]["Average"])
        } for emotion in ("all", "anger", "sadness", "fear", "joy")]
    var_df = pd.DataFrame(to_df_var)

    # Barplot of variances
    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=var_df, x='Emotion', y='Variance', hue='Status'
    )
    plt.show()


from sklearn.metrics import roc_curve, roc_auc_score
def compare_roc(models: Tuple[Tuple[Pipeline, pd.DataFrame, str], ...],
                ) -> None:
    """
    Compare the performance of the models via ROC curves for each, on the same
    canvas.
    """

    # Initialise canvas
    plt.figure(figsize=(6, 5))
    sns.lineplot(
        x=[0, 1], y=[0, 1], label="Unskilled Classifier"
    ).lines[0]

    # Assign titles and labels
    plt.title("ROC Curves for the three implementations.")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Generate and draw the curves
    for model, test_data, name in models:
        x, y = test_data
        predictions = model.predict_proba(x)
        score = roc_auc_score(y, predictions[:, 1])
        false_pos_rate, true_pos_rate, thresholds = roc_curve(
            y, predictions[:, 1]
        )
        sns.lineplot(
            x=false_pos_rate, y=true_pos_rate,
            label=f"{name}, AUC = {score:.4f}"
        )
    plt.legend(loc="lower right")
    plt.show()


# --------------------------------------------------------------------------- #
# ---------------------------------- MAIN ----------------------------------- #

def main(num_embeddings: int = 400_000,
         initial_analysis: bool = INITIAL_ANALYSIS,
         init_data: bool = INITIALISE_DATA,
         use_app: bool = USE_APP,
         verbose: bool = False
         ) -> None:
    """
    Main Function
    """
    start_time = time.time()

    # Configure the parameters for the adversarial debiasing model
    debiasing_params = {
        'embedding_depth': 300,
        'num_epochs': 2000,
        'batch_size': 32,
        'adversary_momentum': 0.5,
        'learn_rate_p': 0.05,
        'learn_rate_a': 0.1,
        'lr_decay': .9,
    }

    # After much tweaking and testing, the following parameters were found to
    # be the most successful; they neutralise the underlying sentiments of the
    # demographic embeddings to the range [-0.05, 0.05].
    """
        'embedding_depth': 300,
        'num_epochs': 2000,
        'batch_size': 32,
        'adversary_momentum': 0.5,
        'learn_rate_p': 0.05,
        'learn_rate_a': 0.1,
        'lr_decay': .9,
    """


    # Collect and initialise data (by default, this is off because the data 
    # used has been provided with the submission)
    if init_data:
        print("Initialising data...")
        initialise_datasets()
        print("Data initialised!")


    # Read in the stanford sentiment treebank dataset locally
    print("Reading in Stanford Sentiment Treebank Dataset...")
    data = read_in_datasets(["stanford_dataset.csv"])[0]
    print("Datasets obtained!")

    
    # Undertake initial data analysis
    if initial_analysis:
        print("\nCommencing data analysis...\n"+'-'*LINE_LENGTH+'\n')
        initial_data_analysis(data.copy())
        print('-'*LINE_LENGTH+"\nAnalysis finished.")


    # Obtain the word embeddings
    print("Collecting word2vec embeddings...")
    w2v_embeddings = obtain_w2v_model(num_embeddings)
    print("Embeddings retrieved!")


    # Commence the conventional, biased machine learning implementation
    print("\nCommencing conventional machine learning sentiment analysis, "
          "sans bias mitigation.\n"+'-'*LINE_LENGTH+"\n")
    biased_model, biased_test_data = conventional_implementation(
        data.copy(), w2v_embeddings, test_model=USE_APP
    )
    print("Conventional machine learning model trained!\n")

    print("Commencing bias analysis for the biased model just trained...")
    eec_conventional, stats_conventional = model_bias_analysis(
        biased_model, w2v_embeddings, 'Biased'
    )
    print("Bias analysis finished.\n")


    # Commence unbiased subsample implementation
    print("Commencing unbiased subsample implementation...\n"
          + "-"*LINE_LENGTH)
    print("Obtaining unbiased subsample of dataset...")
    subsample = debiased_subsample(data.copy())
    print("Unbiased subsample obtained!")
    print("Obtaining sentiment analysis model with the unbiased subsample...")
    subsample_model, subsample_test_data = conventional_implementation(
        subsample, w2v_embeddings, test_model=USE_APP, encode_sentiments=False
    )
    print("Model obtained.")
    print("Commencing bias analysis for the subsample model just trained...")
    eec_subsample, stats_subsample = model_bias_analysis(
        subsample_model, w2v_embeddings, 'Unbiased subsample'
    )
    print("Bias analysis finished.")


    # Debiased approach
    print("Obtaining directional sentiment vector...")
    dsv = obtain_directional_sentiment_vector(w2v_embeddings)
    print("Directional sentiment vector obtained!")
    print("\nCommencing advanced machine learning sentiment analysis, with "
          "bias mitigation.\n" + "-"*LINE_LENGTH)
    debiased_embeddings, debiased_model, debiased_test_data = (
        bias_mitigating_approach(
            data.copy(), dsv, debiasing_params,
            num_embeddings, test_model=USE_APP,
        )
    )
    print("Debiased model trained.")
    print("Commencing bias analysis for the model "
          "using the debiased embeddings...")
    eec_debiased, stats_debiased = model_bias_analysis(
        debiased_model, debiased_embeddings, 'Debiased'
    )
    print("Bias analysis finished.")


    # Compare the outcomes of the three approaches
    compare_models(
        [(stats_conventional, 'Biased'),
         (stats_subsample, 'Unbiased subsample'),
         (stats_debiased, 'Debiased')],
        [(eec_conventional, 'Biased'),
         (eec_subsample, 'Unbiased subsample'),
         (eec_debiased, 'Debiased')],
    )
    compare_roc(
        (
            (biased_model, biased_test_data, "Biased model"),
            (subsample_model, subsample_test_data, "Unbiased subsample"),
            (debiased_model, debiased_test_data, "Debiased model")
        )
    )

    print("\nProgram finished! Thank you.")
    print(f"Total elapsed time: {(time.time() - start_time):.3f} seconds.")


"""
CONCLUDING REMARKS

Although the model has been shown to effectively eradicate underlying sentiment
bias contained within word vectors, whilst simultaneously maintaining a
reasonable degree of semantic similarity with the original embeddings, the
model itself does have several limitations. Perhaps the most significant is
that the potentially biased vocabulary must be specified by the developer, as
the model doesn't detect them independently. The issue here is derived from the
fact that an exhaustive list of demographic terminology is exceedingly
difficult to attain, especially due to language's ever-shifting nature.
Furthermore, this dependence may facilitate bias, intentional or not, on the
developer's end. Additionally, more nuanced aspects of human language such as
slang are not accounted for. That said, it is reasonably simple to implement
the above model into any standard sentiment analysis application, so it is an
incredibly effective way of mitigating sentiment bias in such applications.
"""
if __name__ == '__main__':
    main()
