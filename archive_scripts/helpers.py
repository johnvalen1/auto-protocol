import pandas as pd
import class_labels
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import archive_scripts.filepaths as filepaths
import csv
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

def readData(filepath):
    columns_needed = ["Identifier", "indication", "protocol", "which_protocol", "patient_sex", "patient_age", "original_or_augmented"]
    df = pd.read_excel(filepath)
    return pd.DataFrame(df[columns_needed])


def preprocessTextField(df, col_name):
    df = df.copy()
    def remove_punctuation():
        df[col_name] = df[col_name].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))
        return df

    # Function to remove stopwords
    def remove_stopwords():
        stop_words = set(stopwords.words('english'))

        with open(f"{filepaths.preprocess_filepath}stop_words.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for item in stop_words:
                writer.writerow([item])  # each item goes into a new row

        for ix, indication in df[col_name].items():
            word_tokens = word_tokenize(indication)
            filtered = ' '.join([word for word in word_tokens if word.lower() not in stop_words])
            df.loc[ix, col_name] = filtered
        return df

    def lemmatize_text():
        # Function to lemmatize text
        lemmatizer = WordNetLemmatizer()
        for ix, indication in df[col_name].items():
            word_tokens = word_tokenize(indication)
            filtered = ' '.join([lemmatizer.lemmatize(word) for word in word_tokens])
            df.loc[ix, col_name] = filtered
        return df

    df = remove_punctuation()
    df = remove_stopwords()
    df = lemmatize_text()
    # output Excel file containing pre-processed indication so we can see what the model ingests
    # make sure that things like # stay as fracture in the dataset
    return df


def DefineLabelVectorization(label_array):
    """ This function turns the text-based, comma-separated labels into vector representations. """
    vectorized_label_strings = []
    for label_string in label_array:
        # create a vector of 0's that has length equal to the number of classes; indices range from 0 to len(labels) - 1
        vectorized_label = [0 for i in class_labels.labels]
        print(label_string.lower())
        for single_label in label_string.lower().split(","):
            # get the index of the item relative to the list of classes
            class_index = class_labels.labels.index(single_label.strip()) # call strip() on the label to remove any trailing whitespace which can cause logical check to fail
            vectorized_label[class_index] = 1
        vectorized_label_strings.append(vectorized_label)
    return vectorized_label_strings