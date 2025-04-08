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

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

def readData(filepath):
    """ This function reads in the .csv file with the indication, protocol, and identifier. """
    corpus = pd.read_excel(filepath)
    columns_needed = ["Identifier", "indication", "protocol", "which_protocol"]
    return pd.DataFrame(corpus[columns_needed])


# Function to remove punctuation
def preprocessIndication(indication_df):
    def remove_punctuation(text):
        return ''.join([char for char in text if char not in string.punctuation])

    # Function to remove stopwords
    def remove_stopwords(text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        return ' '.join([word for word in word_tokens if word not in stop_words])

    def lemmatize_text(text):
        # Function to lemmatize text
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        return ' '.join([lemmatizer.lemmatize(word) for word in word_tokens])

    return pd.Series([
    remove_punctuation(
        remove_stopwords(
            lemmatize_text(str(row_indication))))
    for row_indication in indication_df])


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