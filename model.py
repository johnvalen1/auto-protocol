""" This script houses the ML model used for the supervised classification task of learning the protocol from the indication."""

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import class_labels
import hyperparameters
import helpers
import filepaths
import re
# the base model is a simple logistic regression, which we expand to allow for multi-label outputs
def LogisticModel(corpus, file_output_path):
    # Convert protocol strings to lists
    df = corpus.copy()
    df['protocol'] = df['protocol'].apply(lambda x: [p.strip() for p in x.split(',')])

    # (Optional) Preserve order from raw data:
    unique_protocols = []
    for protocols in df['protocol']:
        for p in protocols:
            if p not in unique_protocols:
                unique_protocols.append(p)

    # Initialize MultiLabelBinarizer with your chosen order (or omit "classes" to have sorted order)
    mlb = MultiLabelBinarizer(classes=unique_protocols)
    Y = mlb.fit_transform(df['protocol'])

    # Vectorize the free-text 'indication'
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['indication'])

    # Split the data ensuring that you keep track of the corresponding rows in the original dataframe.
    X_train, X_test, Y_train, Y_test, df_train, df_test = train_test_split(X, Y, df, test_size=0.1, random_state=42)

    df_train.to_csv(f"{filepaths.train_dir}df_train.csv")

    # Set up the OneVsRestClassifier with LogisticRegression
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, Y_train)

    # Define a function that checks for a positive, non-zero probability and returns a string of all such labels.
    def get_labels_with_positive_prob(probabilities, classes):
        # Get all labels where the probability is greater than zero
        positive_labels = [label for label, prob in zip(classes, probabilities) if prob > hyperparameters.prob_cutoff]
        # Concatenate them into a comma-separated string
        return ", ".join(positive_labels)

    # Obtain prediction probabilities for the test set
    probs = clf.predict_proba(X_test)

    # For each test instance, get the concatenated string of predicted labels with non-zero probability
    predicted_labels_str = [get_labels_with_positive_prob(prob, mlb.classes_) for prob in probs]

    # Add these predicted labels as a new column in the test dataframe
    df_test = df_test.reset_index(drop=True)
    df_test['predicted_protocol'] = predicted_labels_str
    # remove characters added to the protocol column - characters like [ ] and '
    df_test['protocol'] = df_test['protocol'].apply(lambda x: re.sub('\'|\[|\]', '', str(x)))

    #[re.sub('\'|\[|\]', '', protocol) for protocol in df_test['protocol']]
    df_test.to_csv(f"{filepaths.test_dir}df_test_with_predictions.csv")

    # calculate model predictive accuracy
    import model_metrics
    print(model_metrics.model_accuracy(df_test, true_label_column="protocol", prediction_column="predicted_protocol"))


