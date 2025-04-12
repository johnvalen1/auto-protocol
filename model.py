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

# the base model is a simple logistic regression
# def LogisticModel(train_corpus, test_corpus, file_output_path):

#     X_train = train_corpus['indication']
#     y_train = train_corpus['protocol']

#     X_test = test_corpus['indication']
#     y_test = test_corpus['protocol']

#     # create a multi label binarizer to allow the model to assign multiple labels to a single indication
#     mlb = MultiLabelBinarizer()
#     y_train = mlb.fit_transform(y_train)
#     y_test = mlb.fit_transform(y_test)
#     # Convert text data to numerical features using TF-IDF
#     vectorizer = TfidfVectorizer(stop_words='english')  # You can adjust this as needed
#     X_train_tfidf = vectorizer.fit_transform(X_train)
#     X_test_tfidf = vectorizer.transform(X_test)

#     # Train a classifier (Logistic Regression in this case)
#     model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
#     model.fit(X_train_tfidf, y_train)

#     # Make predictions
#     Y_pred = model.predict(X_test_tfidf)
#     # (preferred for ranking): get prediction probabilities.
#     probs = model.predict_proba(X_test_tfidf)

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
    X_train, X_test, Y_train, Y_test, df_train, df_test = train_test_split(
        X, Y, df, test_size=0.1, random_state=42)

    df_train.to_csv(f"{filepaths.train_dir}df_train.csv")
    # Set up the OneVsRestClassifier with LogisticRegression
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train, Y_train)

    # Predict the labels for the test set
    Y_pred = clf.predict(X_test)

    # Convert the binary prediction array back into label names
    predicted_labels = mlb.inverse_transform(Y_pred)

    # Optionally convert each tuple of predicted labels to a comma separated string:
    predicted_labels_str = [", ".join(labels) if labels else "" for labels in predicted_labels]

    # Add the predicted labels as a new column in the test dataframe for comparison
    df_test = df_test.reset_index(drop=True)   # Reset index for a clean join (optional)
    df_test['predicted_protocol'] = predicted_labels_str

    df_test.to_csv(f"{filepaths.test_dir}df_test_with_predictions.csv")