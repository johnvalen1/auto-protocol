""" This script houses the ML model used for the supervised classification task of learning the protocol from the indication."""

# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


 # the base model is a simple logistic regression
def LogisticModel(indication, protocol, file_output_path):
    # create a multi label binarizer to allow the model to assign multiple labels to a single indication
    mlb = MultiLabelBinarizer()
    protocol_mulit_label = mlb.fit_transform(protocol)

    X_train, X_test, y_train, y_test = train_test_split(indication, protocol_mulit_label, test_size=0.2, random_state=42)

    # Step 1: Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')  # You can adjust this as needed
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 2: Train a classifier (Logistic Regression in this case)
    model = OneVsRestClassifier(LogisticRegression())
    model.fit(X_train_tfidf, y_train)

    # Step 3: Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Step 4: Evaluate the model
    output_text = "Accuracy: " + str(accuracy_score(y_test, y_pred)) + "\n Classification Report: \n" + str(classification_report(y_test, y_pred))
    with open(file_output_path, "w") as file:
        file.write(output_text)

    # Confusion Matrix (for multi-label classification)
    conf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print("\nConfusion Matrix (Multi-class version):")
    print(conf_matrix)

    # print("\nVerbose Class-wise Metrics:")

    # for idx, label in enumerate(mlb.classes_):
    #     print(label)
    #     precision = classification_report(y_test[:, idx], y_pred[:, idx], output_dict=True)[label]['precision']
    #     recall = classification_report(y_test[:, idx], y_pred[:, idx], output_dict=True)[label]['recall']
    #     f1_score = classification_report(y_test[:, idx], y_pred[:, idx], output_dict=True)[label]['f1-score']
    #     support = classification_report(y_test[:, idx], y_pred[:, idx], output_dict=True)[label]['support']

    #     print(f"Class: {label}")
    #     print(f"  Precision: {precision:.4f}")
    #     print(f"  Recall: {recall:.4f}")
    #     print(f"  F1-Score: {f1_score:.4f}")
    #     print(f"  Support: {support}")
    #     print("-" * 40)

    # Step 5: (Optional) Save the model and vectorizer
    # import joblib
    # joblib.dump(model, 'text_classifier_model.pkl')
    # joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')



