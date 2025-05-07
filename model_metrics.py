import pandas as pd
import re
import filepaths as f
import csv

def model_accuracy(df, true_label_column, prediction_column):
    """ This function returns the overall accuracy of the model's predictions.
    An accurate prediction is defined as the model predicting at least one of the protocols for the instance.
    For example, if the true protocol is 'Head NC, C Spine' and the model predicts 'Head NC', this is a correct prediction.
    Another example is if the model predicted 'Head NC, Head C+' for this same instance, because 'Head NC' is one of the assigned protocols. """
    n_rows = len(df)
    num_correct = 0

    def list_contains_duplicate_elements(x):
        if len(x) != len(set(x)):
            return True
        else:
            return False

    for ix, protocols in enumerate(df[true_label_column]):
        # combine the true protocols with the predicted protocols into a single list, and if there are any duplicates in the result, the model was correct
        protocols_as_list = [protocol for protocol in protocols.split(", ")] #split the string of potentially multiple protocols and put into a list
        predicted_protocols = df[prediction_column][ix] #get the corresponding row's predicted protocol
        predicted_protocols_as_list = [pred_protocol for pred_protocol in predicted_protocols.split(", ")]

        combined_list = protocols_as_list + predicted_protocols_as_list

        if list_contains_duplicate_elements(combined_list):
            # the model's prediction is correct if the combined list contains any duplicates
            num_correct += 1
        continue

    return round(num_correct/n_rows, 3)

def exact_model_accuracy(df, true_label_column, prediction_column):
    """ This function returns the exact accuracy of the model's predictions.
    An exactly accurate prediction is defined as the model predicting the exact as the ground truth protocol(s). """
    n_rows = len(df)
    num_correct = 0

    for ix, _ in enumerate(df[true_label_column]):

        if df[true_label_column][ix] == df[prediction_column][ix]:
            num_correct += 1

    return round(num_correct/n_rows, 3)

def class_accuracies(protocol_labels, df, filepath):
    class_accuracies = {}
    class_counts = {}
    # starting with the test set
    for protocol in protocol_labels:
        # protocol from list is lowercase, so need lowercase to query the pandas dataframe
        filtered_df = df[df['protocol'].str.contains(protocol, case=False, na=False)]
        num_rows = len(filtered_df)
        num_correct = 0
        for predicted_protocols in filtered_df["predicted_protocol"].str.lower():
            if protocol in predicted_protocols:
                num_correct += 1
            else:
                continue
        if num_rows == 0:
            class_accuracy = 0
        else:
            class_accuracy = float(np.round(num_correct/num_rows, 4))
        class_accuracies[protocol] = class_accuracy
        class_counts[protocol] = num_rows

    class_accuracies_test_df = pd.DataFrame({
    'class_accuracy': class_accuracies,
    'class_count': class_counts
    })
    class_accuracies_test_df=class_accuracies_test_df.reset_index().rename(columns={'index': 'protocol'})
    class_accuracies_test_df.to_csv("./doc/" + filepath)