import pandas as pd
import re


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

def model_sensitivity():
    pass