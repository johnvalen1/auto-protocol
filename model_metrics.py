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

def class_accuracies(df, true_label_column, prediction_column, augmentation = False, write_output = True):
    """ This function calculates the class-specific accuracies, returning a dictionary and writing to csv if set to true.
    Each class's accuracy is defined as the number of cases where the protocol(s) are contained in the predicted protocol(s) divided by the class's prevalence in the dataset. """

    class_specific_accuracies = {}
    class_specific_exact_accuracies = {}

    for protocol_string in df[true_label_column].unique():
        df_for_protocol = df[df[true_label_column] == protocol_string]
        # reset the index since subsetting the dataframe will cause each subsampled dataframe to have non-contiguous indexes which will cause it to fail on line 22
        df_for_protocol = df_for_protocol.reset_index(drop=False)

        class_specific_accuracies[protocol_string] = model_accuracy(df_for_protocol, true_label_column, prediction_column)
        class_specific_exact_accuracies[protocol_string] = exact_model_accuracy(df_for_protocol, true_label_column, prediction_column)

    print(class_specific_accuracies)

    if write_output == True:
        # df_accuracy = pd.DataFrame.from_dict(class_specific_accuracies)
        # df_accuracy.reset_index(drop=True, inplace=True)

        # df_exact_accuracy = pd.DataFrame.from_dict(class_specific_exact_accuracies)
        # df_exact_accuracy.reset_index(drop=True, inplace=True)
        if augmentation == True:
            with open(f.logs_filepath + "class_accuracies.csv",  'w', newline='') as a:
                writer = csv.writer(a)
                writer.writerow(['Protocol(s)', 'Sensitivity'])  # Header row
                for key, value in class_specific_accuracies.items():
                    writer.writerow([key, value])
            with open(f.logs_filepath + "class_exact_accuracies.csv",  'w', newline='') as a:
                writer = csv.writer(a)
                writer.writerow(['Protocol(s)', 'Exact Sensitivity'])  # Header row
                for key, value in class_specific_exact_accuracies.items():
                    writer.writerow([key, value])
        if augmentation == False:
            with open(f.logs_filepath + "augmented_class_accuracies.csv",  'w', newline='') as a:
                writer = csv.writer(a)
                writer.writerow(['Protocol(s)', 'Sensitivity'])  # Header row
                for key, value in class_specific_accuracies.items():
                    writer.writerow([key, value])
            with open(f.logs_filepath + "augmented_class_exact_accuracies.csv",  'w', newline='') as a:
                writer = csv.writer(a)
                writer.writerow(['Protocol(s)', 'Exact Sensitivity'])  # Header row
                for key, value in class_specific_exact_accuracies.items():
                    writer.writerow([key, value])

    return class_specific_accuracies, class_specific_exact_accuracies








    # gather all protocols into one list
    # for row in true_label_column:
    #     list_of_protocols_in_row = row.split(", ")
    #     for protocol in list_of_protocols_in_row:
    #         if protocol not in list_of_distinct_labels:
    #             list_of_distinct_labels.append(protocol)

    # class_specific_accuracies = {}
    # for label in list_of_distinct_labels:
    #     filtered_df = df[df["protocol"] == label]
    #     class_specific_accuracy = 0
    #     class_specific_accuracies[label] = class_specific_accuracy