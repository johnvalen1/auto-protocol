import pandas as pd
import class_labels

filepath = './CT_Neuro_Train_Data.csv'

def readData(filepath):
    """ This function reads in the .csv file with the indication, protocol, and identifier. """
    corpus = pd.read_excel(filepath)
    indication = corpus['indication']
    protocol = corpus['protocol']
    zipped = zip(indication, protocol)
    return pd.DataFrame(zipped, columns = ['indication','protocol'])

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