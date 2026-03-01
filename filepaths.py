gold_standard_aug = './CT_Neuro_Gold_Standard - Augmented - v2.xlsx'
# gold_standard_aug = './CT_Neuro_Gold_Standard - Augmented.xlsx'

# the following train and test sets should be based on the above file, and only this file, where we use the augmented/original column to filter out unneeded rows

gold_standard_train = "./train test sets/CT_Neuro_Gold_Standard - Train Set.xlsx"
gold_standard_test = "./train test sets/CT_Neuro_Gold_Standard - Test Set.xlsx"

file_output_path = "./model_results.txt"
# where to store important .csv's and track metadata
logs_filepath = "./doc/"
preprocess_filepath = logs_filepath + "preprocess/"
metrics_filepath = logs_filepath + "metrics/"

train_dir = logs_filepath + "train/"
test_dir = logs_filepath + "test/"


import os
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)