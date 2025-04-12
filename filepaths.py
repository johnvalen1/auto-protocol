gold_standard = './CT_Neuro_Gold_Standard.xlsx'
file_output_path = "./model_results.txt"
# where to store important .csv's and track metadata
logs_filepath = "./doc/"

train_dir = logs_filepath + "train/"
test_dir = logs_filepath + "test/"

import os
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
