import model
import helpers
from sklearn import preprocessing
import pandas as pd
from keras.utils import to_categorical
import hyperparameters
import filepaths

gold_standard_file = filepaths.gold_standard
gold_standard_aug_file = filepaths.gold_standard_aug
gold_standard_train_aug_file = filepaths.gold_standard_train
gold_standard_test_file = filepaths.gold_standard_test

""" Adapt code to load in new train and test sets with representation and augmentation. """
# train_aug = helpers.readData(gold_standard_train_aug_file)
# test = helpers.readData(gold_standard_test_file)

# preprocessed_train_aug = helpers.preprocessTextField(df=train_aug, col_name="indication")
# preprocessed_test = helpers.preprocessTextField(df=test, col_name="indication")

# model.LogisticModel_New(df_gold_standard_train_augmented=preprocessed_train_aug, df_gold_standard_test= preprocessed_test, augmentation=True)

gold_standard_CT_neuro = helpers.readData(filepath=gold_standard_file)
preprocessed_gold_standard_CT_neuro = helpers.preprocessTextField(df=gold_standard_CT_neuro, col_name="indication")
preprocessed_gold_standard_CT_neuro.to_csv(f"{filepaths.logs_filepath}gold_standard_preprocessed_indication.csv")

#model.LogisticModel_with_sex_and_age(dataset=preprocessed_gold_standard_CT_neuro, augmentation=False)
model.LogisticModel(dataset=preprocessed_gold_standard_CT_neuro, augmentation=False)

gold_standard_aug_file = filepaths.gold_standard_aug
gold_standard_aug_CT_neuro = helpers.readData(filepath=gold_standard_aug_file)
preprocessed_gold_standard_aug_CT_neuro = helpers.preprocessTextField(df=gold_standard_aug_CT_neuro, col_name="indication")
preprocessed_gold_standard_aug_CT_neuro.to_csv(f"{filepaths.logs_filepath}gold_standard_preprocessed_indication_augmented.csv")

model.LogisticModel_with_sex_and_age(dataset=preprocessed_gold_standard_CT_neuro, augmentation=True)
#model.LogisticModel(dataset=preprocessed_gold_standard_aug_CT_neuro, augmentation=True)
