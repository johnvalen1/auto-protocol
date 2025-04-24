import model
import helpers
from sklearn import preprocessing
import pandas as pd
from keras.utils import to_categorical
import hyperparameters
import filepaths

gold_standard_file = filepaths.gold_standard
gold_standard_aug_file = filepaths.gold_standard_aug


gold_standard_CT_neuro = helpers.readData(filepath=gold_standard_file)
preprocessed_gold_standard_CT_neuro = helpers.preprocessTextField(df=gold_standard_CT_neuro, col_name="indication")
preprocessed_gold_standard_CT_neuro.to_csv(f"{filepaths.logs_filepath}gold_standard_preprocessed_indication.csv")


#model.LogisticModel_with_sex_and_age(dataset=preprocessed_gold_standard_CT_neuro, augmentation=False)
model.LogisticModel(dataset=preprocessed_gold_standard_CT_neuro, augmentation=False)

gold_standard_aug_file = filepaths.gold_standard_aug
gold_standard_aug_CT_neuro = helpers.readData(filepath=gold_standard_aug_file)
preprocessed_gold_standard_aug_CT_neuro = helpers.preprocessTextField(df=gold_standard_aug_CT_neuro, col_name="indication")
preprocessed_gold_standard_aug_CT_neuro.to_csv(f"{filepaths.logs_filepath}gold_standard_preprocessed_indication_augmented.csv")

#model.LogisticModel_with_sex_and_age(dataset=preprocessed_gold_standard_CT_neuro, augmentation=True)
model.LogisticModel(dataset=preprocessed_gold_standard_aug_CT_neuro, augmentation=True)

#shuffle the corpus before splitting it into train/test
# shuffled_corpus = corpus.sample(frac=1, random_state=42).reset_index(drop=True)

# pre-process the indication column using lemmatization, removing stopwords, and removing punctuation
# train_corpus = shuffled_corpus.iloc[:int(len(shuffled_corpus) * hyperparameters.train_size)]
# train_corpus.to_csv(f"{filepaths.train_dir}train_corpus_raw.csv")
# train_corpus = helpers.preprocessTextField(df=train_corpus, col_name="indication")
# train_corpus.to_csv(f"{filepaths.train_dir}train_corpus_preprocessed_indication.csv")

# test_corpus = shuffled_corpus.iloc[int(len(shuffled_corpus) * hyperparameters.train_size):]
# test_corpus.to_csv(f"{filepaths.test_dir}test_corpus_raw.csv")
# test_corpus = helpers.preprocessTextField(df=test_corpus, col_name="indication")
# test_corpus.to_csv(f"{filepaths.test_dir}test_corpus_preprocessed_indication.csv")