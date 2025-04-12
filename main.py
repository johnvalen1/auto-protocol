import model
import helpers
from sklearn import preprocessing
import pandas as pd
from keras.utils import to_categorical
import hyperparameters
import filepaths

gold_standard_file = filepaths.gold_standard
corpus = helpers.readData(filepath=gold_standard_file)
preprocessed_corpus = helpers.preprocessTextField(df=corpus, col_name="indication")
preprocessed_corpus.to_csv(f"{filepaths.logs_filepath}corpus_preprocessed_indication.csv")

model.LogisticModel(corpus=preprocessed_corpus, file_output_path=filepaths.file_output_path)

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



