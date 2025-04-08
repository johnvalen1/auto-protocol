import model
import helpers
from sklearn import preprocessing
import pandas as pd
from keras.utils import to_categorical
import hyperparameters

filepath = './CT_Neuro_Gold_Standard.xlsx'
file_output_path = "./model_results.txt"
corpus = helpers.readData(filepath=filepath)
#shuffle the corpus before splitting it into train/test
shuffled_corpus = corpus.sample(frac=1, random_state=42).reset_index(drop=True)
#X_train, X_test, y_train, y_test = train_test_split(indication, protocol_mulit_label, test_size=0.2, random_state=42)
train_corpus = shuffled_corpus.iloc[:int(len(shuffled_corpus) * hyperparameters.train_size)]
test_corpus = shuffled_corpus.iloc[int(len(shuffled_corpus) * hyperparameters.train_size):]

train_corpus.to_csv("./Documentation/train_corpus.csv")
test_corpus.to_csv("./Documentation/test_corpus.csv")
# pre-process the indication column using lemmatization, removing stopwords, and removing punctuation


model.LogisticModel(train_corpus, test_corpus, file_output_path)

# label = preprocessing.LabelEncoder()
# # y = label.fit_transform(protocol)
# # y = to_categorical(y)

# y = helpers.DefineLabelVectorization(protocol)
# print(y)