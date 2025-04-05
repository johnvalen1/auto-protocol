import model
import helpers
from sklearn import preprocessing
import pandas as pd
from keras.utils import to_categorical

filepath = './CT_Neuro_Gold_Standard.xlsx'
file_output_path = "./model_results.txt"
corpus = helpers.readData(filepath=filepath)

indication = corpus['indication']
# pre-process the indication column using lemmatization, removing stopwords, and removing punctuation
indication = helpers.preprocessIndication(indication)
protocol = corpus['protocol']

model.LogisticModel(indication, protocol, file_output_path)

# label = preprocessing.LabelEncoder()
# # y = label.fit_transform(protocol)
# # y = to_categorical(y)

# y = helpers.DefineLabelVectorization(protocol)
# print(y)