import helpers
from sklearn import preprocessing
# import tokenization
# import tensorflow as tf
# import tensorflow_hub as hub
from keras.utils import to_categorical

filepath = './CT_Neuro_Gold_Standard.xlsx'

corpus = helpers.readData(filepath=filepath)

indication = corpus['indication']
protocol = corpus['protocol']

print(indication)
print(protocol)


label = preprocessing.LabelEncoder()
# y = label.fit_transform(protocol)
# y = to_categorical(y)

y = helpers.DefineLabelVectorization(protocol)
print(y)