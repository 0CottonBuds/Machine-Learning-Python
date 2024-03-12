import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10_000)

word_index: dict = keras.datasets.imdb.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
# these are extra special words we will use for our data sets
# PAD is for padding
word_index["<PAD>"] = 0
# Start denotes the start of data
word_index["<START>"] = 1
# End denotes end of data
word_index["<END>"] = 2
# Unused is unused
word_index["<UNUSED"] = 3

reversed_word_index = dict([(value, key) for key, value in word_index.items()])

# pre process the data set to add padding so we can have a uniform input
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen= 256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen= 256)

def decode_review(text):
    return " ".join([reversed_word_index.get(i, "?") for i in text]) 

# print(decode_review(test_data[0]))

# model
model = keras.Sequential()
model.add(keras.layers.Embedding(10_000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

train_data_validator = train_data[:10_000]
train_data = train_data[10_000:]

train_labels_validator = train_labels[:10_000]
train_labels =  train_labels[10_000:]

fitted_model = model.fit(train_data, train_labels, epochs=40, batch_size=512, validation_data=(train_data_validator, train_labels_validator), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

