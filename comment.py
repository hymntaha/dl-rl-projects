import os
import pandas as pd
import tensorflow as tf
import numpy as np

df = pd.read_csv(os.path.join('data', 'train.csv', 'train.csv'))

# print(df.head())

from tensorflow.keras.layers import TextVectorization
X = df['comment_text']
y = df[df.columns[2:]].values

print(df.columns)

MAX_FEATURES = 200000

vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode='int')

vectorizer.adapt(X.values)

print(vectorizer("Hello world, life is great!"))

vectorizer.get_vocabulary()

vectorizer_text = vectorizer(X.values)
print(vectorizer_text)

dataset = tf.data.Dataset.from_tensor_slices((vectorizer_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottleneck

batch_X, batch_y = dataset.as_numpy_iterator().next()
print(batch_X.shape)

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).skip(int(len(dataset)*.1))

# Sequential Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional

model = Sequential()
model.add(Embedding(MAX_FEATURES+1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()

history = model.fit(train, epochs=10, validation_data=val)
print(history.history)

from matplotlib import pyplot as plt
plt.figure(figsize=(8.5))
pd.DataFrame(history.history).plot()
plt.show()

input_text = vectorizer('You freaking suck!')
print(input_text)

batch = test.as_numpy_iterator().next()
batch_X, batch_y = test.as_numpy_iterator().next()  
(model.predict(batch_X) > 0.5).astype(int)

res = model.predict(np.expand_dims(batch_X))
res.flatten()

from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy

pre = Precision()
re = Recall()
acc = CategoricalAccuracy()

for batch in test.as_numpy_iterator():
    # Unpack the batch
    X_true, y_true = batch
    # Make predictions
    yhat = model.predict(X_true)

    # Flatten the predictions and true values
    y_true = y_true.flatten()
    yhat = yhat.flatten()

    pre.update_state(y_true, yhat)
    re.updates_state(y_true, yhat)
    acc.update_state(y_true, yhat)  