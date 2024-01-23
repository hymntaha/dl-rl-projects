# import os
# for dirname, _, filenames in os.walk('./kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, tqdm_notebook

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm, tqdm_notebook

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import datasets, layers, models, callbacks
import matplotlib.pyplot as plt

import tensorflow as tf

train_dir = "./kaggle/input/Emergency_Vehicles/train"
test_dir = "./kaggle/input/Emergency_Vehicles/test"
train_df = pd.read_csv('./kaggle/input/Emergency_Vehicles/train.csv')
print(train_df.head())

im = cv2.imread("./kaggle/input/Emergency_Vehicles/train/1013.jpg")
plt.imshow(im)

im2 = cv2.imread("./kaggle/input/Emergency_Vehicles/train/2151.jpg")
plt.imshow(im2)

train_df.emergency_or_not=train_df.emergency_or_not.astype(str)

## In this part of the code, we do the following :-
#### Configure the Image Data Generator and specify the transformations
#### Have a train-test split ratio of 70-30.
#### Get the dimensions of the training and validation generators.
#### Finding the total number of samples for both training and validation generators

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, horizontal_flip = True)
batch_size = 32

traingen = datagen.flow_from_dataframe(dataframe = train_df[:1150],directory = train_dir,x_col ='image_names',
                                            y_col = 'emergency_or_not',class_mode = 'binary',batch_size = batch_size,
                                            target_size = (32,32))


valgen = datagen.flow_from_dataframe(dataframe = train_df[1151:],directory = train_dir,x_col = 'image_names',
                                                y_col ='emergency_or_not',class_mode ='binary',batch_size = 50,
                                                target_size = (32,32))

trainsample = next(traingen)
print("Dimensions of training sample",trainsample[0].shape)
valsample = next(valgen)
print("Dimensions of validation sample",valsample[0].shape)

num_train_samples = traingen.samples
num_val_samples = valgen.samples

print("Number of training samples", num_train_samples)
print("Number of validation samples", num_val_samples)

model = Sequential()
model.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(32,32,3), padding="same"))
model.add(tf.keras.layers.MaxPooling2D(2,2))
model.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

op = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
model.compile(optimizer = op, loss = loss, metrics = ['accuracy'])

history = model.fit(traingen, steps_per_epoch = num_train_samples/50, 
                    epochs = 50, validation_data = valgen,
                    validation_steps = num_val_samples/50)

history.history.keys()

figure = plt.figure(figsize = (10,4))
ax = plt.subplot(121)

ax.plot(history.history['loss'], 'r', label = 'train')
ax.plot(history.history['val_loss'],'g',label = 'val')

plt.legend()
plt.grid(axis = 'y')
plt.xlabel("Epoch")
plt.ylabel("Loss")

ax2 = plt.subplot(122)
ax2.plot(history.history['accuracy'], 'b', label="train")
ax2.plot(history.history['val_accuracy'], 'g', label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid(axis='y')
plt.savefig("Emergency dataset with data augmentation")

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('model_es.h5', monitor = 'val_loss',verbose=0, save_best_only=True)

model2 = Sequential()
model2.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(32,32,3), padding="same"))
model2.add(tf.keras.layers.MaxPooling2D(2,2))
model2.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same"))
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(10))
model2.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

callbacks = [early_stopping_callback, model_checkpoint_callback]

model2.compile(optimizer = op,
              loss = loss,
              metrics=['accuracy'])

history2 = model2.fit(traingen, steps_per_epoch = num_train_samples/50, 
                    epochs = 50, validation_data = valgen,
                    validation_steps = num_val_samples/50, callbacks = callbacks)

model2.summary()

history2.history.keys()
figure = plt.figure(figsize = (10,4))

ax = plt.subplot(121)
ax.plot(history2.history['loss'], 'r', label = 'train')
ax.plot(history2.history['val_loss'],'g',label = 'val')
plt.legend()
plt.grid(axis = 'y')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN + Data Augmentation + Early Stopping - Epoch VS Loss")

ax2 = plt.subplot(122)
ax2.plot(history2.history['accuracy'], 'b', label="train")
ax2.plot(history2.history['val_accuracy'], 'g', label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid(axis='y')

plt.title("CNN + Data Augmentation + Early Stopping - Epoch vs Accuracy")
plt.savefig("Emergency dataset with data augmentation + early stopping")

model3 = Sequential()

model3.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(32,32,3), padding="same")),
model3.add(tf.keras.layers.MaxPooling2D(2,2)),
model3.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same")),
model3.add(tf.keras.layers.Flatten()),
model3.add(tf.keras.layers.Dropout(0.2)),
model3.add(tf.keras.layers.Dense(10)),
model3.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))



model3.compile(optimizer = op,
              loss = loss,
              metrics=['accuracy'])

history3 = model3.fit(traingen, steps_per_epoch = num_train_samples/50, 
                    epochs = 50, validation_data = valgen,
                    validation_steps = num_val_samples/50)
figure = plt.figure(figsize = (10,4))
ax = plt.subplot(121)

ax.plot(history3.history['loss'], 'r', label = 'train')
ax.plot(history3.history['val_loss'],'g',label = 'val')

plt.legend()
plt.grid(axis = 'y')
plt.xlabel("Epoch")
plt.ylabel("Loss")


plt.title("CNN + Data Augmentation  + Dropout = 0.2 - Epoch Vs loss")
plt.savefig("Emergency dataset with data augmentation + early stopping")

ax2 = plt.subplot(122)
ax2.plot(history3.history['accuracy'], 'b', label="train")
ax2.plot(history3.history['val_accuracy'], 'g', label="val")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.grid(axis='y')

plt.title("CNN + Data Augmentation + Dropout = 0.2 - Epoch vs Accuracy")
plt.savefig("Emergency dataset with data augmentation + early stopping")

model4 = Sequential()
model4.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(32,32,3), padding="same"))
model4.add(tf.keras.layers.MaxPooling2D(2,2))
model4.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same"))
model4.add(tf.keras.layers.Flatten())
model4.add(tf.keras.layers.Dropout(0.4))
model4.add(tf.keras.layers.Dense(10))
model4.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))


model4.compile(optimizer = op,
              loss = loss,
              metrics=['accuracy'])

history4 = model4.fit(traingen, steps_per_epoch = num_train_samples/50, 
                    epochs = 50, validation_data = valgen,
                    validation_steps = num_val_samples/50)

figure = plt.figure(figsize = (10,4))

ax = plt.subplot(121)
ax.plot(history2.history['loss'], 'b', label = 'train')
ax.plot(history2.history['val_loss'],'g',label = 'val')
plt.legend()
plt.grid(axis = 'y')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN + Data Augmentation + Dropout = 0.4 - Epoch VS Loss")

figure = plt.figure(figsize = (10,4))

ax = plt.subplot(121)
ax.plot(history2.history['accuracy'], 'r', label = 'train')
ax.plot(history2.history['val_accuracy'],'g',label = 'val')
plt.legend()
plt.grid(axis = 'y')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN + Data Augmentation + Dropout = 0.4 - Epoch VS Accuracy")

model5 = Sequential()
model5.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(32,32,3), kernel_regularizer=tf.keras.regularizers.l1(0.01)))
model5.add(tf.keras.layers.MaxPooling2D(2,2))
model5.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same"))
model5.add(tf.keras.layers.Flatten())
model5.add(tf.keras.layers.Dense(10))
model5.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))


model5.compile(optimizer = op,
              loss = loss,
              metrics=['accuracy'])

history5 = model5.fit(traingen, steps_per_epoch = num_train_samples/50, 
                    epochs = 50, validation_data = valgen,
                    validation_steps = num_val_samples/50)

figure = plt.figure(figsize = (10,4))

ax = plt.subplot(121)
ax.plot(history5.history['loss'], 'b', label = 'train')
ax.plot(history5.history['val_loss'],'g',label = 'val')
plt.legend()
plt.grid(axis = 'y')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN + Data Augmentation + L1 regularization - Epoch VS Loss")

figure = plt.figure(figsize = (10,4))

ax = plt.subplot(121)
ax.plot(history5.history['accuracy'], 'r', label = 'train')
ax.plot(history5.history['val_accuracy'],'g',label = 'val')
plt.legend()
plt.grid(axis = 'y')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN + Data Augmentation + L1 regularization - Epoch VS Accuracy")

model6 = Sequential()
model6.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(32,32,3), kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model6.add(tf.keras.layers.MaxPooling2D(2,2))
model6.add(tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same"))
model6.add(tf.keras.layers.Flatten())
model6.add(tf.keras.layers.Dense(10))
model6.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))


model6.compile(optimizer = op,
              loss = loss,
              metrics=['accuracy'])

history6 = model6.fit(traingen, steps_per_epoch = num_train_samples/50, 
                    epochs = 50, validation_data = valgen,
                    validation_steps = num_val_samples/50)

figure = plt.figure(figsize = (10,4))

ax = plt.subplot(121)
ax.plot(history5.history['loss'], 'b', label = 'train')
ax.plot(history5.history['val_loss'],'g',label = 'val')
plt.legend()
plt.grid(axis = 'y')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN + Data Augmentation + L2 regularization - Epoch VS Loss")

figure = plt.figure(figsize = (10,4))

ax = plt.subplot(121)
ax.plot(history5.history['accuracy'], 'r', label = 'train')
ax.plot(history5.history['val_accuracy'],'g',label = 'val')
plt.legend()
plt.grid(axis = 'y')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN + Data Augmentation + L2 regularization - Epoch VS Accuracy")