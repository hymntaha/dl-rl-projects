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