import numpy as np
np.random.seed(42)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

(X_train, y_train), (X_test, y_test) = mnist.load_data()