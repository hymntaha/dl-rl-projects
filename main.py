!pip install tensorflow==2.10
!pip install opencv
!pip install opencv-python
!pip install matplotlib

!pip install tensorflow-gpu==2.10


import tensorflow as tf
tf.__version__

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

!pip list

import cv2
import imghdr

data_dir = 'data'