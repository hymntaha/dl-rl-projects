import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# IMPORTS

# Torch
import torch
import torchinfo
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ML-related
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Default Python
import random
from pathlib import Path

# Other Libraries
from PIL import Image
from tqdm import tqdm


root_dir = Path('./car_data')
cars_annos = root_dir / 'cars_annos.mat'
cars_test = root_dir / 'cars_test' / 'cars_test'
cars_train = root_dir / 'cars_train' / 'cars_train'

cars_annos_mat = loadmat(cars_annos)
training_images = os.listdir(cars_train)
testing_images = os.listdir(cars_test)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Looking at cars_annos.mat's structure. It's a dictionary with an "annotations" key with nested numpy arrays
# There's also a class_names attribute which is self explanatory. Looking at the last element of the annotations
# array, we see the structure of each numpy array in "annotations": every array will have 7 elements, image path,
# box11, box12, box21, box22, class name, and test
#
# I am not going to use the bounding boxes as a feature in this case, so now the task is to figure out the "test"
# attribute, what it does. My hunch is that "test" is a boolean, 0 if it's a training sample and 1 if testing

# This array will be 0-indexed; we will have to remember that whenever we use it
class_names = [arr[0] for arr in cars_annos_mat['class_names'][0]]

# A bunch of weird indexing will happen because of how nested these annotation arrays are
sample1 = cars_annos_mat['annotations'][0][0]
sample1_path, sample1_class_name, sample1_test = sample1[0][0].split("/")[-1], sample1[5][0][0], sample1[6][0][0]

# So image 1 is class 1 (0 in our class_names array and has a test value of 0. According to our hypothesis, it should be
# a training case. Let's open our image and confirm)

# THIS CODE WILL NOT RUN -> Image.open(os.path.join(cars_train, sample1_path))
# There's a problem with the cars.annos.mat file -> the files it's referencing (000001.jpg) for example doesn't exist in
# the training or testing image sample. How will we proceed?

# We will also use the "standford-cars-dataset-meta" dataset which has an updated version
# of the .mat files including ones for testing as well

# New mat files

root_dir = Path("/kaggle/input/standford-cars-dataset-meta/")
cars_annos_train = root_dir / "devkit" / "cars_train_annos.mat"
cars_annos_test = root_dir / "cars_test_annos_withlabels (1).mat"

cars_meta_mat = loadmat(root_dir / "devkit" / "cars_meta.mat")
cars_annos_train_mat, cars_annos_test_mat = loadmat(cars_annos_train), loadmat(cars_annos_test)

class_names = [arr[0] for arr in cars_meta_mat['class_names'][0]]

# New structure of mat file's "annotations array is going to be" 6 elements with
# box11, box12, box21, box22, class, and filename

sample1 = cars_annos_train_mat['annotations'][0][0]
sample1_path, sample1_class = sample1[-1][0], sample1[-2][0][0] - 1

# Running the same "Image.open()" code as before works now and we have the correct filepaths
# and labels. We're good to go

# Check 3 examples for training and 3 for testing before proceeding
# print(class_names[sample1_class])
# Image.open(os.path.join(cars_train, sample1_path))

w, h = 2, 3
fig, axes_list = plt.subplots(h, w, figsize=(5*w, 3*h)) 
fig.suptitle('Training samples check')

for axes in axes_list:
    for ax in axes:
        ax.axis('off')
        random_index = random.randint(0, 500)
        random_sample = cars_annos_train_mat['annotations'][0][random_index]
        sample_path, sample_class = random_sample[-1][0], random_sample[-2][0][0] - 1
        im = Image.open(os.path.join(cars_train, sample_path))
        ax.imshow(im)
        ax.set_title(class_names[sample_class], fontdict={"fontsize": 10})

fig, axes_list = plt.subplots(h, w, figsize=(5*w, 3*h)) 
fig.suptitle('Testing samples check')

for axes in axes_list:
    for ax in axes:
        ax.axis('off')
        random_index = random.randint(0, 500)
        random_sample = cars_annos_test_mat['annotations'][0][random_index]
        sample_path, sample_class = random_sample[-1][0], random_sample[-2][0][0] - 1
        im = Image.open(os.path.join(cars_test, sample_path))
        ax.imshow(im)
        ax.set_title(class_names[sample_class], fontdict={"fontsize": 10})

# Looking good so far. Now we're ready to define our model, transformations, datasets, dataloaders, and training/testing loop

googlenet_weights = torchvision.models.GoogLeNet_Weights.DEFAULT
googlenet = torchvision.models.googlenet(weights=googlenet_weights).to(device)
googlenet_transforms = googlenet_weights.transforms()

for param in googlenet.parameters():
    param.requires_grad = False

googlenet.fc = nn.Sequential(
    nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
).to(device)

# Datasets - create custom dataset and a dictionary which relates image path to label

training_image_label_dictionary, testing_image_label_dictionary = {}, {}

for arr in cars_annos_train_mat['annotations'][0]:
    image, label = arr[-1][0], arr[-2][0][0] - 1
    training_image_label_dictionary[image] = label

for arr in cars_annos_test_mat['annotations'][0]:
    image, label = arr[-1][0], arr[-2][0][0] - 1
    testing_image_label_dictionary[image] = label

# Using these data structures, we'll be able to return an image and a label easily in our custom dataset as we'll see in a bit
print(len(training_image_label_dictionary), len(testing_image_label_dictionary))