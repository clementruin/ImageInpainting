#### Data Pipeline - Cars Dataset

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle
import csv
import os
from PIL import Image


# Modify to run on your local machine
cars_path = "/Users/Clement/Desktop/CS282_ImageInpainting/datasets/"
test_path = "/Users/Clement/Desktop/CS282_ImageInpainting/datasets/cars_test/"
train_path = "/Users/Clement/Desktop/CS282_ImageInpainting/datasets/cars_train/"
cars_prep_path = "/Users/Clement/Desktop/CS282_ImageInpainting/datasets/cars_pp/"


# Parameters of the images
height = width = 64
channels = 3 #RGB
range_x = np.arange(width)
range_y = np.arange(height)
range_h = np.arange(5, 20)
range_w = np.arange(5, 20)


# Open images as Numpy array
test_images = os.listdir(test_path)
train_images = os.listdir(train_path)

X_train = np.zeros((len(train_images), height, width, channels))
X_test = np.zeros((len(test_images), height, width, channels))

print("Loading Test Set...")
for i in range(len(test_images)):
    img = Image.open(test_path + test_images[i])
    try:
        X_test[i] = np.asarray(img.resize((height, width)))
    except:
        # gray scale images cannot be loaded as RGB
        print(test_images[i], "Could not be loaded")
        X_test[i] = X_test[i-1] #just to not have a null image
print(" ")
        
print("Loading Train Set...")
for i in range(len(train_images)):
    img = Image.open(train_path + train_images[i])
    try:
        X_train[i] = np.asarray(img.resize((height, width)))
    except:
        # gray scale images cannot be loaded as RGB
        print(train_images[i], "Could not be loaded")
        X_train[i] = X_train[i-1] #just to not have a null image


# Sanity check
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# Concat train and test set (we will define our own train and test set later)
X_all = np.vstack([X_train, X_test])
print("X_all shape:", X_all.shape)


# Reshape to standard format (N, C, H, W) for deep learning models
def reshape_NCHW(X):
    return X.transpose((0, 3, 1, 2))

X_all = reshape_NCHW(X_all)
print("X_all shape:", X_all.shape)


# Standardize images
def standardize(X):
    return X / 255.

X_all = standardize(X_all)


# save an original X_all dataset
X_original = X_all.copy()


def mask(img, loc):
    """Set to zero all pixels contained in a given location on the image
    
    Input:
    - img: image of shape (C, H, W)
    - loc: location (x, y, h, w)
    """
    x, y, h, w = loc
    h = min(h, height-y)
    w = min(w, width-x)
    img[:, x:x+w, y:y+h] = 0.


np.random.seed(123)
xs = np.random.choice(range_x, X_all.shape[0])
ys = np.random.choice(range_y, X_all.shape[0])
hs = np.random.choice(range_h, X_all.shape[0])
ws = np.random.choice(range_w, X_all.shape[0])


#Gaussian mask
xg = np.random.normal(16,10,X_all.shape[0]) #get the upper left corner
yg = np.random.normal(48,10,X_all.shape[0])

for img, x, y, h, w in zip(X_all, xg.astype(int), yg.astype(int), hs, ws):
    mask(img, (x, y, h, w))

# Save modified dataset (with masks) as pickle file in cars_prep_path directory
with open(cars_prep_path + "cars_masks.pickle", "wb") as file:
    pickle.dump(X_all, file)


# Save original dataset (without masks) as pickle file in cars_prep_path directory
with open(cars_prep_path + "cars_original.pickle", "wb") as file:
    pickle.dump(X_original, file)
