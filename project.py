from pyexpat import model

import numpy as np 
import keras
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import tensorflow as tf
import os
import glob
import cv2
from glob import glob

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

## path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")
## print("Path to dataset files:", path)


basepath = "C:/Users/User/.cache/kagglehub/datasets/paultimothymooney/breast-histopathology-images/versions/1"
folders = os.listdir(basepath)
print(len(folders), "folders found in basepath:", basepath)

data_all = glob('C:/Users/User/.cache/kagglehub/datasets/paultimothymooney/breast-histopathology-images/**/*.png', recursive=True)

images = []
labels = []

for i in data_all[:1500]:
    img = cv2.imread(i)
    images.append(img)
    labels.append(int(i.split("\\")[-2]))

labels = to_categorical(labels, num_classes=2)

images = np.stack(images, axis=0)

x, y = images, labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train / 255.0
x_test = x_test / 255.0 

CancerNet = Sequential()
CancerNet.add(Conv2D(16, (3,3), padding='same', input_shape = (50, 50, 3), activation = 'relu'))
CancerNet.add(MaxPooling2D(pool_size=(2, 2)))
CancerNet.add(Conv2D(32, (3,3), padding='same', activation = 'relu'))
CancerNet.add(MaxPooling2D(pool_size=(2, 2)))
CancerNet.add(Conv2D(64, (3,3), padding='same', activation = 'relu'))
CancerNet.add(MaxPooling2D(pool_size=(2, 2)))
CancerNet.add(Dropout(0.5))
CancerNet.add(Flatten())
CancerNet.add(Dense(units = 30, activation = 'relu'))
CancerNet.add(Dense(units = 2, activation = 'sigmoid'))

CancerNet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
CancerNet.summary()
    
hist = CancerNet.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data = (x_test, y_test))











