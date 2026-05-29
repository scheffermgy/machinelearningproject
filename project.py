import numpy as np 
import keras
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical

classifier = Sequential()
c1 = Conv2D(16, (3,3), padding='same', input_shape = (50, 50, 3), activation = 'relu')
classifier.add(c1)
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5))
classifier.add(Flatten())

w1 = Dense(units = 30, activation = 'relu')
w3 = Dense(units = 10, activation = 'softmax')
classifier.add(w1)
classifier.add(Dropout(0.5)) 
classifier.add(w3)