# Importing Libraries:
# These libraries allow us to use tools such as:
# Numerical computations (numpy)
# Deep learning (tensorflow, keras)
# Image processing (opencv)
# Data visualization (matplotlib)
# Downloading datasets from Kaggle (kagglehub)
# Directory handling (os, glob)

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os
import kagglehub
import cv2
from glob import glob

# Importing necessary modules from keras and scikit-learn for building the CNN model
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Importing additional modules for evaluation and visualization
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#The dataset contains thousands of breast tissue histopathology images.
path = kagglehub.dataset_download("paultimothymooney/breast-histopathology-images")
print("Dataset path:", path)
basepath = "C:/Users/User/.cache/kagglehub/datasets/paultimothymooney/breast-histopathology-images/versions/1"
basepath = path
folders = os.listdir(basepath)
print(len(folders), "folders found in basepath:", basepath)

# Reading images from the dataset and creating empty lists to store the images and their labels
data_all = glob('C:/Users/User/.cache/kagglehub/datasets/paultimothymooney/breast-histopathology-images/**/*.png', recursive=True)
images = []
labels = []

# Each image is read from disk and stored in memory.
for i in data_all[:50000]:
    # Checking validity of image file and resizing if improperly formatted
    if i.endswith(".png"):
        print(i)
        if i is None:
            print("Invalid image")
            continue
        img = cv2.imread(i)
        if Exception:
            img = cv2.resize(img, (50, 50))
        images.append(img)
        
# Splits the path every time it finds a \
# The folder names contain the class labels:
# The program extracts these labels automatically from the folder structure.
    labels.append(int(i.split("\\")[-2]))

# Visualization 

#The code displays the first 10 histopathology images from the dataset.
plt.figure(figsize=(10,5))

for i in range(10):
    plt.subplot(2,5,i+1)
    
# Displays examples from the dataset.
# Verification that images loaded correctly
# Understand the type of breast tissue images being analyzed

    plt.imshow(images[i])
    plt.title(f"Class {np.argmax(labels[i])}")
    plt.axis("off")

plt.tight_layout()
plt.show()

#class distribution
label_values = np.argmax(labels, axis=1)

plt.figure(figsize=(6,4))

#Shows the number of images in each class.
#(An imbalanced dataset can lead to biased predictions.
plt.hist(label_values, bins=2)

plt.xticks(
    [0,1],
    ["No Cancer", "Cancer"]
)

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.show()

# Preprocessing the data for training the CNN model
# The labels are one-hot encoded to represent cancer, no cancer
labels = to_categorical(labels, num_classes=2)

# Images stacked along a single axis
images = np.stack(np.array(images))
x, y = images, labels

# Training Set Used to teach the CNN. (80%)
# Test set used to evaluate the CNN. (20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalizing the pixel values of the images
x_train = x_train / 255.0
x_test = x_test / 255.0 

# Building the actual CNN model with a mix of convolutional, pooling, dropout, and dense layers.
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

# The curves, training
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)

# These graphs show how the CNN learns.
# 1. Shows improvement in prediction performance "Accuracy"
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'])

plt.subplot(1,2,2)

# 2.Shows reduction in prediction error."Loss"
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Validation'])

plt.tight_layout()
plt.show()

# Model evaluation 
# Tests the CNN using images that were never seen during training.
test_loss, test_acc = CancerNet.evaluate(
    x_test,
    y_test
)

print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# Confusion matrix
# This part summarizes the prediction results (cancer/no cancer)
predictions = CancerNet.predict(x_test)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(
    y_true,
    y_pred
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Cancer", "Cancer"]
)

disp.plot()
plt.show()

# Classification report
print(
    classification_report(
        y_true,
        y_pred
    )
)









