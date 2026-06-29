# Breast Cancer Classification Using a Convolutional Neural Network (CNN)

**Authors:**  
Scheffer Miklós  
Uribe Unigarro Mariana

---

## Project Overview

This project develops a **Convolutional Neural Network (CNN)** to classify breast histopathology images as either **cancerous** or **non-cancerous**. The objective is to demonstrate how deep learning can assist in medical image analysis by automatically identifying patterns associated with breast cancer.

---

## 1. Problem Statement

Breast cancer is one of the most common types of cancer worldwide, and early detection is essential for improving treatment outcomes. Analyzing histopathology images manually is a time-consuming process that requires expert knowledge. This project explores the use of **Convolutional Neural Networks (CNNs)** to automatically classify breast tissue images as **cancerous** or **non-cancerous**, providing a computer-aided approach to medical image analysis.

---

## 2. Project Approach

The project follows a supervised learning approach. A **Convonutional Neural Network (CNN)** is trained using labeled histopathology images to learn visual patterns associated with each class. After training, the model is evaluated on unseen data to measure its classification performance.

---

## 3. Project Phases

The project is divided into the following phases:

- **Data Acquisition:** Download and load the breast histopathology image dataset.
- **Data Preprocessing:** Resize, normalize, and label the images before splitting them into training and testing datasets.
- **Exploratory Analysis:** Visualize sample images and analyze the class distribution.
- **Model Development:** Design and build the CNN architecture.
- **Model Training:** Train the network while monitoring learning performance.
- **Model Evaluation:** Assess the trained model using accuracy, loss, a confusion matrix, and a classification report.

---

## 4. Repository Structure

```text
Project/
│
├── project.py          # Main program
├── README.md           # Project documentation
└── dataset/            # Downloaded dataset 
```

---

## 5. Usage Information

1. Install the required Python libraries.
2. Download the dataset automatically using KaggleHub.
3. Run `project.py`.
4. The program will:
   - Load and preprocess the dataset.
   - Display sample images and class distribution.
   - Train the CNN model.
   - Evaluate the model.
   - Display the training curves, confusion matrix, and classification report.

---

## 6. Implementation Overview

The program automatically downloads the **Breast Histopathology Images** dataset using KaggleHub and loads **80%** of the labeled histopathology images (as per the 1. Project Idea). Each image is resized to **50 × 50 pixels**, normalized, and assigned a class label based on the dataset folder structure before being split into **90% training** and **10% testing** datasets.

Before training, the program visualizes sample images and the distribution of cancerous and non-cancerous samples, allowing the dataset to be inspected and verified.

A custom **Convolutional Neural Network (CancerNet)** is then built using **three convolutional layers (16, 32, and 64 filters)** with max-pooling layers for feature extraction, followed by a **Dropout layer (0.5)** to reduce overfitting and fully connected layers for binary classification.

The model is trained using the **Adam optimizer**, **categorical cross-entropy loss**, a **batch size of 32**, and **100 training epochs**. During training, both **accuracy** and **loss** are monitored for the training and validation datasets to evaluate the learning process and detect potential overfitting.

Finally, the trained model is evaluated on previously unseen test images. Its performance is assessed using **test accuracy**, **test loss**, a **confusion matrix**, and a **classification report** containing precision, recall, and F1-score, providing a comprehensive evaluation of its ability to distinguish between cancerous and non-cancerous breast tissue images.


---
