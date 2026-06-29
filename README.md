Breast Cancer Classification Using a Convolutional Neural Network (CNN)
Scheffer Miklós, Uribe Unigarro Mariana

Project Overview

This project develops a Convolutional Neural Network (CNN) to classify breast histopathology images as either cancerous or non-cancerous. The objective is to demonstrate how deep learning can assist in medical image analysis by automatically identifying patterns associated with breast cancer.

1. Problem Statement

Breast cancer is one of the most common types of cancer worldwide, and early detection is essential for improving treatment outcomes. Analyzing histopathology images manually is a time-consuming process that requires expert knowledge. This project explores the use of Convolutional Neural Networks (CNNs) to automatically classify breast tissue images as cancerous or non-cancerous, providing a computer-aided approach to medical image analysis.

2. Project Approach

The project follows a supervised learning approach. A CNN is trained using labeled histopathology images to learn visual patterns associated with each class. After training, the model is evaluated on unseen data to measure its classification performance.

3. Project Phases

The development of the project is divided into the following phases:

Data Acquisition: Download and load the breast histopathology image dataset.
Data Preprocessing: Resize, normalize, and label the images before splitting them into training and testing datasets.
Exploratory Analysis: Visualize sample images and analyze the class distribution.
Model Development: Design and build the CNN architecture.
Model Training: Train the network using the training dataset while monitoring learning performance.
Model Evaluation: Evaluate the trained model using accuracy, loss, a confusion matrix, and a classification report.

4. Repository Structure
Project/
│
├── project.py            # Main program
├── README.md             # Project documentation
├── requirements.txt      # Required Python libraries
├── dataset/              # Downloaded dataset (generated locally)
└── results/              # Generated figures and evaluation results (optional)

5. Usage Information

To run the project:

Install the required Python libraries.
Download the dataset automatically through KaggleHub.
Execute the main program.
The program will:
Load and preprocess the dataset.
Display sample images and class distribution.
Train the CNN model.
Evaluate the trained model.
Display the accuracy, loss curves, confusion matrix, and classification report.


6. Detailed Description

This project implements a complete deep learning pipeline for breast cancer image classification. The program begins by downloading and loading a publicly available histopathology image dataset. Each image is preprocessed by resizing it to a fixed size, assigning its corresponding label, and normalizing its pixel values.

The processed data is divided into training and testing sets. A Convolutional Neural Network (CNN) is then constructed to automatically extract relevant image features and classify each sample as either cancerous or non-cancerous. During training, the model's accuracy and loss are monitored to evaluate its learning progress.

Finally, the trained model is tested on previously unseen images. Its performance is assessed using standard classification metrics, including test accuracy, loss, a confusion matrix, and a classification report. These results provide insight into the model's effectiveness and its ability to correctly identify breast cancer from histopathology images.
