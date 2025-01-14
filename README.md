# Network Intrusion Detection System (NIDS)

A ML/DL solution using XAI to detect and classify network intrusions in real-time using packet flow data.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Technologies Used](#technologies-used)
6. [Setup Instructions](#setup-instructions)
7. [Project Workflow](#project-workflow)
8. [Results](#results)
9. [Future Enhancements](#future-enhancements)
10. [Acknowledgments](#acknowledgments)
11. [Contact](#contact)

## Introduction
Network Intrusion Detection Systems (NIDS) are essential for monitoring and identifying malicious activities within a network. This project leverages ML and DL models to classify and predict network traffic in the cyber space.

## Features
- **Binary Classification:** Detects whether network traffic is benign or malicious.
- **Multi-Class Classification:** Classifies specific types of attacks.
- **Explainable Decision Tree Classification:** Provides interpretability for multi-class results.
- **Real-Time Detection:** Processes uploaded packet flow data in CSV format.
- **Visualization:** Generates confusion matrices, ROC-AUC curves, and spectrograms for insights.

## Dataset
This is the link to the dataset used for this project->[CIC-IDS- 2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

The part of dataset used for our analysis includes network traffic data from:
- **Friday Working Hours:** Afternoon DDoS traffic.
- **Thursday Working Hours:** Morning Web Attacks.
  

### Class Labels
1. Benign
2. PortScan (Attack)
3. Web Attacks (Brute Force, XSS, SQL Injection)

## Model Architecture
### Machine Learning Models:
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting

### Deep Learning Models:
- 1D CNN
- LSTM
- ANN

### Architecture Diagram
![Spectrogram Example](https://ibb.co/m8LL9MD "Generated Spectrogram")

## Technologies Used
- **Programming Languages:** Python
- **Frameworks:** TensorFlow, Keras, Scikit-learn
- **Libraries:** Matplotlib, Seaborn, Librosa, Joblib
- **Deployment Platform:** Streamlit
- **Cloud Services:** AWS S3 for data and model storage, Sagemaker for model deployment

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
