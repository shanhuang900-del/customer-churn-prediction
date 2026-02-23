# Customer Churn Prediction with Deep Neural Networks

## Overview
This project builds a deep neural network model to predict customer churn using a cleaned subset of the Iranian Churn dataset from the UCI Machine Learning Repository.  
The goal is to identify customers likely to discontinue a service and demonstrate how deep learning can capture nonlinear behavioral patterns in structured data.

## Dataset
- Source: Iranian Churn dataset (UCI Machine Learning Repository)
- Samples: 990 customers
- Features: 13 numerical and categorical customer attributes
- Target: Binary churn indicator (1 = churn, 0 = stay)
- Balanced class distribution (~1:1)

## Data Preprocessing
- Removed index column and separated target variable
- Standardized continuous variables using z-score normalization
- One-hot encoded categorical variables
- Final feature matrix: 19 inputs
- Stratified train/test split (90% train, 10% test)

## Model Architecture
Implemented using TensorFlow/Keras in Google Colab.

Final model (Model 4):
- Input layer: 19 features
- Hidden layer 1: 80 neurons (ReLU) + Dropout(0.3)
- Hidden layer 2: 30 neurons (ReLU) + Dropout(0.3)
- Output layer: 1 neuron (Sigmoid)

Training configuration:
- Optimizer: Adam
- Loss: Binary cross-entropy
- Epochs: up to 200 with early stopping
- Batch size: 32
- Random seeds fixed for reproducibility

## Results
Final test performance:

- Accuracy: **92.93%**
- ROC-AUC: **0.9616**

The selected model achieved the best balance between predictive performance and generalization among four tested neural network architectures.

## Repository Structure
- `churn_prediction.ipynb` – full preprocessing, modeling, and evaluation pipeline  
- `churn.csv` – dataset used in the analysis  
- `Churn Prediction 2026:02:01.pdf` – detailed project report  

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Google Colab

## Key Takeaways
- Neural networks can effectively model churn behavior even in small structured datasets
- Dropout regularization improved generalization and stability
- Model comparison using ROC-AUC provided deeper insight than accuracy alone

## Author
Shan Huang

