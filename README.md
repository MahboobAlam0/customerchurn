# Customer Churn using Random Forest Classifier

## Project Overview
This project aims to predict customer churn using machine learning techniques. It utilizes the Telco Customer Churn dataset to analyze customer behavior and determine the likelihood of churn based on various factors such as tenure, contract type, and payment method.

## Dataset
The dataset consists of customer information, including:
- Tenure
- Monthly Charges
- Total Charges
- Contract Type
- Payment Method
- Internet Service
- Churn (Target Variable)

## Model Used
A **Random Forest Classifier** was used to predict customer churn. The model was trained and evaluated using accuracy scores and cross-validation.

## Performance Metrics
- **Accuracy Score:** 77.29%
- **Cross-Validation Scores:**
  - Fold 1: 76.44%
  - Fold 2: 79.35%
  - Fold 3: 76.01%
  - Fold 4: 77.41%
  - Fold 5: 78.12%

## Features
The following features were used for training the model:
- **Numerical Features:** Tenure, Monthly Charges, Total Charges
- **Categorical Features:** Internet Service, Contract, Payment Method (Encoded)

## Model Saving
The trained model was saved using **Pickle** with gzip compression for efficient storage and reuse.

## Prediction Function
A function was designed to take customer details as input and return a prediction indicating whether the customer is likely to churn.

## Usage
This model can be integrated into a customer management system to identify high-risk customers and take proactive measures to retain them.

## Future Improvements
- Experiment with other machine learning models.
- Implement feature engineering techniques to improve prediction accuracy.
- Develop a user-friendly web interface for customer churn prediction.
