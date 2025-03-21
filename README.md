# Customer Churn using Random Forest Classifier

## Overview
This project focuses on predicting customer churn using machine learning techniques. The dataset used is `Customer Churn.csv`, which includes various customer attributes such as tenure, monthly charges, contract type, and payment method.

## Dataset
The dataset consists of 7043 records and 7 selected features:
- `tenure`
- `MonthlyCharges`
- `TotalCharges`
- `Contract`
- `PaymentMethod`
- `InternetService`
- `Churn` (Target variable)

## Data Preprocessing
- Missing values in `TotalCharges` were converted to numeric and replaced with 0.
- Categorical columns (`Contract`, `PaymentMethod`, `InternetService`, `Churn`) were label encoded.
- Features were scaled using `StandardScaler`.
- The dataset was split into training (80%) and testing (20%) sets.

## Model Used
A `RandomForestClassifier` with 100 estimators and a fixed random state was trained on the processed dataset.

## Model Evaluation
- **Accuracy Score:** `0.7729`
- **Cross-validation Scores:** `[0.7644, 0.7935, 0.7601, 0.7741, 0.7813]`

## Saving the Model
The trained model is saved using `pickle` with gzip compression:
```python
import pickle
import gzip

with gzip.open("model.pkl.gz", "wb") as f:
    pickle.dump(model, f)
```

## Prediction Function
A function to predict churn based on input features:
```python
def prediction(tenure, TotalCharges, MonthlyCharges, InternetService, Contract, PaymentMethod):
    data = {
        'tenure': [tenure],
        'TotalCharges': [TotalCharges],
        'MonthlyCharges': [MonthlyCharges],
        'InternetService': [InternetService],
        'Contract': [Contract],
        'PaymentMethod': [PaymentMethod]
    }
    df = pd.DataFrame(data)
    categorical_columns = ['InternetService', 'Contract']
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])
    df = scaler.fit_transform(df)
    result = model.predict(df).reshape(1,-1)
    return result[0]
```

## Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/MahboobAlam0/housepriceprediction.git
   ```
2. Navigate to the project directory:
   ```sh
   cd housepriceprediction
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Contributions
Feel free to fork the repository, create a new branch, and submit a pull request for improvements!
