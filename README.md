# Cancer Mortality Prediction Models

This project implements two predictive models (Linear Regression and Deep Neural Network)
on the **cancer_reg.csv** dataset.

## Requirements
Install the dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow joblib
```

## Files
- `linear_regression.py` → Train & test Linear Regression model
- `dnn_model.py` → Train & test Deep Neural Network model
- `cancer_reg.csv` → Input dataset (must be in the same folder)

## Usage
### 1. Linear Regression
Train and test:
```bash
python linear_regression.py
```

### 2. Deep Neural Network
Train and test:
```bash
python dnn_model.py
```

## Notes
- Both scripts save trained models (`linear_regression.pkl` and `dnn_64_32_16.h5`).
- Each script has a `test_model` function to reload the trained model and make predictions.
- Ensure that `cancer_reg.csv` is present before running.
