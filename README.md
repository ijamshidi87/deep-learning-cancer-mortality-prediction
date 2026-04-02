# Cancer Mortality Rate Prediction

**Course**: Deep Learning — Homework 1  
**Instructor**: Professor Jun Bai  
**Author**: Iman Jamshidi  

## Overview

This project predicts cancer mortality rates (`TARGET_deathRate`) using a dataset of 3,047 U.S. county-level records with 33 socioeconomic and demographic features. A baseline Linear Regression model was compared against multiple Deep Neural Network (DNN) architectures to identify the best-performing approach.

## Dataset

- **Source**: `cancer_reg.csv`
- **Samples**: 3,047
- **Features**: 33 input features + 1 label (`TARGET_deathRate`)
- **Missing values**: handled by mean imputation
- **Split**: 70% train / 15% validation / 15% test

## Models

| Model | Test R² |
|---|---|
| Linear Regression | 0.755 |
| DNN-16 | 0.788 |
| DNN-30-8 | 0.852 |
| DNN-30-16-8 | 0.834 |
| DNN-30-16-8-4 | 0.807 |
| **DNN-64-32-16** | **0.877** ✅ |

Best model: **DNN-64-32-16** with Adam optimizer (LR = 0.001)

## Key Findings

- MSE outperformed MAE and Huber loss as the training objective
- Adam optimizer significantly outperformed SGD
- Learning rate of 0.01 was optimal for most architectures; 0.001 worked best for DNN-64-32-16
- Deeper is not always better — DNN-30-16-8-4 underperformed simpler models

## Files

| File | Description |
|---|---|
| `SURF1.ipynb` | Main Jupyter notebook with full experiments |
| `dnn_model.py` | DNN architecture and training code |
| `linear_regression.py` | Baseline linear regression implementation |
| `dnn_64_32_16.h5` | Saved best DNN model weights |
| `linear_regression.pkl` | Saved linear regression model |
| `cancer_reg.csv` | Dataset |
| `Homework_report_ImanJamshidi.pdf` | Full written report |

## Requirements
```bash
pip install tensorflow scikit-learn pandas numpy joblib
```
