# PMOC-SMOC Classifier Model

A machine learning pipeline for classifying different types of ovarian cancer (PMOC, SMOC) and their origins using DNA methylation data.  

## Overview
This repository contains machine learning models designed to differentiate between:
- Primary Mucinous Ovarian Carcinoma (PMOC)
- Secondary Mucinous Ovarian Carcinoma (SMOC)
- SMOC origin classification

The models utilize DNA methylation data and employ both Logistic Regression and Support Vector Machine (SVM) classifiers with Principal Component Analysis (PCA) for dimensionality reduction.

## Project Structure
```
.
├── data/                    # Data files and annotations
│   ├── EPIC/               # EPIC array data
│   ├── EPICv2/             # EPIC v2 array data
│   ├── all_sample_anno.csv  # Sample annotations
│   ├── feature_cpg_list_*.json  # Feature lists for different steps
│   └── step_*_*.csv        # Training and test data splits
├── models/                  # Trained model files
│   ├── lr_model_*.pkl      # Logistic Regression models
│   ├── pca_model_*_*.pkl   # PCA transformation models
│   └── svm_model_*.pkl     # SVM models
├── data_prep.ipynb         # Data preprocessing and feature selection
├── pipeline_training.ipynb # Model training and optimization
└── model_eval.ipynb        # Model evaluation and results
```


## Data Preparation

The `data_prep.ipynb` notebook handles:
- Loading and preprocessing DNA methylation data
- Feature selection using mutual information
- Data splitting into training and test sets
- Saving processed data for model training

## Model Training

The `pipeline_training.ipynb` notebook includes:
- Implementation of Logistic Regression and SVM classifiers
- Hyperparameter optimization using Optuna
- Model training with cross-validation
- Saving trained models for inference

## Evaluation

The `model_eval.ipynb` notebook provides:
- Model performance metrics (accuracy, confusion matrices)
- Comparison between different models
- Visualization of results
