<<<<<<< HEAD
# Diabetes Prediction System

This project implements a machine learning-based diabetes prediction system with a web interface. It includes model training, cross-validation, and a user-friendly web application for predictions.

## Features

- Data preprocessing and cleaning
- Model training with Random Forest Classifier
- Cross-validation for model evaluation
- Learning curve analysis
- Bias-variance tradeoff visualization
- Web interface for predictions
- Model persistence using pickle

## Setup

1. Install Python 3.8 or higher
2. Install pipenv:
   ```bash
   pip install pipenv
   ```
3. Install dependencies:
   ```bash
   pipenv install
   ```
4. Download the diabetes dataset (diabetes.csv) and place it in the AppModelTraining directory

## Usage

1. Train the model:
   ```bash
   cd AppModelTraining
   python model_training.py
   ```
   This will:
   - Preprocess the data
   - Train the model
   - Perform cross-validation
   - Generate learning curves
   - Save the trained model

2. Run the web application:
   ```bash
   cd ..
   python diabetespredictor/app.py
   ```
3. Open your browser and navigate to `http://localhost:5000`

## Model Training Details

The model training process includes:
- Data cleaning and preprocessing
- Feature scaling
- Train-test split (80-20)
- 5-fold cross-validation
- Learning curve analysis
- Model evaluation metrics

## Web Interface

The web interface provides:
- Input fields for all relevant features
- Real-time predictions
- Probability scores
- Clean and responsive design

## Model Performance

The model's performance is evaluated using:
- Accuracy
- Cross-validation scores
- Learning curves
- Classification report

## Data Leakage Prevention

To prevent data leakage:
- Feature scaling is performed after train-test split
- Cross-validation is performed on the training set
- Test set is only used for final evaluation 
=======
# ProjectFInal
>>>>>>> 30b64c3a5ad229f75b7bdd89ca20c9f84180a073
