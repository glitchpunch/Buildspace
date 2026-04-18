# Nexus FinTech - Fraud Detection System

A machine learning-based fraud detection system for financial transactions, built with **Random Forest Classifier** and served via a **Flask** API with a web frontend.

## 🎯 Project Overview

This project detects fraudulent transactions in real-time using various behavioral, financial, and device-related features. The model is trained on a synthetic fraud dataset and provides both a prediction (fraud/not fraud) and explainability through feature importance weights.

## 📁 Project Structure
nexus-fraud-detection/
├── train_model.py              # Script to train and save the model
├── main.py                     # Flask backend API
├── nexus_fintech_app.html      # Frontend web interface
├── nexus_fraud_dataset.csv     # Training dataset
├── fraud_model.joblib          # Trained model (generated)
├── model_columns.joblib        # Feature columns (generated)
└── README.md

## ✨ Features

- **Random Forest Classifier** with balanced class weights
- Real-time fraud prediction API
- Feature importance explainability
- Handles categorical encoding (transaction_type)
- Responsive web interface
- CORS enabled for frontend integration

## 🛠️ Technologies Used

- **Python**
- **scikit-learn** (RandomForestClassifier)
- **Flask** + **Flask-CORS**
- **Pandas** & **NumPy**
- **Joblib** (model serialization)
- **HTML/CSS/JavaScript** (Frontend)

## 🚀 How to Run

### 1. Train the Model (if needed)

```bash
python train_model.py

This will:

Load the dataset
Preprocess features
Train the Random Forest model
Save fraud_model.joblib and model_columns.joblib```

