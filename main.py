from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the model and columns
model_path = 'fraud_model.joblib'
columns_path = 'model_columns.joblib'

if os.path.exists(model_path) and os.path.exists(columns_path):
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
else:
    model = None
    model_columns = None

@app.route('/')
def serve_frontend():
    return send_file('nexus_fintech_app.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
        
    try:
        data = request.json
        
        # Convert JSON into DataFrame matching model_columns
        df_input = pd.DataFrame([data])
        
        # Ensure all columns are present, fill missing with 0
        for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0
                
        # Reorder columns to match the training data exactly
        df_input = df_input[model_columns]
        
        # Get prediction and probabilities
        prediction = model.predict(df_input)[0]
        prediction_proba = model.predict_proba(df_input)[0].tolist()
        
        # Extract feature importances
        importances = model.feature_importances_
        
        # Create a list of dictionaries with feature names and their weights
        # Filter out zero-weight features and sort by weight descending
        features_weights = []
        for name, weight in zip(model_columns, importances):
            if weight > 0:
                features_weights.append({
                    'feature': name,
                    'weight': float(weight),
                    'value': float(df_input[name].iloc[0])
                })
                
        features_weights = sorted(features_weights, key=lambda x: x['weight'], reverse=True)
        
        return jsonify({
            'fraud_prediction': int(prediction),
            'fraud_probability': prediction_proba[1], # Probability of class 1 (fraud)
            'feature_weights': features_weights
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
