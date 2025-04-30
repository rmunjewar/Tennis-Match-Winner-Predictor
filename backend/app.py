import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Print debug information
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir()}")

# Try to load model directly
try:
    print("Attempting to load model.pkl directly...")
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Successfully loaded model and scaler!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Model not found in current directory. Trying absolute paths...")
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {current_dir}")
    
    try:
        model_path = os.path.join(current_dir, 'model.pkl')
        scaler_path = os.path.join(current_dir, 'scaler.pkl')
        print(f"Trying to load from: {model_path}")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Successfully loaded model and scaler from script directory!")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise FileNotFoundError("Could not find model.pkl in any expected location. Please run save_model.py first.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Create a dataframe with the required feature columns
        features = pd.DataFrame({
            'player1_seed': [data.get('player1_seed', np.nan)],
            'player2_seed': [data.get('player2_seed', np.nan)],
            'player1_rank': [data.get('player1_rank', np.nan)],
            'player2_rank': [data.get('player2_rank', np.nan)],
            'player1_ioc': [data.get('player1_ioc', np.nan)],
            'surface': [data.get('surface', np.nan)],
            'player1_age': [data.get('player1_age', np.nan)],
            'player2_age': [data.get('player2_age', np.nan)],
            'tourney_level': [data.get('tourney_level', np.nan)],
            'player1_ht': [data.get('player1_ht', np.nan)],
            'player2_ht': [data.get('player2_ht', np.nan)]
        })
        
        # Handle missing values
        for column in features.columns:
            if features[column].isnull().any():
                if features[column].dtype == 'object':
                    features[column] = features[column].fillna('Unknown')
                else:
                    features[column] = features[column].fillna(0)
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        probability = model.predict_proba(scaled_features)[0][1]  # Probability of player1 winning
        
        return jsonify({
            'prediction': int(prediction[0]),
            'win_probability': float(probability),
            'player1_win': bool(prediction[0] == 1)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True)