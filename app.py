# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model, scaler, and label encoders
try:
    model = pickle.load(open('public/model.pkl', 'rb'))
    scaler = pickle.load(open('public/scaler.pkl', 'rb'))
    le_surface = pickle.load(open('public/le_surface.pkl', 'rb'))
    le_ioc = pickle.load(open('public/le_ioc.pkl', 'rb'))
    le_level = pickle.load(open('public/le_level.pkl', 'rb'))

    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    le_surface = None
    le_ioc = None
    le_level = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract data from the request
        player1_seed = data['player1_seed']
        player1_rank = data['player1_rank']
        player1_ioc = data['player1_ioc']
        player1_age = data['player1_age']
        player1_ht = data['player1_ht']

        player2_seed = data['player2_seed']
        player2_rank = data['player2_rank']
        player2_ioc = data['player2_ioc']
        player2_age = data['player2_age']
        player2_ht = data['player2_ht']

        surface = data['surface']
        tourney_level = data['tourney_level']

        # Preprocess the input data
        input_data = {
            'player1_seed': player1_seed,
            'player1_rank': player1_rank,
            'player1_ioc': player1_ioc,
            'player1_age': player1_age,
            'player1_ht': player1_ht,
            'player2_seed': player2_seed,
            'player2_rank': player2_rank,
            'player2_ioc': player2_ioc,
            'player2_age': player2_age,
            'player2_ht': player2_ht,
            'surface': surface,
            'tourney_level': tourney_level
        }

        # Convert categorical variables to numerical
        input_data['surface'] = le_surface.transform([input_data['surface']])[0]
        input_data['player1_ioc'] = le_ioc.transform([input_data['player1_ioc']])[0]
        input_data['player2_ioc'] = le_ioc.transform([input_data['player2_ioc']])[0]
        input_data['tourney_level'] = le_level.transform([input_data['tourney_level']])[0]

        # Convert input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Scale the input data using the pre-trained scaler
        scaled_input = scaler.transform(input_df)

        # Make the prediction
        prediction = model.predict(scaled_input)[0]
        confidence = np.max(model.predict_proba(scaled_input))

        # Return the prediction and confidence
        return jsonify({'winner': int(prediction), 'confidence': float(confidence)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)