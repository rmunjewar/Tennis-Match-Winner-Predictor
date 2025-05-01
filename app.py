# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and encoders
try:
    model = pickle.load(open('public/model.pkl', 'rb'))
    # Load label encoders
    encoders = {}
    import glob, os
    for file in glob.glob("public/le_*.pkl"):
        name = os.path.splitext(os.path.basename(file))[0][3:]
        encoders[name] = pickle.load(open(file, 'rb'))

    # Load feature importance
    feature_importance = pd.read_csv('public/feature_importance.csv')
    print("Model and encoders loaded successfully")
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    model = None
    encoders = {}
    feature_importance = None

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        print("Received data:", data) # Debugging

        # Convert data to DataFrame (important for consistent feature order)
        df = pd.DataFrame([data])  # Put the data in a list

        # Handle categorical encoding
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    print(f"Error encoding column {col}: {e}")
                    return jsonify({'error': f'Error encoding column {col}: {str(e)}'}), 400 # Or a more informative error

        # Ensure the DataFrame has the same columns as the training data
        expected_cols = feature_importance['feature'].tolist() if feature_importance is not None else [] # List of expected columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        for col in missing_cols:
            df[col] = 0  # Or use the mean/median from your training data

        df = df[expected_cols] # Select only the expected columns in the correct order

        # Make prediction
        prediction = model.predict(df)[0]  # Get the actual prediction (0 or 1)
        probability = model.predict_proba(df)[0][1] # Probability of Player 1 winning

        # Prepare response
        result = {
            'winner': 1 if prediction == 1 else 2,  # Convert to Player 1 or Player 2
            'confidence': float(probability),  # Convert to regular float
        }

        print("Prediction:", result)
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500  # Return error message

@app.route('/api/model-info')
def model_info():
    if feature_importance is None:
        return jsonify({'error': 'Feature importance data not loaded'}), 500

    # Return the top N features and their importances
    top_features = feature_importance.head(10).to_dict(orient='records')
    return jsonify({'top_features': top_features})

if __name__ == '__main__':
    app.run(debug=True) # Development only!  Use a production server for deployment.