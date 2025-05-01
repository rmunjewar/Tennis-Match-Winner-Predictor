# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
from flask_cors import CORS  # Import Flask-CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# load the model and encoders
try:
    model = pickle.load(open('public/model.pkl', 'rb'))
    # load label encoders
    encoders = {}
    import glob, os
    for file in glob.glob("public/le_*.pkl"):
        name = os.path.splitext(os.path.basename(file))[0][3:]
        encoders[name] = pickle.load(open(file, 'rb'))

    # load feature importance
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
        print("Received data:", data) 

        # convert data to DataFrame (important for consistent feature order)
        df = pd.DataFrame([data])  # Put the data in a list

        # handle categorical encoding
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    print(f"Error encoding column {col}: {e}")
                    return jsonify({'error': f'Error encoding column {col}: {str(e)}'}), 400 

        # ensure the DataFrame has the same columns as the training data
        expected_cols = feature_importance['feature'].tolist() if feature_importance is not None else [] # List of expected columns
        missing_cols = [col for col in expected_cols if col not in df.columns]
        for col in missing_cols:
            df[col] = 0  

        df = df[expected_cols] 

   
        prediction = model.predict(df)[0]  
        probability = model.predict_proba(df)[0][1] 


        result = {
            'winner': 1 if prediction == 1 else 2,  
            'confidence': float(probability), 
        }

        print("Prediction:", result)
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500  

@app.route('/api/model-info')
def model_info():
    if feature_importance is None:
        return jsonify({'error': 'Feature importance data not loaded'}), 500

    top_features = feature_importance.head(10).to_dict(orient='records')
    return jsonify({'top_features': top_features})

if __name__ == '__main__':
    app.run(debug=True) # Development only!  Use a production server for deployment. // need to look at later