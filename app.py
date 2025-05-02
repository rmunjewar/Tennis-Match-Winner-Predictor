# app.py
from flask import Flask, request, jsonify
import pickle
import traceback
import pandas as pd
import numpy as np
from flask_cors import CORS
import glob
from sklearn.pipeline import Pipeline
import os

app = Flask(__name__)
CORS(app)  

models = {}
encoders = {}
expected_cols = []
model_names = ['decision_tree', 'random_forest', 'knn']  # models to load

try:
    print("Loading resources...")
    # load label encoders
    encoder_files = glob.glob("public/le_*.pkl")
    if not encoder_files:
        print("Warning: No encoder files found (public/le_*.pkl). Encoding might fail.")
    for file in encoder_files:
        name = os.path.splitext(os.path.basename(file))[0][3:]
        try:
            with open(file, 'rb') as f:
                encoders[name] = pickle.load(f)
            print(f"Loaded encoder: {name}")
        except Exception as e:
            print(f"Error loading encoder {file}: {e}")

    # load models
    for name in model_names:
        model_path = f'public/model_{name}.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
                print(f"Loaded model: {name}")
            except Exception as e:
                print(f"Error loading model {model_path}: {e}")
        else:
            print(f"Warning: Model file not found: {model_path}")

    # load expected feature order (using RandomForest's importance file as reference)
    feature_file = 'public/feature_importance_random_forest.csv'
    if os.path.exists(feature_file):
        try:
            feature_importance_df = pd.read_csv(feature_file)
            if 'feature' in feature_importance_df.columns:
                expected_cols = feature_importance_df['feature'].tolist()
                print(f"Loaded expected feature order ({len(expected_cols)} features) from {feature_file}")
                print("Expected columns:", expected_cols)
            else:
                print(f"Warning: 'feature' column not found in {feature_file}. Cannot determine feature order.")
        except Exception as e:
            print(f"Error reading feature importance file {feature_file}: {e}")
    else:
        print(f"Warning: Feature importance file '{feature_file}' not found. Feature order might be incorrect.")

    if not models:
        print("Error: No models were successfully loaded. Predictions will fail.")
    if not encoders:
        print("Warning: No encoders loaded. Categorical feature processing might fail.")
    if not expected_cols and models:
        print("CRITICAL WARNING: Could not determine feature order. Model predictions might be inaccurate or fail.")

except Exception as e:
    print(f"Critical error during initialization: {e}")

# --- API Endpoints ---

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")

        # Create DataFrame with a single row for prediction
        df = pd.DataFrame([data])
        
        # Calculate derived features that were used during training
        # Age difference
        if 'player1_age' in df.columns and 'player2_age' in df.columns:
            df['age_diff'] = df['player1_age'] - df['player2_age']
            
        # Height difference
        if 'player1_ht' in df.columns and 'player2_ht' in df.columns:
            df['height_diff'] = df['player1_ht'] - df['player2_ht']
            
        # Rank difference
        if 'player1_rank' in df.columns and 'player2_rank' in df.columns:
            df['rank_diff'] = df['player2_rank'] - df['player1_rank']
        
        # Add missing columns with default values
        # These are features that the model was trained with but aren't in the request
        required_columns = [
            'rank_diff', 'wins_w', 'wins_l', 'matches_played_w', 'matches_played_l', 
            'win_percentage_w', 'win_percentage_l', 'player2_rank_points', 'player1_rank', 
            'player1_rank_points', 'player2_rank', 'player2_ace', 'player1_ace', 'age_diff', 
            'player2_age', 'player1_age', 'player2_df', 'player1_df', 'player1_ioc', 
            'player2_ioc', 'height_diff', 'player1_ht', 'player2_ht', 'player2_seed', 
            'player1_seed', 'tourney_level', 'surface', 'best_of'
        ]
        
        # Add any missing columns with default values
        for col in required_columns:
            if col not in df.columns:
                if col in ['wins_w', 'wins_l', 'matches_played_w', 'matches_played_l']:
                    df[col] = 0
                elif col in ['win_percentage_w', 'win_percentage_l']:
                    df[col] = 0.5
                elif col in ['player1_rank_points', 'player2_rank_points']:
                    df[col] = 0
                elif col in ['player1_ace', 'player2_ace', 'player1_df', 'player2_df']:
                    df[col] = 0
                elif col == 'best_of':
                    # Default to 3 sets for most tournaments, 5 for grand slams
                    df[col] = 5 if df.get('tourney_level', 'G').iloc[0] == 'G' else 3
        
        # Make sure column names match exactly what the model expects
        # Replace any variations in column names
        if 'tourney_level' in df.columns:
            df.rename(columns={'tourney_level': 'tourney_level'}, inplace=True)
            
        # Encode categorical variables
        categorical_cols = ['player1_ioc', 'player2_ioc', 'surface', 'tourney_level']
        for col in categorical_cols:
            if col in df.columns:
                # Load the corresponding label encoder
                encoder_path = f'public/le_{col}.pkl'
                if os.path.exists(encoder_path):
                    encoder = pickle.load(open(encoder_path, 'rb'))
                    df[col] = df[col].astype(str)  # Ensure string type for encoding
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError as e:
                        # Handle values not seen during training
                        print(f"Warning: Value not in encoder for {col}: {df[col].values[0]}")
                        # Use a default value (typically the most common category)
                        df[col] = 0
        
        # Make predictions with each model
        results = {}
        
        for model_name in model_names:
            try:
                # Get the model from our loaded models dictionary
                model = models.get(model_name)
                if model is None:
                    raise ValueError(f"Model {model_name} not found in loaded models")
                
                # Get feature names from the model
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                else:
                    # For pipelines, the feature names are in the first step
                    feature_names = model.steps[0][1].feature_names_in_
                
                # Reorder columns to match training data
                prediction_df = pd.DataFrame(index=df.index)
                for feature in feature_names:
                    if feature in df.columns:
                        prediction_df[feature] = df[feature]
                    else:
                        # If feature is missing, add with default values
                        prediction_df[feature] = 0
                
                # Make prediction
                prediction_val = model.predict(prediction_df)[0]
                probability = model.predict_proba(prediction_df)[0]
                
                # Log the prediction details for debugging
                print(f"\nPrediction details for {model_name}:")
                print(f"Input features: {prediction_df.to_dict('records')[0]}")
                print(f"Prediction value: {prediction_val}")
                print(f"Probabilities: {probability}")
                
                results[model_name] = {
                    'prediction': int(prediction_val),
                    'probability': {
                        'player1_wins': float(probability[1]),
                        'player2_wins': float(probability[0])
                    }
                }
                
            except Exception as e:
                print(f"Error predicting with model {model_name}: {str(e)}")
                traceback.print_exc()
                results[model_name] = {'error': str(e)}
        
        # Combine results from all models
        response = {
            'predictions': results,
            'player1_name': data.get('player1_name', 'Player 1'),
            'player2_name': data.get('player2_name', 'Player 2'),
            'match_details': {
                'surface': data.get('surface', 'Unknown'),
                'tournament': data.get('tourney_level', 'Unknown')
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in prediction endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info')
def model_info():
    feature_file = 'public/feature_importance_random_forest.csv'
    if os.path.exists(feature_file):
        try:
            feature_importance_df = pd.read_csv(feature_file)
            if 'feature' in feature_importance_df.columns:
                top_features = feature_importance_df.head(10).to_dict(orient='records')
                return jsonify({'top_features_random_forest': top_features}) 
            else:
                return jsonify({'error': "'feature' column missing in importance file"}), 500
        except Exception as e:
            return jsonify({'error': f'Could not read feature importance: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Feature importance data (random_forest) not found'}), 500


if __name__ == '__main__':
    # running on port 5001
    app.run(debug=True, host='0.0.0.0', port=5001)