# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS
import glob
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
    if not models:
        return jsonify({'error': 'Models not loaded or failed to initialize'}), 500

    if not expected_cols:
        return jsonify({'error': 'Feature order could not be determined. Cannot make predictions.'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data received'}), 400

        print("Received data:", data)

        default_data = {}
        for col in expected_cols:
            if 'rank' in col:
                default_data[col] = 1000
            elif 'seed' in col:
                default_data[col] = 999
            elif 'age' in col:
                default_data[col] = 25
            elif 'ht' in col:
                default_data[col] = 180
            elif 'win_percentage' in col:
                default_data[col] = 0.0
            elif 'diff' in col:
                default_data[col] = 0
            elif 'ace' in col or 'df' in col:
                default_data[col] = 0
            elif 'best_of' in col:
                default_data[col] = 3
            elif 'matches_played' in col or 'wins' in col:
                default_data[col] = 0
            else:
                default_data[col] = 0 
        df = pd.DataFrame([default_data])

        player1_rank = data.get('player1_rank', 1000)
        player2_rank = data.get('player2_rank', 1000)
        player1_age = data.get('player1_age', 25)
        player2_age = data.get('player2_age', 25)
        player1_ht = data.get('player1_ht', 180)
        player2_ht = data.get('player2_ht', 180)
        

        if 'player1_rank' in expected_cols:
            df['player1_rank'] = player1_rank
        if 'player2_rank' in expected_cols:
            df['player2_rank'] = player2_rank
        if 'player1_seed' in expected_cols:
            df['player1_seed'] = data.get('player1_seed', 999)
        if 'player2_seed' in expected_cols:
            df['player2_seed'] = data.get('player2_seed', 999)
        if 'player1_age' in expected_cols:
            df['player1_age'] = player1_age
        if 'player2_age' in expected_cols:
            df['player2_age'] = player2_age
        if 'player1_ht' in expected_cols:
            df['player1_ht'] = player1_ht
        if 'player2_ht' in expected_cols:
            df['player2_ht'] = player2_ht
        if 'player1_rank_points' in expected_cols:
            df['player1_rank_points'] = data.get('player1_rank_points', 0)
        if 'player2_rank_points' in expected_cols:
            df['player2_rank_points'] = data.get('player2_rank_points', 0)
        if 'player1_ace' in expected_cols:
            df['player1_ace'] = data.get('player1_ace', 0)
        if 'player2_ace' in expected_cols:
            df['player2_ace'] = data.get('player2_ace', 0)
        if 'player1_df' in expected_cols:
            df['player1_df'] = data.get('player1_df', 0)
        if 'player2_df' in expected_cols:
            df['player2_df'] = data.get('player2_df', 0)
        
    
        if 'rank_diff' in expected_cols:
            df['rank_diff'] = player2_rank - player1_rank  
        if 'age_diff' in expected_cols:
            df['age_diff'] = player1_age - player2_age
        if 'height_diff' in expected_cols:
            df['height_diff'] = player1_ht - player2_ht
        if 'best_of' in expected_cols:
            df['best_of'] = data.get('best_of', 3)
        
        
        if 'wins_w' in expected_cols:
            df['wins_w'] = data.get('wins_w', 0)
        if 'wins_l' in expected_cols:
            df['wins_l'] = data.get('wins_l', 0)
        if 'matches_played_w' in expected_cols:
            df['matches_played_w'] = data.get('matches_played_w', 0)
        if 'matches_played_l' in expected_cols:
            df['matches_played_l'] = data.get('matches_played_l', 0)
        if 'win_percentage_w' in expected_cols:
            df['win_percentage_w'] = data.get('win_percentage_w', 0.0)
        if 'win_percentage_l' in expected_cols:
            df['win_percentage_l'] = data.get('win_percentage_l', 0.0)
            
        # categorical data
        for col, encoder in encoders.items():
            if col == 'tourney_level' and 'tourney_level' in expected_cols:
                current_val = data.get('tourney_level', 'A')
                if current_val in encoder.classes_:
                    df['tourney_level'] = encoder.transform([current_val])[0]
                else:
                    print(f"Warning: Unseen value '{current_val}' for column 'tourney_level'. Assigning default.")
                    df['tourney_level'] = 0  
                    
            elif col == 'surface' and 'surface' in expected_cols:
                current_val = data.get('surface', 'Hard')
                if current_val in encoder.classes_:
                    df['surface'] = encoder.transform([current_val])[0]
                else:
                    print(f"Warning: Unseen value '{current_val}' for column 'surface'. Assigning default.")
                    df['surface'] = 0  
                    
            elif col == 'player1_ioc' and 'player1_ioc' in expected_cols:
                current_val = data.get('player1_ioc', 'USA')
                if current_val in encoder.classes_:
                    df['player1_ioc'] = encoder.transform([current_val])[0]
                else:
                    print(f"Warning: Unseen value '{current_val}' for player1_ioc. Assigning default.")
                    df['player1_ioc'] = 0 
                    
            elif col == 'player2_ioc' and 'player2_ioc' in expected_cols:
                current_val = data.get('player2_ioc', 'USA')
                if current_val in encoder.classes_:
                    df['player2_ioc'] = encoder.transform([current_val])[0]
                else:
                    print(f"Warning: Unseen value '{current_val}' for player2_ioc. Assigning default.")
                    df['player2_ioc'] = 0  
                    
        df = df[expected_cols]
        
        # ---  Predictions for Each Model ---
        predictions = {}
        for name, model in models.items():
            try:
                prediction_val = model.predict(df)[0]
                probability_val = model.predict_proba(df)[0]
                confidence_p1_wins = float(probability_val[1])

                predictions[name] = {
                    'winner': 1 if prediction_val == 1 else 2,
                    'confidence_player1_wins': round(confidence_p1_wins, 4)
                }
                print(f"Prediction ({name}): {predictions[name]}")

            except Exception as model_e:
                print(f"Error predicting with model {name}: {model_e}")
                import traceback
                traceback.print_exc()
                predictions[name] = {'error': f'Prediction failed for {name}: {str(model_e)}'}

        return jsonify({'predictions': predictions})

    except Exception as e:
        print(f"General prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

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