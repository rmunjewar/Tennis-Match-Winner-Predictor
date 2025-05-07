# This is our web server that helps predict tennis match winners!

from flask import Flask, request, jsonify
import pickle
import traceback
import pandas as pd
import numpy as np
from flask_cors import CORS
import glob
from sklearn.pipeline import Pipeline
import os

# Create our web server
app = Flask(__name__)
CORS(app)  # This lets our website talk to our server

# Keep track of our trained models and tools
models = {}
encoders = {}
expected_cols = []
model_names = ['decision_tree', 'random_forest', 'knn']  # The types of models we are using

try:
    print("Loading our prediction tools...")
    # Load our text-to-number converters
    encoder_files = glob.glob("public/le_*.pkl")
    if not encoder_files:
        print("No convertor files found")
    for file in encoder_files:
        name = os.path.splitext(os.path.basename(file))[0][3:]
        try:
            with open(file, 'rb') as f:
                encoders[name] = pickle.load(f)
        except Exception as e:
            print(f" ouldn't load converter {file}: {e}")

    # Load trained models
    for name in model_names:
        model_path = f'public/model_{name}.pkl'
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                print(f"couldn't load model {model_path}: {e}")
        else:
            print(f"Model file not found - {model_path}")

    # Load the list of things we need to know about each match
    feature_file = 'public/feature_importance_random_forest.csv'
    if os.path.exists(feature_file):
        try:
            feature_importance_df = pd.read_csv(feature_file)
            if 'feature' in feature_importance_df.columns:
                expected_cols = feature_importance_df['feature'].tolist()
            else:
                print(f"couldn't find the list of match details in {feature_file}")
        except Exception as e:
            print(f"couldn't read the list of match details: {e}")
    else:
        print(f"couldn't find the list of match details file")

except Exception as e:
    print(f"something went wrong while loading: {e}")

# This is where our website sends match information to get predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get the match information from the website
        data = request.get_json()
        print(f"Got match information: {data}")

        # Put the match information in a table
        df = pd.DataFrame([data])
        
        # Calculate some helpful numbers
        # Age difference between players
        if 'player1_age' in df.columns and 'player2_age' in df.columns:
            df['age_diff'] = df['player1_age'] - df['player2_age']
            
        # Height difference between players
        if 'player1_ht' in df.columns and 'player2_ht' in df.columns:
            df['height_diff'] = df['player1_ht'] - df['player2_ht']
            
        # Rank difference between players
        if 'player1_rank' in df.columns and 'player2_rank' in df.columns:
            df['rank_diff'] = df['player2_rank'] - df['player1_rank']
        
        # Add any missing information with default values
        required_columns = [
            'rank_diff', 'wins_w', 'wins_l', 'matches_played_w', 'matches_played_l', 
            'win_percentage_w', 'win_percentage_l', 'player2_rank_points', 'player1_rank', 
            'player1_rank_points', 'player2_rank', 'player2_ace', 'player1_ace', 'age_diff', 
            'player2_age', 'player1_age', 'player2_df', 'player1_df', 'player1_ioc', 
            'player2_ioc', 'height_diff', 'player1_ht', 'player2_ht', 'player2_seed', 
            'player1_seed', 'tourney_level', 'surface', 'best_of'
        ]
        
        # Fill in missing information
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
                    df[col] = 5 if df.get('tourney_level', 'G').iloc[0] == 'G' else 3
        
        # Convert text information to numbers
        categorical_cols = ['player1_ioc', 'player2_ioc', 'surface', 'tourney_level']
        for col in categorical_cols:
            if col in df.columns:
                # Get the converter for this type of information
                encoder_path = f'public/le_{col}.pkl'
                if os.path.exists(encoder_path):
                    encoder = pickle.load(open(encoder_path, 'rb'))
                    df[col] = df[col].astype(str)  # Make sure it's text
                    try:
                        # Convert the text to a number
                        df[col] = encoder.transform(df[col])
                    except ValueError as e:
                        print(f"Warning: Couldn't convert {col}: {df[col].values[0]}")
                        df[col] = 0
                else:
                    print(f"Warning: Couldn't find converter for {col}")
                    df[col] = 0
        
        # Get predictions from each model
        results = {}
        
        for model_name in model_names:
            try:
                # Get the model
                model = models.get(model_name)
                if model is None:
                    raise ValueError(f"Couldn't find model {model_name}")
                
                # Get the list of things the model needs to know
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                else:
                    feature_names = model.steps[0][1].feature_names_in_
                
                # Make sure we have all the information the model needs
                prediction_df = pd.DataFrame(index=df.index)
                for feature in feature_names:
                    if feature in df.columns:
                        prediction_df[feature] = df[feature]
                    else:
                        prediction_df[feature] = 0
                
                # Get the prediction
                prediction_val = model.predict(prediction_df)[0]
                probability = model.predict_proba(prediction_df)[0]
                
                # Print what the model thinks
                print(f"\n{model_name}:")
                print(f"Match information: {prediction_df.to_dict('records')[0]}")
                print(f"Prediction: {prediction_val}")
                print(f"Probabilities: {probability}")
                
                results[model_name] = {
                    'prediction': int(prediction_val),
                    'probability': {
                        'player1_wins': float(probability[1]),
                        'player2_wins': float(probability[0])
                    }
                }
                
            except Exception as e:
                print(f"Oops! {model_name} had trouble predicting: {str(e)}")
                traceback.print_exc()
                results[model_name] = {'error': str(e)}
        
        # Send the predictions back to the website
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
        print(f"something went wrong: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# This tells the website which match details are most important
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
                return jsonify({'error'}), 500
        except Exception as e:
            return jsonify({'error'}), 500
    else:
        return jsonify({'error'}), 500

# Start our web server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)