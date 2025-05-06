# This is a program that learns to predict tennis match winners!
# It looks at lots of tennis matches and learns patterns about who wins and why.

import pandas as pd  # Helps us work with data in tables
import numpy as np   # Helps us do math with numbers
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier  # Our main prediction model
from sklearn.ensemble import RandomForestClassifier  # Another prediction model
from sklearn.neighbors import KNeighborsClassifier  # Yet another prediction model
from sklearn.linear_model import LogisticRegression  # One more prediction model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle  # Helps us save our trained models
import os
import glob

# Where we keep our tennis match data
data_dir = "data"

# Step 1: Load tennis match data
def load_datasets(data_dir):
    
    # Look for tennis match files on our computer
    all_files = glob.glob(os.path.join(data_dir, "atp_matches_*.csv"))

    if not all_files:
        # If no files on computer, get them from internet
        years = range(2021, 2024)
        dfs = []

        for year in years:
            url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"  
            print(f"Getting {year} matches from {url}")
            try:
                df = pd.read_csv(url)
                dfs.append(df)
                print(f"Got {len(df)} matches from {year}")
            except Exception as e:
                print(f"Oops! Couldn't get {year} data: {e}")

        if not dfs:
            print("Using just 2023 matches")
            url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"
            dfs = [pd.read_csv(url)]
    else:
        print(f"Found {len(all_files)} files on computer")
        dfs = [pd.read_csv(file) for file in all_files]

    if not dfs:
        raise Exception("No tennis match data found!")

    # Put all the matches together in one big table
    return pd.concat(dfs, ignore_index=True)

# Step 2: Clean up and prepare the data
def preprocess_data(df):

    # These are the things we want to know about each match
    columns_to_select = [
        'winner_seed', 'loser_seed',  # Tournament seed numbers
        'winner_rank', 'loser_rank',  # Player rankings
        'winner_rank_points', 'loser_rank_points',  # Ranking points
        'winner_ioc', 'loser_ioc',    # Player countries
        'surface', 'tourney_level',   # Court type and tournament type
        'winner_age', 'loser_age',    # Player ages
        'winner_ht', 'loser_ht',      # Player heights
        'w_ace', 'l_ace',            # Number of aces
        'w_df', 'l_df',              # Number of double faults
        'best_of'                     # Best of 3 or 5 sets
    ]

    # Calculate how many matches each player has won
    new_stat_cols_added = []
    if 'winner_id' in df.columns and 'loser_id' in df.columns:
        try:
            player_stats = calculate_player_stats(df)

            # Add stats for winners and losers
            player_stats_w = player_stats.add_suffix('_w')
            player_stats_l = player_stats.add_suffix('_l')

            # Add these stats to our main table
            df = pd.merge(df, player_stats_w, left_on='winner_id', right_index=True, how='left')
            df = pd.merge(df, player_stats_l, left_on='loser_id', right_index=True, how='left')

            new_stat_cols_added = list(player_stats_w.columns) + list(player_stats_l.columns)
            columns_to_select.extend(new_stat_cols_added)
            print(f"Added player stats: {new_stat_cols_added}")

        except Exception as stat_e:
             print(f"Oops! Couldn't calculate player stats: {stat_e}")

    # Pick only the columns we want to use
    existing_columns = [col for col in columns_to_select if col in df.columns]
    print(f"Using {len(existing_columns)} features")
    df_selected = df[existing_columns].copy()

    # Fill in any missing information
    
    # Fill in missing seed numbers with 999
    if 'winner_seed' in df_selected.columns:
        df_selected['winner_seed'] = df_selected['winner_seed'].fillna(999)
    if 'loser_seed' in df_selected.columns:
        df_selected['loser_seed'] = df_selected['loser_seed'].fillna(999)

    # Fill in missing player stats with 0
    for col in new_stat_cols_added:
        if col in df_selected.columns:
            df_selected[col] = df_selected[col].replace([np.inf, -np.inf], 0)
            df_selected[col] = df_selected[col].fillna(0)

    # Fill in other missing numbers with the middle value
    numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['winner_seed', 'loser_seed'] and col not in new_stat_cols_added:
            if df_selected[col].isnull().any():
                median_val = df_selected[col].median()
                df_selected[col] = df_selected[col].fillna(median_val)

    # Fill in missing text with the most common value
    categorical_cols = df_selected.select_dtypes(include=['object']).columns
    for col in categorical_cols:
         if df_selected[col].isnull().any():
             mode_val = df_selected[col].mode()
             if not mode_val.empty:
                df_selected[col] = df_selected[col].fillna(mode_val[0])
             else:
                 df_selected[col] = df_selected[col].fillna("Unknown")

    print("All done cleaning up!")
    return df_selected

# Calculate how many matches each player has won
def calculate_player_stats(df):
    # Get a list of all unique players
    player_ids = pd.concat([df['winner_id'], df['loser_id']]).dropna().unique()
    stats = pd.DataFrame(index=player_ids)

    # Count wins and losses for each player
    wins = df['winner_id'].value_counts()
    losses = df['loser_id'].value_counts()

    # Calculate total matches and win percentage
    stats['matches_played'] = wins.add(losses, fill_value=0).astype(int)
    stats['wins'] = wins.fillna(0).astype(int)
    stats['win_percentage'] = 0.0
    stats.loc[stats['matches_played'] > 0, 'win_percentage'] = \
        stats['wins'] / stats['matches_played']

    # Fill in any missing stats with 0
    stats = stats.fillna(0)

    print(f"Calculated stats for {len(stats)} players.")
    return stats

# Step 3: Create a balanced dataset
def create_balanced_dataset(df_selected):

    # Make two copies of our data
    df_winner = df_selected.copy()  # When player 1 wins
    df_winner['target'] = 1
    df_loser = df_selected.copy()   # When player 1 loses
    df_loser['target'] = 0

    # Rename columns to make it easier to understand
    winner_to_player1 = {
        'winner_seed': 'player1_seed',
        'loser_seed': 'player2_seed',
        'winner_rank': 'player1_rank',
        'loser_rank': 'player2_rank',
        'winner_rank_points': 'player1_rank_points',
        'loser_rank_points': 'player2_rank_points',
        'winner_ioc': 'player1_ioc',
        'loser_ioc': 'player2_ioc',
        'winner_age': 'player1_age',
        'loser_age': 'player2_age',
        'winner_ht': 'player1_ht',
        'loser_ht': 'player2_ht',
        'w_ace': 'player1_ace',
        'l_ace': 'player2_ace',
        'w_df': 'player1_df',
        'l_df': 'player2_df'
    }

    # Only rename columns that exist in our data
    winner_rename = {k: v for k, v in winner_to_player1.items() if k in df_winner.columns}
    df_winner = df_winner.rename(columns=winner_rename)

    # Do the same for the loser data
    loser_to_player1 = {
        'winner_seed': 'player2_seed',
        'loser_seed': 'player1_seed',
        'winner_rank': 'player2_rank',
        'loser_rank': 'player1_rank',
        'winner_rank_points': 'player2_rank_points',
        'loser_rank_points': 'player1_rank_points',
        'winner_ioc': 'player2_ioc',
        'loser_ioc': 'player1_ioc',
        'winner_age': 'player2_age',
        'loser_age': 'player1_age',
        'winner_ht': 'player2_ht',
        'loser_ht': 'player1_ht',
        'w_ace': 'player2_ace',
        'l_ace': 'player1_ace',
        'w_df': 'player2_df',
        'l_df': 'player1_df'
    }

    loser_rename = {k: v for k, v in loser_to_player1.items() if k in df_loser.columns}
    df_loser = df_loser.rename(columns=loser_rename)

    # Put both datasets together
    df_balanced = pd.concat([df_winner, df_loser], axis=0).reset_index(drop=True)

    # Remove any ID columns we don't need
    id_cols = [col for col in df_balanced.columns if 'id' in col]
    df_balanced = df_balanced.drop(columns=id_cols, errors='ignore')

    return df_balanced

# Step 4: Convert text data to numbers
def encode_and_prepare(df_balanced):

    # Find all text columns
    categorical_cols = df_balanced.select_dtypes(include=['object']).columns

    # Convert each text column to numbers
    encoders = {}
    for col in categorical_cols:
        df_balanced[col] = df_balanced[col].astype(str)
        le = LabelEncoder()
        df_balanced[col] = le.fit_transform(df_balanced[col])
        encoders[col] = le

    # Fill in any missing numbers
    for column in df_balanced.columns:
        if df_balanced[column].isnull().sum() > 0:
            if df_balanced[column].dtype == 'object':
                df_balanced[column] = df_balanced[column].fillna(df_balanced[column].mode()[0])
            else:
                df_balanced[column] = df_balanced[column].fillna(df_balanced[column].median())

    # Add some helpful features
    if 'player1_rank' in df_balanced.columns and 'player2_rank' in df_balanced.columns:
        df_balanced['rank_diff'] = df_balanced['player2_rank'] - df_balanced['player1_rank']

    if 'player1_ht' in df_balanced.columns and 'player2_ht' in df_balanced.columns:
        df_balanced['height_diff'] = df_balanced['player1_ht'] - df_balanced['player2_ht']

    if 'player1_age' in df_balanced.columns and 'player2_age' in df_balanced.columns:
        df_balanced['age_diff'] = df_balanced['player1_age'] - df_balanced['player2_age']

    return df_balanced, encoders

# Step 5: Train our prediction model
def train_model(df_balanced, model_type='decision_tree'):
    print(f"Training {model_type} model")

    # Separate features (what we know) from target (what we want to predict)
    X = df_balanced.drop(columns=['target'])
    y = df_balanced['target']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Choose which model to use
    if model_type == 'decision_tree':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier(
                random_state=42,
                min_samples_leaf=50,     # Need at least 50 matches in each leaf
                max_depth=6,             # Tree can be 6 levels deep
                min_samples_split=100,   # Need 100 matches to split a node
                class_weight='balanced',  # Make sure we don't favor one player
                criterion='entropy'       # Use entropy for better probabilities
            ))
        ])
        param_grid = {
            'classifier__criterion': ['entropy'],
            'classifier__min_samples_leaf': [30, 50, 70],
            'classifier__max_depth': [5, 6, 7],
            'classifier__min_samples_split': [80, 100, 120]
        }

    elif model_type == 'random_forest':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7, None],
            'classifier__min_samples_split': [2, 5, 10]
        }

    elif model_type == 'knn':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 10],
            'classifier__weights': ['uniform', 'distance']
        }

    elif model_type == 'logistic_regression':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
        ])
        param_grid = {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.1, 1.0, 10.0]
        }

    else:
        raise ValueError(f"Don't know how to train {model_type} model!")

    # Find the best settings for our model
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )

    # Train the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # See how well our model works
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy ({model_type}): {accuracy:.4f}")

    # Check if our model works well on different sets of matches
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy ({model_type}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Find out which features are most important
    feature_importance = None
    if hasattr(best_model['classifier'], 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 important features:")
        print(feature_importance.head(10))
    elif hasattr(best_model['classifier'], 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model['classifier'].coef_[0]
        }).sort_values('importance', ascending=False)
        print("\nTop 10 important features:")
        print(feature_importance.head(10))

    return best_model, feature_importance

# Main program
def main():
    # Create a folder to save our models if it doesn't exist
    os.makedirs('public', exist_ok=True)

    try:
        # Step 1: Load the data
        combined_df = load_datasets(data_dir)
        print(f"Loaded {len(combined_df)} total matches")

        # Step 2: Clean up the data
        df_selected = preprocess_data(combined_df)
        print(f"Selected features for {len(df_selected)} matches")

        # Step 3: Create balanced dataset
        df_balanced = create_balanced_dataset(df_selected)
        print(f"Created balanced dataset with {len(df_balanced)} examples")

        # Step 4: Convert text to numbers
        df_balanced, encoders = encode_and_prepare(df_balanced)

        # Step 5: Train different types of models
        model_types = ['decision_tree', 'random_forest', 'knn', 'logistic_regression']

        for model_type in model_types:
            model, feature_importance = train_model(df_balanced, model_type=model_type)

            # Save our trained model
            pickle.dump(model, open(f'public/model_{model_type}.pkl', 'wb'))

            # Save which features are most important
            if feature_importance is not None:
                feature_importance.to_csv(f'public/feature_importance_{model_type}.csv', index=False)

            # Save our text-to-number converters
            if model_type == 'decision_tree':
                for name, encoder in encoders.items():
                    pickle.dump(encoder, open(f'public/le_{name}.pkl', 'wb'))

            print(f"Model ({model_type}) trained and saved successfully!")

    except Exception as e:
        print(f"Oops! Something went wrong: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()