import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression  # init model i started with - delete later
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
import os
import glob

# dir containing the data files
data_dir = "data"

# --------------------------
# Load and combine datasets from 2021-2024
# --------------------------
def load_datasets(data_dir):
    print("Loading datasets...")
    all_files = glob.glob(os.path.join(data_dir, "atp_matches_*.csv"))

    if not all_files:
        print("No local data files found. Fetching from GitHub...")
        years = range(2021, 2024)
        dfs = []

        for year in years:
            url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"  
            print(f"Fetching {year} data from {url}")
            try:
                df = pd.read_csv(url)
                dfs.append(df)
                print(f"Successfully loaded {year} data: {len(df)} matches")
            except Exception as e:
                print(f"Error loading {year} data: {e}")

        if not dfs:
            print("Falling back to 2023 data only")
            url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"
            dfs = [pd.read_csv(url)]
    else:
        print(f"Found {len(all_files)} local data files")
        dfs = [pd.read_csv(file) for file in all_files]

    if not dfs:
        raise Exception("No data could be loaded!")

    return pd.concat(dfs, ignore_index=True)

# --------------------------
# Feature Engineering and Preprocessing
# --------------------------
def preprocess_data(df):
    print("Preprocessing data...")

    # Define the initial set of features we want
    columns_to_select = [
        'winner_seed', 'loser_seed',
        'winner_rank', 'loser_rank',
        'winner_rank_points', 'loser_rank_points',
        'winner_ioc', 'loser_ioc',
        'surface', 'tourney_level',
        'winner_age', 'loser_age',
        'winner_ht', 'loser_ht',
        'w_ace', 'l_ace',  # added aces
        'w_df', 'l_df',  # added double faults
        'best_of'  # added match format (3 or 5 sets)
    ]

    # --- Player Stats Calculation and Merging ---
    # Check if IDs exist in the original DataFrame to calculate stats
    new_stat_cols_added = [] # Keep track of columns added by merge
    if 'winner_id' in df.columns and 'loser_id' in df.columns:
        print("Calculating and merging player stats...")
        try:
            player_stats = calculate_player_stats(df) # Columns: matches_played, wins, win_percentage

            # Create explicit column names for merging to avoid conflicts
            player_stats_w = player_stats.add_suffix('_w') # e.g., win_percentage_w
            player_stats_l = player_stats.add_suffix('_l') # e.g., win_percentage_l

            # Merge winner stats into the main df
            df = pd.merge(df, player_stats_w, left_on='winner_id', right_index=True, how='left')
            # Merge loser stats into the main df
            df = pd.merge(df, player_stats_l, left_on='loser_id', right_index=True, how='left')

            # Get the list of the stat columns actually added
            new_stat_cols_added = list(player_stats_w.columns) + list(player_stats_l.columns)

            # Add these new columns to the list of columns we eventually want to select
            columns_to_select.extend(new_stat_cols_added)
            print(f"Added player stats columns: {new_stat_cols_added}")

        except Exception as stat_e:
             print(f"Error during player stats calculation/merge: {stat_e}")
             # Continue without stats if calculation fails
    else:
        print("Skipping player stats calculation: 'winner_id' or 'loser_id' not found.")

    # --- Feature Selection ---
    # Select only the columns that actually exist in the (potentially augmented) df
    existing_columns = [col for col in columns_to_select if col in df.columns]
    print(f"Selecting {len(existing_columns)} features...")
    # Create the final df_selected with the chosen features
    df_selected = df[existing_columns].copy()

    # --- Imputation (now performed on df_selected) ---
    print("Starting imputation on selected features...")
    # Fill NaNs for seeds
    if 'winner_seed' in df_selected.columns:
        df_selected['winner_seed'] = df_selected['winner_seed'].fillna(999)
    if 'loser_seed' in df_selected.columns:
        df_selected['loser_seed'] = df_selected['loser_seed'].fillna(999)

    # Fill NaNs for the added stat columns (if any player had 0 matches, stats would be NaN/inf)
    for col in new_stat_cols_added:
        if col in df_selected.columns:
            # Replace potential Infinities resulting from division by zero in stats
            df_selected[col] = df_selected[col].replace([np.inf, -np.inf], 0)
            # Fill remaining NaNs (e.g., for players not found in merge) with 0
            df_selected[col] = df_selected[col].fillna(0)

    # Fill other numeric features with median
    numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Avoid re-filling columns already handled (seeds, stats)
        if col not in ['winner_seed', 'loser_seed'] and col not in new_stat_cols_added:
            # Check if column still has NaNs before filling
            if df_selected[col].isnull().any():
                median_val = df_selected[col].median()
                df_selected[col] = df_selected[col].fillna(median_val)

    # Fill categorical features with mode
    categorical_cols = df_selected.select_dtypes(include=['object']).columns
    for col in categorical_cols:
         if df_selected[col].isnull().any():
             mode_val = df_selected[col].mode()
             # Handle cases where mode() might return multiple values or empty
             if not mode_val.empty:
                df_selected[col] = df_selected[col].fillna(mode_val[0])
             else:
                 # Fallback if mode is empty (e.g., all values are NaN)
                 df_selected[col] = df_selected[col].fillna("Unknown") # Or another placeholder


    print("Imputation complete.")
    # Final check for NaNs
    if df_selected.isnull().sum().sum() > 0:
        print("Warning: NaNs still present after imputation:")
        print(df_selected.isnull().sum()[df_selected.isnull().sum() > 0])


    return df_selected

# You also need the calculate_player_stats function (make sure it's correctly defined)
def calculate_player_stats(df):
    # Combine winner and loser IDs to find all unique players
    player_ids = pd.concat([df['winner_id'], df['loser_id']]).dropna().unique()
    stats = pd.DataFrame(index=player_ids)

    # Calculate wins and losses for each player
    wins = df['winner_id'].value_counts()
    losses = df['loser_id'].value_counts()

    # Combine to get total matches played
    stats['matches_played'] = wins.add(losses, fill_value=0).astype(int) # Ensure integer type
    stats['wins'] = wins.fillna(0).astype(int) # Fill NaN for players with 0 wins

    # Calculate win percentage, handle division by zero
    # Use .loc to avoid SettingWithCopyWarning if stats was a slice
    stats['win_percentage'] = 0.0 # Initialize with float
    # Calculate only where matches_played > 0
    stats.loc[stats['matches_played'] > 0, 'win_percentage'] = \
        stats['wins'] / stats['matches_played']

    # Fill any remaining NaNs in stats DataFrame (e.g., if a player ID was NaN initially)
    stats = stats.fillna(0)

    print(f"Calculated stats for {len(stats)} players.")
    return stats

# --------------------------
# Create balanced dataset with winner and loser data
# --------------------------
def create_balanced_dataset(df_selected):
    print("Creating balanced dataset...")

    # create winner dataframe with target=1
    df_winner = df_selected.copy()
    df_winner['target'] = 1

    # create loser dataframe with target=0
    df_loser = df_selected.copy()
    df_loser['target'] = 0

    # rename columns to player1 and player2 format
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

    # only rename columns that exist in our dataset
    winner_rename = {k: v for k, v in winner_to_player1.items() if k in df_winner.columns}
    df_winner = df_winner.rename(columns=winner_rename)

    # do the same for loser to player1 mapping
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

    # combine datasets
    df_balanced = pd.concat([df_winner, df_loser], axis=0).reset_index(drop=True)

    # drop player ID columns if they exist
    id_cols = [col for col in df_balanced.columns if 'id' in col]
    df_balanced = df_balanced.drop(columns=id_cols, errors='ignore')


    return df_balanced

# --------------------------
# Encode Categorical Variables
# --------------------------
def encode_and_prepare(df_balanced):
    print("Encoding categorical variables...")

    # find categorical columns
    categorical_cols = df_balanced.select_dtypes(include=['object']).columns

    # create label encoders for each categorical column
    encoders = {}
    for col in categorical_cols:
        df_balanced[col] = df_balanced[col].astype(str)
        le = LabelEncoder()
        df_balanced[col] = le.fit_transform(df_balanced[col])
        encoders[col] = le

    # final preprocessing
    for column in df_balanced.columns:
        if df_balanced[column].isnull().sum() > 0:
            if df_balanced[column].dtype == 'object':
                df_balanced[column] = df_balanced[column].fillna(df_balanced[column].mode()[0])
            else:
                df_balanced[column] = df_balanced[column].fillna(df_balanced[column].median())

    # add feature: Rank difference
    if 'player1_rank' in df_balanced.columns and 'player2_rank' in df_balanced.columns:
        df_balanced['rank_diff'] = df_balanced['player2_rank'] - df_balanced['player1_rank']

    # add feature: Height difference
    if 'player1_ht' in df_balanced.columns and 'player2_ht' in df_balanced.columns:
        df_balanced['height_diff'] = df_balanced['player1_ht'] - df_balanced['player2_ht']

    # add feature: Age difference
    if 'player1_age' in df_balanced.columns and 'player2_age' in df_balanced.columns:
        df_balanced['age_diff'] = df_balanced['player1_age'] - df_balanced['player2_age']

    return df_balanced, encoders

# --------------------------
# Train Model
# --------------------------
def train_model(df_balanced, model_type='decision_tree'):
    print(f"Training model: {model_type}...")

    # preapre features and target
    X = df_balanced.drop(columns=['target'])
    y = df_balanced['target']

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # create a pipeline with scaling and model
    if model_type == 'decision_tree':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', DecisionTreeClassifier(
                random_state=42,
                min_samples_leaf=50,  # Require more samples in leaf nodes
                max_depth=5,          # Limit tree depth to prevent overfitting
                min_samples_split=100  # Require more samples before splitting
            ))
        ])
        param_grid = {
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__min_samples_leaf': [30, 50, 70],
            'classifier__max_depth': [4, 5, 6],
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

    elif model_type == 'logistic_regression':  # LR kept for funsies can be removed later
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, solver='liblinear'))  # added solver
        ])
        param_grid = {
            'classifier__penalty': ['l1', 'l2'],
            'classifier__C': [0.1, 1.0, 10.0]
        }


    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # gridsearchcv
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )

    # fit the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # evalute model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy ({model_type}): {accuracy:.4f}")

    # cross-validation score
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy ({model_type}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # get important features (using feature_importances_ for tree-based models)
    feature_importance = None
    if hasattr(best_model['classifier'], 'feature_importances_'):  #for Decision Trees and Random Forests
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model['classifier'].feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 important features:")
        print(feature_importance.head(10))
    elif hasattr(best_model['classifier'], 'coef_'):  #for Logistic Regression
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model['classifier'].coef_[0]  # use the first row of coefficients
        }).sort_values('importance', ascending=False)
        print("\nTop 10 important features:")
        print(feature_importance.head(10))

    return best_model, feature_importance


# --------------------------
# Main Function
# --------------------------
def main():
    # create output directory if it doesn't exist
    os.makedirs('public', exist_ok=True)

    # load data
    try:
        combined_df = load_datasets(data_dir)
        print(f"Loaded {len(combined_df)} total matches")

        # preprocess data
        df_selected = preprocess_data(combined_df)
        print(f"Selected features for {len(df_selected)} matches")

        # create balanced dataset
        df_balanced = create_balanced_dataset(df_selected)
        print(f"Created balanced dataset with {len(df_balanced)} examples")

        # encode categorical variables
        df_balanced, encoders = encode_and_prepare(df_balanced)

        # train models
        model_types = ['decision_tree', 'random_forest', 'knn', 'logistic_regression']  # Add logistic regression

        for model_type in model_types:
            model, feature_importance = train_model(df_balanced, model_type=model_type)

            # save model and encoders
            pickle.dump(model, open(f'public/model_{model_type}.pkl', 'wb'))

            # save feature importance if available
            if feature_importance is not None:
                feature_importance.to_csv(f'public/feature_importance_{model_type}.csv', index=False)

            # save label encoders (only save once, as they are the same for all models)
            if model_type == 'decision_tree':
                for name, encoder in encoders.items():
                    pickle.dump(encoder, open(f'public/le_{name}.pkl', 'wb'))

            print(f"Model ({model_type}) trained and saved successfully!")

    except Exception as e:
        print(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()