import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

print("Starting model training and saving process...")

# --------------------------
# Load dataset
# --------------------------
print("Loading dataset...")
url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"
df = pd.read_csv(url)

# --------------------------
# Relevant Columns
# --------------------------
columns = [
    # most important
    'winner_seed', 'loser_seed', 
    # important
    'winner_rank', 'loser_rank', 
    'winner_ioc',
    # less important
    'surface', 'loser_age', 'winner_age', 'tourney_level',    
    'winner_ht', 'loser_ht'
]

df = df[columns].dropna()

# --------------------------
# Encode Variables
# --------------------------
print("Encoding variables...")
le_surface = LabelEncoder()
le_ioc = LabelEncoder()
le_level = LabelEncoder()
df['surface'] = le_surface.fit_transform(df['surface'])
df['winner_ioc'] = le_ioc.fit_transform(df['winner_ioc'])
df['tourney_level'] = le_level.fit_transform(df['tourney_level'])

# --------------------------
# Create Winner and Loser DFs
# --------------------------
print("Creating balanced dataset...")
df_winner = df.copy()
df_winner['target'] = 1

df_loser = df.copy()
df_loser['target'] = 0

# columns renamed for player1 vs player2 format
df_winner = df_winner.rename(columns={
    'winner_seed': 'player1_seed', 'loser_seed': 'player2_seed',
    'winner_rank': 'player1_rank', 'loser_rank': 'player2_rank',
    'winner_ioc': 'player1_ioc', 'surface': 'surface',
    'winner_age': 'player1_age', 'loser_age': 'player2_age',
    'tourney_level': 'tourney_level',
    'winner_ht': 'player1_ht', 'loser_ht': 'player2_ht'
})

df_loser = df_loser.rename(columns={
    'winner_seed': 'player2_seed', 'loser_seed': 'player1_seed',
    'winner_rank': 'player2_rank', 'loser_rank': 'player1_rank',
    'winner_ioc': 'player2_ioc', 'surface': 'surface',
    'winner_age': 'player2_age', 'loser_age': 'player1_age',
    'tourney_level': 'tourney_level',
    'winner_ht': 'player2_ht', 'loser_ht': 'player1_ht'
})

df_winner = df_winner.dropna(thresh=int(0.9 * df_winner.shape[1]))
df_loser = df_loser.dropna(thresh=int(0.9 * df_loser.shape[1]))

# --------------------------
# Combine Winner and Loser into one dataset
# --------------------------
df_balanced = pd.concat([df_winner, df_loser], axis=0).reset_index(drop=True)
df_balanced = df_balanced.drop(columns=[col for col in ['player1_id', 'player2_id'] if col in df_balanced.columns])

for column in df_balanced.columns:
    if df_balanced[column].isnull().sum() > 0:
        if df_balanced[column].dtype == 'object':
            df_balanced[column] = df_balanced[column].fillna(df_balanced[column].mode()[0])
        else:
            df_balanced[column] = df_balanced[column].fillna(df_balanced[column].median())

print(f"Target distribution:\n{df_balanced['target'].value_counts()}")

# --------------------------
# Prepare Features and Labels
# --------------------------
print("Preparing features and labels...")
X = df_balanced.drop(columns=['target'])
y = df_balanced['target']

# --------------------------
# Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# Standardize Features
# --------------------------
print("Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# Train Model
# --------------------------
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# --------------------------
# Save Model and Scaler
# --------------------------
# Get the current working directory
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Define paths for model and scaler
model_path = os.path.join(current_dir, 'model.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

# Save the model and scaler
print(f"Saving model to: {model_path}")
joblib.dump(model, model_path)

print(f"Saving scaler to: {scaler_path}")
joblib.dump(scaler, scaler_path)

# Verify files were saved
if os.path.exists(model_path):
    print(f"✅ Model file saved successfully: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1024:.2f} KB")
else:
    print(f"❌ Failed to save model file: {model_path}")

if os.path.exists(scaler_path):
    print(f"✅ Scaler file saved successfully: {scaler_path}")
    print(f"File size: {os.path.getsize(scaler_path) / 1024:.2f} KB")
else:
    print(f"❌ Failed to save scaler file: {scaler_path}")

print("Done!")