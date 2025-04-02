# imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# load dataset
url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"
df = pd.read_csv(url)
# print(df.head())

# columns - need to add/remove but temporary
columns = ['winner_id', 'loser_id', 'surface', 'round', 'minutes', 'w_ace', 'w_df', 'w_bpSaved', 'w_bpFaced',
           'l_ace', 'l_df', 'l_bpSaved', 'l_bpFaced', 'winner_rank', 'loser_rank'] # btw github has definitions for terms
df = df[columns]

# cleaning
df = df.dropna()

# encoding -> changing categorical data into numerical
label_encoder = LabelEncoder()
df['surface'] = label_encoder.fit_transform(df['surface'])
df['round'] = label_encoder.fit_transform(df['round'])

df_winner = df.copy()
df_winner['target'] = 1

df_loser = df.copy()
df_loser['target'] = 0

# need to check what we want for columns
df_loser = df_loser.rename(columns={
    'winner_id': 'player2_id', 'loser_id': 'player1_id',
    'winner_rank': 'player2_rank', 'loser_rank': 'player1_rank',
    'w_ace': 'player2_ace', 'l_ace': 'player1_ace',
    'w_df': 'player2_df', 'l_df': 'player1_df',
    'w_bpSaved': 'player2_bpSaved', 'l_bpSaved': 'player1_bpSaved',
    'w_bpFaced': 'player2_bpFaced', 'l_bpFaced': 'player1_bpFaced'
})

df_winner = df_winner.rename(columns={
    'winner_id': 'player1_id', 'loser_id': 'player2_id',
    'winner_rank': 'player1_rank', 'loser_rank': 'player2_rank',
    'w_ace': 'player1_ace', 'l_ace': 'player2_ace',
    'w_df': 'player1_df', 'l_df': 'player2_df',
    'w_bpSaved': 'player1_bpSaved', 'l_bpSaved': 'player2_bpSaved',
    'w_bpFaced': 'player1_bpFaced', 'l_bpFaced': 'player2_bpFaced'
})

# originally, when i split the data, we only had one class, so this splits losers and winners
df_balanced = pd.concat([df_winner, df_loser], axis=0)

# print(df_balanced['target'].value_counts())


X = df_balanced.drop(columns=['target'])
y = df_balanced['target']

# training testing split - may need to adjust parameters?
# needed to straify y for class balance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # standarization 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# # models: log reg and decision tree - can add more
models = {
    "Logistic Regression": LogisticRegression(solver='saga'),
    "Decision Tree": DecisionTreeClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    X_scaled = scaler.fit_transform(X)  
    # cross validation on 10 folds
    # tests strongness of model
    # had to scale X because the number of iterations was being surpassed causing an error
    scores = cross_val_score(model, X_scaled, y, cv=10, scoring='accuracy')

    print("-" * 50)
    print(f"{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("-" * 50)
    
    #added cross validation
    print(f"{name} Cross-Validation:")
    print(f"Mean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")
    print("-" * 50)

    
    # added visual confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", xticklabels=["Lose", "Win"], yticklabels=["Lose", "Win"])
    plt.title(f"Confusion Matrix for {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()