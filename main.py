# --------------------------
# Imports
# --------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# --------------------------
# Load dataset
# --------------------------
url = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_2023.csv"
df = pd.read_csv(url)
# print(df.head())

# --------------------------
# Relevant Columns
# --------------------------
columns = [
    # most important
    'winner_seed', 'loser_seed', 
    
    #'winner_rank_points', 'loser_rank_points',

    # important
    'winner_rank', 'loser_rank', 
    #'w_1stIn', 'w_ace', 
    'winner_ioc',
    
    # less important
    'surface', 'loser_age', 'winner_age', 'tourney_level',
    
    #'l_bpSaved', 'w_bpFaced','w_2ndWon', 'w_1stWon', 
    
    'winner_ht', 'loser_ht'
]

df = df[columns].dropna()

# --------------------------
# Encode Variables
# --------------------------
le_surface = LabelEncoder()
le_ioc = LabelEncoder()
le_level = LabelEncoder()
df['surface'] = le_surface.fit_transform(df['surface'])
df['winner_ioc'] = le_ioc.fit_transform(df['winner_ioc'])
df['tourney_level'] = le_level.fit_transform(df['tourney_level'])


# --------------------------
# Create Winner and Loser DFs
# --------------------------

df_winner = df.copy()
df_winner['target'] = 1

df_loser = df.copy()
df_loser['target'] = 0

# columns renamed for player1 vs player2 format
df_winner = df_winner.rename(columns={
    #'winner_id': 'player1_id', 'loser_id': 'player2_id',
    'winner_seed': 'player1_seed', 'loser_seed': 'player2_seed',
    'winner_rank': 'player1_rank', 'loser_rank': 'player2_rank',
    'winner_ioc': 'player1_ioc', 'surface': 'surface',
    'winner_age': 'player1_age', 'loser_age': 'player2_age',
    'tourney_level': 'tourney_level',
    'winner_ht': 'player1_ht', 'loser_ht': 'player2_ht'
})

df_loser = df_loser.rename(columns={
    #'winner_id': 'player2_id', 'loser_id': 'player1_id',
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

# print(df_balanced['target'].value_counts())
df_balanced = df_balanced.drop(columns=[col for col in ['player1_id', 'player2_id'] if col in df_balanced.columns])

for column in df_balanced.columns:
    if df_balanced[column].isnull().sum() > 0:
        if df_balanced[column].dtype == 'object':
            df_balanced[column] = df_balanced[column].fillna(df_balanced[column].mode()[0])
        else:
            df_balanced[column] = df_balanced[column].fillna(df_balanced[column].median())

df_balanced = df_balanced.drop(columns=[col for col in ['player1_id', 'player2_id'] if col in df_balanced.columns])

print(f"Target distribution:\n{df_balanced['target'].value_counts()}")


# --------------------------
# Prepare Features and Labels
# --------------------------
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
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)

# --------------------------
# PCA - Dimensionality Reduction Visualization
# --------------------------

# This is for 1D PCA
    # finds directions where data varies the most 
    # purpose: show similarities between groups of samples in data set
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# plt.figure(figsize=(10, 6))
# plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
# plt.xlabel("Number of Components")
# plt.ylabel("Total Explained Variance")
# plt.title("PCA")
# plt.grid(True)
# plt.show()

# This is for PCA 2D model
    # each point represents 1 match
    # PCA 1 = most important direction of variance
    # PCA 2 = second-most important
    # 1 - match win, 0 - match loss
pca_2 = PCA(n_components=2)
X_pca_2d = pca_2.fit_transform(X_scaled)
# plt.figure(figsize=(8, 5))
# sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=y, alpha=0.6)
# plt.title("PCA -2D Projection")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.show()

#showing KNN visually
knn_2d = KNeighborsClassifier(n_neighbors=7)
knn_2d.fit(X_pca_2d, y)

#creates a mesh grid of evenly spaced points acrossed the 2d space using steps of 0.1
x_min, x_max = X_pca_2d[:, 0].min() - 1, X_pca_2d[:, 0].max() + 1
y_min, y_max = X_pca_2d[:, 1].min() - 1, X_pca_2d[:, 1].max() + 1

#xx and yy are 2d arrays from the meshgrid
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

#predict for each point in the mesh grid
#raveling flattens xx and yy into 1d arrays of all the x or y coordinates in the grid
#z tells you the value predicted at each (x,y) spot (either 1 or 0)
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=y, palette='coolwarm', alpha=0.7)
plt.title('k-NN Decision Boundary (PCA 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Match Outcome', labels=["lose (0)", "win (1)"])
plt.grid(True)
plt.show()

# --------------------------
# Models
# --------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "k-NN (k=7)": KNeighborsClassifier(n_neighbors=7)
}


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
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
    
    # added cross validation
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