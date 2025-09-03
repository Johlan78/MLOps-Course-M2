import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(csv_path, target_col):
    df = pd.read_csv(csv_path)
    # Handle the Telco Churn specific issue: 'TotalCharges' is read as object
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=[target_col, 'customerID'])
    y = df[target_col]
    return X, y

def split_data(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Vous pouvez ajouter ici des fonctions pour les graphiques de confusion matrix, etc.
# par exemple : plot_confusion_matrix, plot_roc_curve
