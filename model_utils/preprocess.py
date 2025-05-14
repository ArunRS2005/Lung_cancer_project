import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


def load_and_preprocess(csv_path, scaler_path=None, is_train=True):
    df = pd.read_csv(csv_path)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Encode categorical features
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

    # Features and target
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

    # Scale features
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path or "scaler.pkl")
    else:
        scaler = joblib.load(scaler_path or "scaler.pkl")
        X_scaled = scaler.transform(X)

    return X_scaled, y


def train_test_data(csv_path, scaler_path="scaler.pkl", test_size=0.2, random_state=42):
    X, y = load_and_preprocess(csv_path, scaler_path, is_train=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
