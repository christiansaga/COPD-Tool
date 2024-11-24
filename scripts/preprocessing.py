import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess dataset by separating features and target."""
    X = df.drop(columns=['Hospital_Admission'])  # Features
    y = df['Hospital_Admission']  # Target
    return X, y

def scale_features(X_train, X_test):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
