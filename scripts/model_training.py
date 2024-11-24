from sklearn.linear_model import LogisticRegression
import joblib

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model with hyperparameter tuning.
    """
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=5000,
        C=0.1,  # Stronger regularization to reduce overfitting
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)

def load_model(file_path):
    """Load a saved model from a file."""
    return joblib.load(file_path)
