from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
import os

# Load your preprocessed data
# Adjust the relative path to match the correct location
data_file = os.path.join(os.path.dirname(__file__), "../data/copd_data_preprocessed.csv")

# Check if the file exists and raise an error if not
if not os.path.exists(data_file):
    raise FileNotFoundError(f"File not found: {data_file}")

# Load the data
df = pd.read_csv(data_file)

# Separate features and target
X = df.drop(columns=["Hospital_Admission"])
y = df["Hospital_Admission"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(
    class_weight="balanced",  # Adjust for unbalanced classes
    max_iter=5000,
    C=0.1,  # Regularization strength
    solver="liblinear"
)
model.fit(X_train_scaled, y_train)

# Save the trained model to a file
# Ensure the model directory exists
model_dir = os.path.join(os.path.dirname(__file__), "../model")
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "logistic_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
