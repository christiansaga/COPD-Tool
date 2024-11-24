from scripts.preprocessing import load_data, preprocess_data, scale_features
from scripts.model_training import train_logistic_regression, save_model, load_model
from scripts.evaluation import evaluate_model, plot_confusion_matrix
from scripts.utilities import split_data
from sklearn.model_selection import cross_val_score
from collections import Counter

# Step 1: Load the dataset
data_file = "data/copd_data_preprocessed.csv"  # Use preprocessed data
df = load_data(data_file)

# Step 2: Preprocess the data
X, y = preprocess_data(df)

# Step 3: Split the data
X_train, X_test, y_train, y_test = split_data(X, y)

# Step 4: Scale the features
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

# Step 5: Train the Logistic Regression model
model = train_logistic_regression(X_train_scaled, y_train)

# Step 6: Evaluate the model
metrics = evaluate_model(model, X_test_scaled, y_test)
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
print(f"Classification Report:\n{metrics['classification_report']}")
print(f"Accuracy: {metrics['accuracy']:.2f}")
print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)
scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())


# Step 7: Visualize the confusion matrix
plot_confusion_matrix(model, X_test_scaled, y_test)
