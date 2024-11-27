from scripts.preprocessing import load_data, preprocess_data, scale_features
from scripts.model_training import train_logistic_regression
from scripts.evaluation import evaluate_model, plot_confusion_matrix
from scripts.utilities import split_data
from sklearn.model_selection import cross_val_score
#from collections import Counter

#Load data
data_file = "data/copd_data_preprocessed.csv"  # Use preprocessed data
df = load_data(data_file)

#Preprocess the data
X, y = preprocess_data(df)

#Split the data
X_train, X_test, y_train, y_test = split_data(X, y)

#Scaling the features
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

#Training of the Logistic Regression model
model = train_logistic_regression(X_train_scaled, y_train)

# Evaluation of the model
metrics = evaluate_model(model, X_test_scaled, y_test)
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
print(f"Classification Report:\n{metrics['classification_report']}")
print(f"Accuracy: {metrics['accuracy']:.2f}\n")
print("Model Coefficients:", model.coef_)
print("\nIntercept:", model.intercept_)
scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print("\nCross-Validation Scores:", scores)
print("\nMean Cross-Validation Score:", scores.mean())
print(f"\nRoc_AUC Score: {metrics['roc_auc_score']}")
print(f"\nPrecision: {metrics['precision_score']:.2f}\n")
# the confusion matrix
plot_confusion_matrix(model, X_test_scaled, y_test)
