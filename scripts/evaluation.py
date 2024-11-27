from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, precision_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "probabilities": y_prob,
        "roc_auc_score": roc_auc_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred),
    }
    return metrics

def plot_confusion_matrix(model, X_test, y_test):
    """Plot the confusion matrix."""
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
