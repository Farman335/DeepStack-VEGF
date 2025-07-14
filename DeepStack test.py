import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix, matthews_corrcoef
import tensorflow as tf
from tensorflow.keras.models import model_from_json


# Load and preprocess test data
def load_test_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))  # Reshape for Conv1D
    return X_scaled, y


# Evaluation metrics
def evaluate_metrics(y_true, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)

    return acc, sensitivity, specificity, precision, f1, auc, mcc


# Plot ROC curve
def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main testing function
def test_model(test_file_path, model_json_path, model_weights_path):
    # Load test data
    X_test, y_test = load_test_data(test_file_path)

    # Load model architecture
    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)

    # Load weights
    model.load_weights(model_weights_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Predict
    y_pred_prob = model.predict(X_test)

    # Evaluate
    acc, sn, sp, prec, f1, auc, mcc = evaluate_metrics(y_test, y_pred_prob)

    # Print results
    print("=== Test Performance ===")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sn:.4f}")
    print(f"Specificity: {sp:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"AUC:         {auc:.4f}")
    print(f"MCC:         {mcc:.4f}")

    # ROC Curve
    plot_roc_curve(y_test, y_pred_prob)


# ==== Run Testing ====
test_model(
    test_file_path="Fused set test.csv",             # 🔁 Replace with your test file
    model_json_path="GAC_BiTCN.json",
    model_weights_path="GAC_BiTCN.weights.h5"
)
