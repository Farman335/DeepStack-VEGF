import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
# Set the path to your CSV file
file_path = "Protbert_train_CAB.csv"

# The name of the column you are trying to predict.
# This assumes the label column is the last column in your CSV.
# If your label column is named something else, change this string.
label_column = "label"

# --- Data Loading and Preparation ---

try:
    # Load the dataset
    print(f"Loading data from '{file_path}'...")
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    print("DataFrame shape:", df.shape)

    # Clean column names by removing leading/trailing whitespace
    df.columns = df.columns.str.strip()

    # Separate features (X) and target (y)
    if label_column not in df.columns:
        raise ValueError(f"The specified label column '{label_column}' was not found in the CSV file.")

    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Handle categorical features by converting them to dummy variables
    X = pd.get_dummies(X)

    # Ensure all feature columns are numeric for the model
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    if X.empty:
        raise ValueError("No numeric features found in the dataset after processing.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Model Training ---
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- SHAP Analysis ---
    print("\nStarting SHAP analysis...")

    # Create a SHAP Explainer object.
    # The explainer computes the SHAP values for each feature of each data point.
    explainer = shap.Explainer(model)

    # Calculate SHAP values for the test set
    shap_values = explainer(X_test)

    # --- Plotting ---
    print("Generating SHAP summary plot...")

    # Use matplotlib to show the plot
    # The summary plot shows the overall feature importance and their impact on the model's output
    shap.summary_plot(shap_values, X_test, show=False)

    # Save the plot to a file and display it
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png", dpi=300)
    print("SHAP summary plot saved as 'shap_summary_plot.png'")
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
