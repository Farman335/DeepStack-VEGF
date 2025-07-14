import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix, matthews_corrcoef
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Activation, Add, Dense, Flatten, Bidirectional, LSTM, \
    LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Labels
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def residual_tcn_block(x, filters, kernel_size=3, dilation_rate=1):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='selu')(x)
    x = LayerNormalization()(x)
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='selu')(x)
    x = LayerNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('selu')(x)
    return x

def build_gac_bitcn(input_shape):
    inputs = Input(shape=input_shape)
    x1 = residual_tcn_block(inputs, filters=64, kernel_size=3, dilation_rate=1)
    x2 = residual_tcn_block(inputs, filters=64, kernel_size=5, dilation_rate=2)
    x3 = residual_tcn_block(inputs, filters=64, kernel_size=7, dilation_rate=3)
    x = Add()([x1, x2, x3])
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

def main(file_path):
    X, y = load_data(file_path)
    X = preprocess_data(X)

    # Reshape for Conv1D (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    input_shape = (X.shape[1], 1)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    accs, sens, specs, precs, f1s, aucs, mccs = [], [], [], [], [], [], []

    model = build_gac_bitcn(input_shape)

    plt.figure(figsize=(8, 6))  # Initialize ROC curve plot

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=100, verbose=1, validation_data=(X_test, y_test),
                  callbacks=[early_stop])

        y_pred_prob = model.predict(X_test)

        acc, sensitivity, specificity, precision, f1, auc, mcc = evaluate_metrics(y_test, y_pred_prob)

        accs.append(acc)
        sens.append(sensitivity)
        specs.append(specificity)
        precs.append(precision)
        f1s.append(f1)
        aucs.append(auc)
        mccs.append(mcc)

    model_json = model.to_json()
    with open("GAC_BiTCN.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("GAC_BiTCN.weights.h5")

    # Print averaged results
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Sensitivity: {np.mean(sens):.4f}")
    print(f"Specificity: {np.mean(specs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"F1-score: {np.mean(f1s):.4f}")
    print(f"AUC: {np.mean(aucs):.4f}")
    print(f"MCC: {np.mean(mccs):.4f}")


# Example usage
file_path = 'Fused set train.csv'
main(file_path)
