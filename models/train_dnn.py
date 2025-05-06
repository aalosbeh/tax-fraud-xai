import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Reshape, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# === Load Dataset ===
df = pd.read_csv("D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/data/synthetic_tax_fraud_dataset.csv")
df = df.drop(columns=["Unnamed: 30"], errors="ignore")  # remove if exists

# === Encode Categorical Variables ===
for col in ["filing_status", "occupation_category"]:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# === Define Feature Columns ===
features = [
    "income_reported",
    "deductions_claimed",
    "tax_credits_claimed",
    "num_dependents",
    "filing_status",
    "occupation_category",
    "days_to_deadline",
    "deduction_to_income_ratio",
    "credit_to_income_ratio",
    "expense_per_dependent",
    "income_per_dependent"
]
X = df[features].values
y = df["fraud_flag"].values

# === Standardize Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === Define Attention-Enhanced DNN Model ===
input_layer = Input(shape=(X_train.shape[1],))

# Dense projection
dense_proj = Dense(128, activation='relu')(input_layer)
dense_proj = Dropout(0.3)(dense_proj)

# Reshape for attention: (batch_size, time_steps=1, features)
reshaped = Reshape((1, X_train.shape[1]))(input_layer)
normalized = LayerNormalization()(reshaped)
attention_output = MultiHeadAttention(num_heads=2, key_dim=2)(normalized, normalized)
flattened = Flatten()(attention_output)

# Combine with dense layers
x = Dense(64, activation='relu')(flattened)
x = Dropout(0.3)(x)
output_layer = Dense(1, activation='sigmoid')(x)

# Build and compile model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Train Model ===
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# === Evaluate Model ===
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

# === Save Model ===
model_dir = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/models"
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "dnn_attention_model.h5"))
print(" DNN model saved to models/dnn_attention_model.h5")
