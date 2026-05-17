# =========================================================
# APL LOGISTICS - MODEL TRAINING SCRIPT
# Machine Learning Late Delivery Risk Prediction
# =========================================================

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# LOAD DATA
# =========================================================

possible_paths = [
    "data/Logistics.csv",
    "Logistics.csv",
    "data.csv"
]

df = None

for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin1")
        break

if df is None:
    raise FileNotFoundError("CSV file not found.")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("(", "", regex=False)
df.columns = df.columns.str.replace(")", "", regex=False)

# =========================================================
# TARGET COLUMN
# =========================================================

target_column = "Late_delivery_risk"

if target_column not in df.columns:
    raise Exception("Late_delivery_risk column not found.")

# =========================================================
# REMOVE UNUSED COLUMNS
# =========================================================

drop_columns = [
    "Customer_Fname",
    "Customer_Lname",
    "Customer_Street"
]

for col in drop_columns:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

for col in df.columns:
    
    # Handle text columns
    if df[col].dtype == "object" or str(df[col].dtype) == "string":
        df[col] = df[col].fillna("Unknown")

    # Handle numeric columns
    else:
        try:
            df[col] = df[col].fillna(df[col].median())
        except:
            df[col] = df[col].fillna(0)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

if (
    "Days_for_shipping_real" in df.columns and
    "Days_for_shipment_scheduled" in df.columns
):
    df["Shipping_Pressure_Index"] = (
        df["Days_for_shipping_real"] -
        df["Days_for_shipment_scheduled"]
    )

if "Shipping_Mode" in df.columns:

    df["Express_Flag"] = np.where(
        df["Shipping_Mode"].str.contains(
            "Express",
            case=False,
            na=False
        ),
        1,
        0
    )

if (
    "Order_Item_Quantity" in df.columns and
    "Sales" in df.columns
):
    df["Order_Complexity_Score"] = (
        df["Order_Item_Quantity"] * df["Sales"]
    )

# =========================================================
# ENCODE CATEGORICAL COLUMNS
# =========================================================

encoders = {}

categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col].astype(str))

    encoders[col] = encoder

# =========================================================
# FEATURES & TARGET
# =========================================================

X = df.drop(target_column, axis=1)
y = df[target_column]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# =========================================================
# BASELINE MODEL
# =========================================================

print("\nTraining Logistic Regression...")

baseline_model = LogisticRegression(max_iter=1000)

baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print("Baseline Accuracy:", round(baseline_accuracy, 4))

# =========================================================
# ADVANCED MODEL
# =========================================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=15,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================================================
# PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# =========================================================
# EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC AUC  :", round(roc_auc, 4))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:\n")
print(importance_df.head(10))

# =========================================================
# SAVE MODELS
# =========================================================

os.makedirs("models", exist_ok=True)

pickle.dump(
    model,
    open("models/model.pkl", "wb")
)

pickle.dump(
    list(X.columns),
    open("models/columns.pkl", "wb")
)

pickle.dump(
    encoders,
    open("models/encoders.pkl", "wb")
)

pickle.dump(
    importance_df,
    open("models/feature_importance.pkl", "wb")
)

print("\nModel Saved Successfully")
print("Files Created:")
print("✔ model.pkl")
print("✔ columns.pkl")
print("✔ encoders.pkl")
print("✔ feature_importance.pkl")