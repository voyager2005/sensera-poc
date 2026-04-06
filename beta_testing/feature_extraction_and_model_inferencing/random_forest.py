"""
Continuous Health Monitoring: ML Inference Pipeline (MVP)
=========================================================
Loads synthetic epoch data, computes baseline deviations, 
and trains a Random Forest Classifier to detect latent health states.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Loading data...")
    try:
        epoch_path = r"sensera-poc\health_epochs.csv"
        users_path = r"sensera-poc\health_users.csv"
        epochs_df = pd.read_csv(epoch_path)
        users_df = pd.read_csv(users_path)
    except FileNotFoundError:
        print("Error: Could not find CSV files. Make sure they are in the same directory.")
        return

    # ---------------------------------------------------------
    # STEP 3 & 4: Feature Engineering & Baseline Integration
    # ---------------------------------------------------------
    print("Engineering features and computing baseline deviations...")
    
    # Merge epochs with user baselines
    df = epochs_df.merge(users_df, on="user_id", how="left")

    # Compute Delta (Δ) Features - How far is the user from THEIR normal?
    df["delta_hr"] = df["mean_hr"] - df["baseline_hr"]
    df["delta_spo2"] = df["mean_spo2"] - df["baseline_spo2"]
    df["delta_br"] = df["mean_br"] - df["baseline_br"]

    # Define our feature matrix (X) and target labels (y)
    feature_cols = [
        "mean_hr", "min_hr", "max_hr", 
        "mean_spo2", "min_spo2", 
        "mean_br", "hr_br_ratio",
        "delta_hr", "delta_spo2", "delta_br"
    ]
    
    X = df[feature_cols]
    y = df["true_latent_state"]

    # ---------------------------------------------------------
    # STEP 5: ML Training Pipeline
    # ---------------------------------------------------------
    print(f"Dataset ready. Total epochs: {len(df)}")
    print("Splitting data into Training (80%) and Testing (20%) sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest Classifier...")
    # We use class_weight="balanced" because anomalies like "respiratory_strain" 
    # happen much less frequently than "normal" states.
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight="balanced",
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # STEP 6: Evaluation & Decision Making
    # ---------------------------------------------------------
    print("\nEvaluating Model on Test Data...")
    y_pred = rf_model.predict(X_test)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred))

    # Feature Importance - Let's see what the model cares about most
    print("\n" + "="*60)
    print("TOP 5 MOST IMPORTANT FEATURES")
    print("="*60)
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    print(importances.sort_values(ascending=False).head(5))

if __name__ == "__main__":
    main()