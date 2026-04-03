"""
Continuous Health Monitoring: Medical Review Visualizations
===========================================================
Runs both Supervised (RF) and Unsupervised (Isolation Forest) models,
and saves individual high-resolution plots for clinical review.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix

def main():
    print("Loading synthetic data...")
    try:
        epochs_df = pd.read_csv("health_epochs.csv")
        users_df = pd.read_csv("health_users.csv")
    except FileNotFoundError:
        print("Error: CSV files not found. Run the generator script first.")
        return

    # Convert timestamp for time-series plotting
    epochs_df["timestamp"] = pd.to_datetime(epochs_df["timestamp"])

    # Create output directory for images
    out_dir = "visualizations"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Feature Engineering
    # ---------------------------------------------------------
    df = epochs_df.merge(users_df, on="user_id", how="left")
    df["delta_hr"] = df["mean_hr"] - df["baseline_hr"]
    df["delta_spo2"] = df["mean_spo2"] - df["baseline_spo2"]
    df["delta_br"] = df["mean_br"] - df["baseline_br"]

    feature_cols = [
        "mean_hr", "min_hr", "max_hr", "mean_spo2", "min_spo2", 
        "mean_br", "hr_br_ratio", "delta_hr", "delta_spo2", "delta_br"
    ]
    X = df[feature_cols]
    y = df["true_latent_state"]

    # ---------------------------------------------------------
    # STEP 2: Train Both Models
    # ---------------------------------------------------------
    print("Training Random Forest (Supervised)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    test_df = df.loc[X_test.index].copy()
    test_df["rf_prediction"] = rf.predict(X_test)

    print("Training Isolation Forest (Unsupervised)...")
    iso = IsolationForest(n_estimators=100, contamination=0.10, random_state=42, n_jobs=-1)
    df["anomaly_score"] = iso.fit_predict(X)
    df["is_anomaly"] = df["anomaly_score"] == -1

    # ---------------------------------------------------------
    # STEP 3: Generate and Save Visualizations
    # ---------------------------------------------------------
    print(f"Saving high-res plots to '{out_dir}/'...")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # --- Plot 1: Patient Time-Series ---
    plt.figure(figsize=(12, 6))
    sample_user = df["user_id"].unique()[0] 
    user_data = df[df["user_id"] == sample_user].sort_values("timestamp").head(288) 
    
    ax1 = sns.lineplot(data=user_data, x="timestamp", y="mean_hr", label="Heart Rate", color="red")
    ax1_twin = ax1.twinx()
    sns.lineplot(data=user_data, x="timestamp", y="mean_spo2", label="SpO2", color="blue", ax=ax1_twin)
    
    anomalies = user_data[user_data["is_anomaly"]]
    ax1.scatter(anomalies["timestamp"], anomalies["mean_hr"], color="black", s=100, zorder=5, label="Detected Anomaly")
    
    plt.title(f"Patient Trend (24h) w/ Unsupervised Anomalies - User: {sample_user}", fontsize=14)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Heart Rate (BPM)")
    ax1_twin.set_ylabel("SpO2 (%)")
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/01_patient_trend.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Physiological Coupling ---
    plt.figure(figsize=(10, 8))
    corr = X[["mean_hr", "mean_spo2", "mean_br", "delta_hr", "delta_spo2", "delta_br"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
    plt.title("Physiological Feature Correlation", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/02_feature_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Clinical Distributions (Warning Fixed) ---
    plt.figure(figsize=(10, 6))
    # FIX: Added hue="true_latent_state" and legend=False
    sns.violinplot(
        data=df, x="true_latent_state", y="mean_hr", 
        hue="true_latent_state", palette="pastel", legend=False
    )
    plt.title("Distribution of Heart Rate by Health State", fontsize=14)
    plt.xlabel("Latent State")
    plt.ylabel("Mean HR (BPM)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/03_hr_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 4: Random Forest Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_df["true_latent_state"], test_df["rf_prediction"], labels=rf.classes_)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf.classes_, yticklabels=rf.classes_)
    plt.title("Random Forest Classification Accuracy", fontsize=14)
    plt.xlabel("Predicted State")
    plt.ylabel("True State")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/04_rf_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 5: Random Forest Feature Importance (Warning Fixed) ---
    plt.figure(figsize=(10, 6))
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    # FIX: Added hue=importances.index and legend=False
    sns.barplot(
        x=importances.values, y=importances.index, 
        hue=importances.index, palette="viridis", legend=False
    )
    plt.title("Feature Importance (Random Forest)", fontsize=14)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/05_rf_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 6: Unsupervised Anomaly Scatter ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df[~df["is_anomaly"]].sample(min(5000, len(df))), 
        x="delta_hr", y="delta_spo2", color="lightgray", alpha=0.5, label="Normal"
    )
    sns.scatterplot(
        data=df[df["is_anomaly"]], 
        x="delta_hr", y="delta_spo2", hue="true_latent_state", palette="bright"
    )
    plt.title("Isolation Forest Decision Space (Delta HR vs Delta SpO2)", fontsize=14)
    plt.xlabel("Deviation from Baseline HR (BPM)")
    plt.ylabel("Deviation from Baseline SpO2 (%)")
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/06_isolation_forest_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("All visual assets saved successfully!")

if __name__ == "__main__":
    main()