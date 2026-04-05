"""
Continuous Health Monitoring: Unsupervised ML Inference (MVP)
=============================================================
Uses an Isolation Forest to detect physiological anomalies 
without relying on explicit health labels.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

def main():
    print("Loading data...")
    try:
        epochs_df = pd.read_csv("health_epochs.csv")
        users_df = pd.read_csv("health_users.csv")
    except FileNotFoundError:
        print("Error: Could not find CSV files. Make sure they are generated.")
        return

    # ---------------------------------------------------------
    # STEP 3 & 4: Feature Engineering
    # ---------------------------------------------------------
    df = epochs_df.merge(users_df, on="user_id", how="left")

    # Compute Delta (Δ) Features (Crucial for unsupervised learning!)
    df["delta_hr"] = df["mean_hr"] - df["baseline_hr"]
    df["delta_spo2"] = df["mean_spo2"] - df["baseline_spo2"]
    df["delta_br"] = df["mean_br"] - df["baseline_br"]

    feature_cols = [
        "mean_hr", "min_hr", "max_hr", 
        "mean_spo2", "min_spo2", 
        "mean_br", "hr_br_ratio",
        "delta_hr", "delta_spo2", "delta_br"
    ]
    
    X = df[feature_cols]

    # ---------------------------------------------------------
    # STEP 5: Unsupervised ML (Isolation Forest)
    # ---------------------------------------------------------
    print("Training Isolation Forest...")
    # 'contamination' is the expected proportion of outliers. 
    # We estimate about 10% of a user's life might be in an "anomalous" state.
    iso_forest = IsolationForest(
        n_estimators=100, 
        contamination=0.10, 
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the model AND predict in one step. 
    # Returns: 1 for normal (inliers), -1 for anomaly (outliers)
    df["anomaly_score"] = iso_forest.fit_predict(X)

    # ---------------------------------------------------------
    # STEP 6: MVP Evaluation (Checking our work)
    # ---------------------------------------------------------
    # Even though the model is unsupervised, WE know the true states 
    # because we synthesized the data. Let's see how well it did!
    
    print("\nEvaluating Unsupervised Model...")
    
    # Let's map our synthetic states: 
    # "normal" and "sleep" = 1 (Normal)
    # "stress", "exercise", "respiratory_strain" = -1 (Anomaly)
    df["true_anomaly"] = df["true_latent_state"].apply(
        lambda x: 1 if x in ["normal", "sleep"] else -1
    )

    print("\n" + "="*60)
    print("ISOLATION FOREST PERFORMANCE")
    print("="*60)
    # Note: Focus on the recall/precision of the '-1' class (the anomalies)
    print(classification_report(df["true_anomaly"], df["anomaly_score"]))
    
    # Show a few examples of what it caught
    caught_anomalies = df[(df["anomaly_score"] == -1) & (df["true_anomaly"] == -1)]
    print(f"\nSuccessfully caught {len(caught_anomalies)} true anomalies.")
    print("Sample of caught anomalies (Notice the high deltas):")
    print(caught_anomalies[["true_latent_state", "delta_hr", "delta_spo2", "delta_br"]].head())

if __name__ == "__main__":
    main()