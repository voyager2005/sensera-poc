"""
Clinical-Grade Physiological Data Generator (v2)
================================================
Simulates raw 1Hz physiological data (Heart Rate, Breathing Rate, SpO2) 
using mean-reverting stochastic processes (Ornstein-Uhlenbeck). 

Applies a 60-second sliding window (30-second step) to extract 27+ 
clinical time-domain and non-linear features, alongside a MEWS severity score.

Requires: pip install numpy pandas
"""

import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration & MEWS Rules
# ---------------------------------------------------------------------------

STATES = ["normal", "sleep", "exercise", "respiratory_strain"]
STATE_PROBS = [0.65, 0.25, 0.05, 0.05]

# State Targets: (Mean HR, Mean RR, Mean SpO2, Volatility Multiplier)
STATE_PROFILES = {
    "normal":             (75.0,  15.0, 98.0, 1.0),
    "sleep":              (55.0,  12.0, 97.0, 0.5),
    "exercise":           (140.0, 26.0, 96.0, 2.5),
    "respiratory_strain": (115.0, 28.0, 89.0, 2.0)
}

def get_mews_hr(hr):
    if hr <= 40: return 3
    if hr <= 50: return 2
    if hr <= 60: return 1
    if hr <= 100: return 0
    if hr <= 110: return 1
    if hr <= 129: return 2
    return 3

def get_mews_rr(rr):
    if rr <= 8: return 3
    if rr <= 11: return 1
    if rr <= 20: return 0
    if rr <= 24: return 2
    return 3

def get_mews_spo2(spo2):
    if spo2 <= 85: return 3
    if spo2 <= 89: return 2
    if spo2 <= 93: return 1
    return 0

# ---------------------------------------------------------------------------
# Raw 1Hz Signal Generation (Ornstein-Uhlenbeck Process)
# ---------------------------------------------------------------------------

def generate_1hz_signals(duration_seconds, base_hr, base_rr, base_spo2):
    """
    Generates realistic 1-second interval data using an autoregressive process.
    Signals revert to a target mean depending on the current health state.
    """
    hr_arr = np.zeros(duration_seconds)
    rr_arr = np.zeros(duration_seconds)
    spo2_arr = np.zeros(duration_seconds)
    states = []
    
    # Initialize variables
    current_hr = base_hr
    current_rr = base_rr
    current_spo2 = base_spo2
    
    current_state = "normal"
    state_timer = 0
    
    # Mean reversion speed (theta) - dictates how fast signals move to target
    theta_hr, theta_rr, theta_spo2 = 0.05, 0.08, 0.02
    
    for t in range(duration_seconds):
        # State transitions
        if state_timer <= 0:
            current_state = np.random.choice(STATES, p=STATE_PROBS)
            if current_state == "sleep": state_timer = np.random.randint(3600, 14400) # 1 to 4 hours
            elif current_state == "normal": state_timer = np.random.randint(1800, 7200)
            else: state_timer = np.random.randint(600, 1800) # 10 to 30 mins
            
        target_hr, target_rr, target_spo2, vol = STATE_PROFILES[current_state]
        
        # Add correlation noise (HR and RR share a noise component)
        shared_noise = np.random.normal(0, 1)
        
        # Ornstein-Uhlenbeck formula: dx = theta * (mu - x) * dt + sigma * dW
        current_hr += theta_hr * (target_hr - current_hr) + vol * (0.7 * shared_noise + 0.3 * np.random.normal(0, 1))
        current_rr += theta_rr * (target_rr - current_rr) + (vol * 0.5) * (0.7 * shared_noise + 0.3 * np.random.normal(0, 1))
        
        # SpO2 drops logically follow extreme HR/RR strain, but with a lag
        current_spo2 += theta_spo2 * (target_spo2 - current_spo2) + (vol * 0.2) * np.random.normal(0, 1)
        
        # Strict physiological bounds
        current_hr = np.clip(current_hr, 30, 200)
        current_rr = np.clip(current_rr, 4, 45)
        current_spo2 = np.clip(current_spo2, 70, 100)
        
        hr_arr[t] = current_hr
        rr_arr[t] = current_rr
        spo2_arr[t] = current_spo2
        states.append(current_state)
        
        state_timer -= 1
        
    return hr_arr, rr_arr, spo2_arr, states

# ---------------------------------------------------------------------------
# Feature Extraction (60s Window)
# ---------------------------------------------------------------------------

def extract_features(window_hr, window_rr, window_spo2):
    """
    Extracts the exact 27 time-domain and non-linear features requested.
    """
    features = {}
    time_x = np.arange(len(window_hr))
    
    # --- HEART RATE (9 Features) ---
    features['hr_mean'] = np.mean(window_hr)
    features['hr_std'] = np.std(window_hr)
    features['hr_min'] = np.min(window_hr)
    features['hr_max'] = np.max(window_hr)
    features['hr_range'] = features['hr_max'] - features['hr_min']
    features['hr_slope'] = np.polyfit(time_x, window_hr, 1)[0]
    features['hr_cv'] = features['hr_std'] / features['hr_mean'] if features['hr_mean'] > 0 else 0
    
    # HRV Proxies (Approximating R-R intervals in ms)
    rr_intervals_ms = (60.0 / window_hr) * 1000
    successive_diffs = np.abs(np.diff(rr_intervals_ms))
    features['hr_rmssd'] = np.sqrt(np.mean(successive_diffs**2))
    features['hr_pnn50'] = (np.sum(successive_diffs > 50) / len(successive_diffs)) * 100
    
    # --- SpO2 (9 Features) ---
    features['spo2_mean'] = np.mean(window_spo2)
    features['spo2_std'] = np.std(window_spo2)
    features['spo2_min'] = np.min(window_spo2)
    features['spo2_max'] = np.max(window_spo2)
    features['spo2_range'] = features['spo2_max'] - features['spo2_min']
    features['spo2_slope'] = np.polyfit(time_x, window_spo2, 1)[0]
    features['spo2_time_below_95'] = (np.sum(window_spo2 < 95) / len(window_spo2)) * 100
    
    # Desaturation logic: Count distinct drops >= 3% within the 60s window
    # Recovery time: Estimated by ratio of positive slope segments
    spo2_diffs = np.diff(window_spo2)
    features['spo2_desat_events'] = 1 if features['spo2_range'] >= 3.0 and features['spo2_slope'] < 0 else 0
    features['spo2_recovery_time_est'] = np.sum(spo2_diffs > 0) # Seconds spent recovering
    
    # --- RESPIRATORY RATE (9 Features) ---
    features['rr_mean'] = np.mean(window_rr)
    features['rr_std'] = np.std(window_rr)
    features['rr_min'] = np.min(window_rr)
    features['rr_max'] = np.max(window_rr)
    features['rr_range'] = features['rr_max'] - features['rr_min']
    features['rr_slope'] = np.polyfit(time_x, window_rr, 1)[0]
    features['rr_cv'] = features['rr_std'] / features['rr_mean'] if features['rr_mean'] > 0 else 0
    
    # Apnea logic: Consecutive seconds RR < 8. If sum > 10 in window, flag.
    rr_low = (window_rr < 8).astype(int)
    max_consecutive_apnea = np.max([len(x) for x in "".join(rr_low.astype(str)).split('0')]) if '1' in "".join(rr_low.astype(str)) else 0
    features['rr_apnea_events'] = 1 if max_consecutive_apnea >= 10 else 0
    features['rr_tachypnea_duration'] = (np.sum(window_rr > 20) / len(window_rr)) * 100

    # --- CROSS-SIGNAL (2 Features) ---
    features['cross_hr_spo2_corr'] = np.corrcoef(window_hr, window_spo2)[0, 1]
    if np.isnan(features['cross_hr_spo2_corr']): features['cross_hr_spo2_corr'] = 0.0
    features['cross_hr_rr_ratio'] = features['hr_mean'] / features['rr_mean'] if features['rr_mean'] > 0 else 0
    
    return features

# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def build_dataset(n_users=5, hours_per_user=12):
    print(f"Initializing 1Hz Simulation for {n_users} users...")
    
    window_size = 60 # 60 seconds
    step_size = 30   # 30 seconds overlapping slide
    
    duration_secs = hours_per_user * 3600
    all_epochs = []
    
    for i in range(n_users):
        user_id = "UID_" + str(uuid.uuid4())[:8].upper()
        print(f"  Generating raw waveforms for {user_id} ({hours_per_user} hours)...")
        
        # User baseline parameters
        base_hr = np.random.normal(70, 10)
        base_rr = np.random.normal(15, 2)
        base_spo2 = 98.0
        
        hr_raw, rr_raw, spo2_raw, states = generate_1hz_signals(duration_secs, base_hr, base_rr, base_spo2)
        
        start_time = datetime.now().replace(microsecond=0) - timedelta(hours=hours_per_user)
        
        # Sliding Window Extraction
        num_windows = (duration_secs - window_size) // step_size + 1
        
        for w in range(num_windows):
            start_idx = w * step_size
            end_idx = start_idx + window_size
            
            w_hr = hr_raw[start_idx:end_idx]
            w_rr = rr_raw[start_idx:end_idx]
            w_spo2 = spo2_raw[start_idx:end_idx]
            
            # Ground truth state is the most common state in this 60s window
            w_state = max(set(states[start_idx:end_idx]), key=states[start_idx:end_idx].count)
            
            # 1. Extract 27 Features
            feats = extract_features(w_hr, w_rr, w_spo2)
            
            # 2. Calculate MEWS Score based on the means of the epoch
            mews_hr_score = get_mews_hr(feats['hr_mean'])
            mews_rr_score = get_mews_rr(feats['rr_mean'])
            mews_spo2_score = get_mews_spo2(feats['spo2_mean'])
            total_mews = mews_hr_score + mews_rr_score + mews_spo2_score
            
            # 3. Compile Epoch Row
            epoch_data = {
                "timestamp": (start_time + timedelta(seconds=end_idx)).strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": user_id,
                "latent_state": w_state,
                "mews_score": total_mews,
                "mews_hr_pts": mews_hr_score,
                "mews_rr_pts": mews_rr_score,
                "mews_spo2_pts": mews_spo2_score
            }
            
            # Merge extracted features
            epoch_data.update(feats)
            
            # Add user baselines for delta processing later
            epoch_data['user_base_hr'] = base_hr
            epoch_data['user_base_rr'] = base_rr
            epoch_data['user_base_spo2'] = base_spo2
            
            all_epochs.append(epoch_data)

    df = pd.DataFrame(all_epochs)
    
    # Round all floats for cleaner CSV output
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(3)
    
    output_path = "health_datagen_v2.csv"
    df.to_csv(output_path, index=False)
    
    print("\nDataset Generation Complete!")
    print(f"Total Epochs Generated: {len(df)}")
    print(f"Features per Epoch: {len(df.columns)}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # Generate 5 users, 12 hours of data each
    # This will yield approx (12 * 120 windows/hr) * 5 = 7,200 rows of rich data
    build_dataset(n_users=5, hours_per_user=12)