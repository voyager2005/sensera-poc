"""
Epoch-Based Synthetic Physiological Dataset Generator
===================================================
Generates two CSVs for continuous health monitoring:
  - users.csv  : user metadata and personal physiological baselines
  - epochs.csv : 5-minute windowed feature summaries (mean, std, min, max)

Usage
-----
  python health_synthetic_generator.py                        # defaults
  python health_synthetic_generator.py --users 100 --days 7

Requirements: Python 3.8+, no third-party dependencies.
"""

import argparse
import csv
import math
import os
import random
import string
import sys
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config & Latent Health States
# ---------------------------------------------------------------------------

STATES = ["normal", "exercise", "stress", "respiratory_strain", "sleep"]
STATE_W = [60, 5, 10, 5, 20] # Probabilities of being in a state

# Define how signals behave in each state compared to the user's baseline.
# Format: (HR_multiplier, SpO2_shift, BR_multiplier)
STATE_PROFILES = {
    "normal":             (1.0,  0.0, 1.0),
    "sleep":              (0.85, 0.0, 0.8),
    "exercise":           (1.8, -1.0, 2.0),
    "stress":             (1.3,  0.0, 1.4),
    "respiratory_strain": (1.4, -4.0, 1.6), # HR up, SpO2 down, BR up
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_choice(population, weights):
    total = sum(weights)
    r = random.uniform(0, total)
    cumulative = 0
    for item, w in zip(population, weights):
        cumulative += w
        if r <= cumulative:
            return item
    return population[-1]

def uid(length=8):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))

def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def progress(current, total, bar_width=40):
    pct = current / total
    filled = int(bar_width * pct)
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(f"\r  [{bar}] {current:,}/{total:,}  ({pct*100:.1f}%)")
    sys.stdout.flush()

# Generates two correlated variables (e.g., HR and BR usually move together)
def correlated_gauss(mean1, std1, mean2, std2, rho=0.7):
    z1 = random.gauss(0, 1)
    z2 = random.gauss(0, 1)
    # Apply correlation (rho)
    y1 = mean1 + std1 * z1
    y2 = mean2 + std2 * (rho * z1 + math.sqrt(1 - rho**2) * z2)
    return y1, y2

# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate(
    n_users: int = 50,
    n_days: int = 7,
    seed: int = 42,
    output_dir: str = ".",
) -> dict:
    random.seed(seed)
    
    end_date   = datetime.now().replace(microsecond=0)
    start_date = end_date - timedelta(days=n_days)

    users_rows = []
    epochs_rows = []

    print(f"\nGenerating {n_days} days of 5-min epochs for {n_users} users...")

    # ---- 1. Generate Users & Baselines ----
    users = []
    for _ in range(n_users):
        age = random.randint(20, 75)
        # Baseline resting heart rate (varies by age/fitness)
        base_hr = random.gauss(70, 8) 
        # Baseline SpO2 (usually 97-100)
        base_spo2 = min(100.0, random.gauss(98.5, 1.0))
        # Baseline breathing rate
        base_br = random.gauss(14, 2)

        user = {
            "user_id": "USR-" + uid(),
            "age": age,
            "baseline_hr": round(base_hr, 1),
            "baseline_spo2": round(base_spo2, 1),
            "baseline_br": round(base_br, 1)
        }
        users.append(user)
        users_rows.append(user)

    # ---- 2. Generate Time-Series Epochs ----
    total_epochs = n_users * (n_days * 24 * 12) # 12 epochs (5-min) per hour
    current_epoch_count = 0

    for user in users:
        current_time = start_date
        
        # We use a Markov-chain style approach to keep the state sticky.
        # Once you are in "exercise", you likely stay in it for a few epochs.
        current_state = "normal"
        state_duration = 0

        while current_time < end_date:
            if current_epoch_count % 1000 == 0:
                progress(current_epoch_count, total_epochs)

            # State transition logic
            if state_duration <= 0:
                current_state = weighted_choice(STATES, STATE_W)
                if current_state == "sleep":
                    state_duration = random.randint(48, 96) # 4 to 8 hours (12 epochs/hr)
                elif current_state == "exercise":
                    state_duration = random.randint(6, 18)  # 30 to 90 mins
                elif current_state == "respiratory_strain":
                    state_duration = random.randint(24, 72) # Strain lasts a while
                else:
                    state_duration = random.randint(2, 12)  # Normal/Stress fluctuates faster

            # Get target means based on the current state and personal baseline
            hr_mult, spo2_shift, br_mult = STATE_PROFILES[current_state]
            
            target_mean_hr = user["baseline_hr"] * hr_mult
            target_mean_spo2 = min(100.0, user["baseline_baseline_spo2"] + spo2_shift if "baseline_baseline_spo2" in user else user["baseline_spo2"] + spo2_shift)
            target_mean_br = user["baseline_br"] * br_mult

            # Add circadian rhythm drift (HR drops slightly at night)
            hour = current_time.hour
            if 2 <= hour <= 5 and current_state == "normal":
                target_mean_hr -= 5
            elif 14 <= hour <= 17:
                target_mean_hr += 5

            # Generate correlated HR and BR (they move together)
            # Higher HR state -> higher variance (std)
            hr_std = 3.0 if current_state in ["normal", "sleep"] else 10.0
            br_std = 1.0 if current_state in ["normal", "sleep"] else 4.0
            
            epoch_mean_hr, epoch_mean_br = correlated_gauss(
                target_mean_hr, hr_std, 
                target_mean_br, br_std, 
                rho=0.85 # Strong positive correlation
            )

            # Generate SpO2 (slightly negative correlation to HR if strained)
            spo2_std = 0.5 if current_state in ["normal", "sleep"] else 2.0
            epoch_mean_spo2 = random.gauss(target_mean_spo2, spo2_std)
            
            # Cap realistic values
            epoch_mean_hr = max(40, min(200, epoch_mean_hr))
            epoch_mean_br = max(8, min(40, epoch_mean_br))
            epoch_mean_spo2 = max(80, min(100, epoch_mean_spo2))

            # Simulate min/max/variance within the 5-minute epoch
            # (Features your ML model will actually use to detect anomalies)
            epoch_max_hr = epoch_mean_hr + abs(random.gauss(hr_std, 2))
            epoch_min_hr = epoch_mean_hr - abs(random.gauss(hr_std, 2))
            epoch_min_spo2 = epoch_mean_spo2 - abs(random.gauss(spo2_std * 1.5, 0.5))

            epochs_rows.append({
                "timestamp": fmt_dt(current_time),
                "user_id": user["user_id"],
                "true_latent_state": current_state, # The label for training ML!
                "mean_hr": round(epoch_mean_hr, 1),
                "min_hr": round(epoch_min_hr, 1),
                "max_hr": round(epoch_max_hr, 1),
                "mean_spo2": round(epoch_mean_spo2, 1),
                "min_spo2": round(epoch_min_spo2, 1),
                "mean_br": round(epoch_mean_br, 1),
                "hr_br_ratio": round(epoch_mean_hr / epoch_mean_br, 2) if epoch_mean_br > 0 else 0
            })

            # Advance time by 5 minutes
            current_time += timedelta(minutes=5)
            state_duration -= 1
            current_epoch_count += 1

    progress(total_epochs, total_epochs)
    print()

    return {
        "users": users_rows,
        "epochs": epochs_rows,
    }

# ---------------------------------------------------------------------------
# CSV Writer & CLI
# ---------------------------------------------------------------------------

def write_csv(rows: list, path: str):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Wrote {len(rows):>10,} rows  ->  {path}  ({size_kb:,.1f} KB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Epoch-Based Health Data")
    parser.add_argument("--users", type=int, default=10, help="Number of users")
    parser.add_argument("--days", type=int, default=7, help="Days of data per user")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = generate(n_users=args.users, n_days=args.days, output_dir=args.out)
    
    print("\nWriting files...")
    write_csv(data["users"], os.path.join(args.out, "health_users.csv"))
    write_csv(data["epochs"], os.path.join(args.out, "health_epochs.csv"))
    
    print("\nDone! Your ML pipeline now has labeled data.")