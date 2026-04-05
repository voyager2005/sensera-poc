class AlertManager:
    def __init__(self, anomaly_penalty=15, normal_decay=5, max_score=100):
        """
        Initializes the Alert Engine.
        - anomaly_penalty: Points added for an anomalous 5-min epoch.
        - normal_decay: Points subtracted for a normal 5-min epoch.
        - max_score: The ceiling for the risk score.
        """
        self.anomaly_penalty = anomaly_penalty
        self.normal_decay = normal_decay
        self.max_score = max_score
        
        # State dictionary to track ongoing user risk scores
        # Format: { user_id: current_risk_score }
        self.user_states = {}

    def get_alert_level(self, score):
        """Translates a numerical score into a clinical severity level."""
        if score >= 75:
            return "🔴 HIGH RISK (Immediate Action Required)"
        elif score >= 50:
            return "🟠 MODERATE CONCERN (Monitor Closely)"
        elif score >= 25:
            return "🟡 MILD DEVIATION (Self-Check Suggested)"
        else:
            return "🟢 NORMAL"

    def process_epoch(self, user_id, timestamp, is_anomaly):
        """
        Processes a single 5-minute epoch for a user and updates their state.
        """
        # Initialize user if they don't exist in the state tracker
        if user_id not in self.user_states:
            self.user_states[user_id] = 0

        current_score = self.user_states[user_id]

        # Apply penalty or decay
        if is_anomaly:
            current_score += self.anomaly_penalty
        else:
            current_score -= self.normal_decay

        # Clamp the score between 0 and max_score
        current_score = max(0, min(self.max_score, current_score))
        
        # Update state
        self.user_states[user_id] = current_score

        # Determine if we crossed a reporting threshold
        alert_level = self.get_alert_level(current_score)
        
        return {
            "user_id": user_id,
            "timestamp": timestamp,
            "is_anomaly": is_anomaly,
            "risk_score": current_score,
            "alert_status": alert_level
        }

# --- Example of how it integrates with our existing pipeline ---
# Assume `df` is the dataframe we evaluated in the last script, 
# sorted by timestamp, containing 'user_id', 'timestamp', and 'is_anomaly'

if __name__ == "__main__":
    alert_engine = AlertManager(anomaly_penalty=20, normal_decay=5)
    
    print("Simulating real-time data stream...\n")
    
    # Simulating a user going into a continuous strain event
    simulated_stream = [
        ("User_A", "10:00", False), # Normal
        ("User_A", "10:05", True),  # Spike
        ("User_A", "10:10", True),  # Sustained
        ("User_A", "10:15", True),  # Sustained
        ("User_A", "10:20", False), # Recovery starts
        ("User_A", "10:25", True),  # Relapse
        ("User_A", "10:30", True),  # Relapse
    ]
    
    for uid, ts, anomaly in simulated_stream:
        result = alert_engine.process_epoch(uid, ts, anomaly)
        print(f"[{ts}] Epoch Anomaly: {str(anomaly):<5} | Score: {result['risk_score']:<3} | State: {result['alert_status']}")