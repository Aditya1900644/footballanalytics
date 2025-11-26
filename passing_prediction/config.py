

DEFAULTS = {
    # Frame buffer size for slow-mo and trigger lookback
    "buffer_frames": 10,
    # Prediction parameters
    "speed_spike_threshold": 80.0,  # distance in normalized pixels per frame
    "away_movement_threshold": 40.0,  # distance in normalized pixels per frame
    "score_thresh": 0.60,  # Minimum score to visualize a prediction
    # Visualization parameters
    "pause_frames": 45,
    "slowmo_factor": 2,
    "slowmo_frames": 30,
}
