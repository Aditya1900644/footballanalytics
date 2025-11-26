import math
import numpy as np
from typing import List, Dict, Any

# --- Constants ---
GOAL_X = 528
GOAL_Y = 176
POST_TOP = np.array([GOAL_X, GOAL_Y - 18])
POST_BOTTOM = np.array([GOAL_X, GOAL_Y + 18])

# Tuning Parameters
DECAY_CONSTANT = 0.04
PRESSURE_RADIUS = 40.0
MAX_DIST_THRESHOLD = 300


class RealTimeThreatEstimator:
    def __init__(self):
        pass

    def _calculate_distance_score(self, ball_pos: np.ndarray) -> float:
        goal_center = np.array([GOAL_X, GOAL_Y])
        dist = np.linalg.norm(ball_pos - goal_center)
        return math.exp(-DECAY_CONSTANT * (dist / 5.0))

    def _calculate_angle_score(self, ball_pos: np.ndarray) -> float:
        v_top = POST_TOP - ball_pos
        v_bottom = POST_BOTTOM - ball_pos
        d_top = np.linalg.norm(v_top)
        d_bottom = np.linalg.norm(v_bottom)

        if d_top == 0 or d_bottom == 0:
            return 0.0

        dot = np.dot(v_top, v_bottom)
        angle_rad = np.arccos(np.clip(dot / (d_top * d_bottom), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return np.clip(angle_deg / 60.0, 0.0, 1.0)

    def _calculate_pressure(self, ball_pos: np.ndarray, opponents: List[Dict]) -> float:
        defender_count = 0
        for opp in opponents:
            opp_pos = np.array(opp['projection'])
            dist = np.linalg.norm(ball_pos - opp_pos)
            if dist < PRESSURE_RADIUS:
                defender_count += 1
        penalty = min(0.9, defender_count * 0.15)
        return 1.0 - penalty

    def get_threat_score(self, frame_objects: List[Dict[str, Any]]) -> float:
        # 1. Find Ball
        ball = next(
            (obj for obj in frame_objects if obj['class'] == 'ball'), None)
        if ball is None:
            return 0.0

        ball_pos = np.array(ball['projection'])

        # 2. Safety Check: If Ball is in own defensive half (Far left), Threat is 0
        # (Assuming attacking right)
        if ball_pos[0] < 264:
            return 0.0

        # 3. Find Ball Owner
        ball_owner = next(
            (obj for obj in frame_objects if obj['has_ball']), None)

        # --- FIX: GOALKEEPER POSSESSION ---
        # If the Goalkeeper has the ball, it is almost never an attacking threat.
        if ball_owner and ball_owner['class'] == 'goalkeeper':
            return 0.0
        # ----------------------------------

        # 4. Identify Opponents
        opponents = []
        if ball_owner:
            opponents = [p for p in frame_objects if p['team']
                         != ball_owner['team'] and p['class'] != 'ball']

        # 5. Calculate Score
        s_dist = self._calculate_distance_score(ball_pos)
        s_angle = self._calculate_angle_score(ball_pos)
        s_pressure = self._calculate_pressure(ball_pos, opponents)

        final_score = s_dist * s_angle * s_pressure
        return float(final_score)
