from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
from utils import point_distance

# Normalized field dimensions (same as in pass_trigger.py)
FIELD_W = 528
FIELD_H = 352
# Normalized center of opponent's goal (using the right goal line points [24-29])
# Assuming the goal is centered at x=528 and y=(176/352)*FIELD_H = 176
OPP_GOAL_CENTER_PROJECTION = np.array([FIELD_W, FIELD_H / 2.0])


class PassingPredictor:
    """
    Evaluates and ranks potential pass candidates based on tactical metrics.
    """

    def __init__(self,
                 max_short_pass_dist: float = 60.0,  # pixels (normalized)
                 max_long_pass_dist: float = 150.0,  # pixels (normalized)
                 density_radius: float = 50.0,      # pixels (normalized)
                 ):
        """
        Initializes the PassingPredictor with scoring and distance parameters.
        """
        self.max_short_dist = max_short_pass_dist
        self.max_long_dist = max_long_pass_dist
        self.density_radius = density_radius

    def _calculate_angle_to_goal(self, pos: np.ndarray, target_pos: np.ndarray) -> float:
        """
        Calculates the angle (in degrees) of the pass vector relative to the opponent's goal vector.
        0 degrees means the pass is perfectly aligned with the shot on goal.
        """
        goal_vector = OPP_GOAL_CENTER_PROJECTION - target_pos
        pass_vector = target_pos - pos

        dot_product = np.dot(pass_vector, goal_vector)
        mag_pass = np.linalg.norm(pass_vector)
        mag_goal = np.linalg.norm(goal_vector)

        if mag_pass == 0 or mag_goal == 0:
            return 0.0

        cosine_angle = dot_product / (mag_pass * mag_goal)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    def _check_lane_blocked(self, passer_pos: np.ndarray, receiver_pos: np.ndarray, opponents: List[Dict[str, Any]]) -> bool:
        """
        Checks if any opponent is blocking the passing lane.
        Uses a simple distance-to-line segment check.
        """
        lane_blocked = False
        p1 = passer_pos
        p2 = receiver_pos
        line_seg_sq = np.sum((p2 - p1)**2)

        if line_seg_sq == 0:
            return False  # Passer and receiver are at the same point

        # Check each opponent
        for opp in opponents:
            opp_pos = np.array(opp['projection'])

            # Calculate the projection of opponent onto the pass line segment
            t = np.dot(opp_pos - p1, p2 - p1) / line_seg_sq

            # Clamp 't' to [0, 1] to ensure the projection is on the segment
            t = np.clip(t, 0.0, 1.0)

            # Closest point on the line segment to the opponent
            closest_point = p1 + t * (p2 - p1)

            # Distance from the opponent to the line segment
            distance_sq = np.sum((opp_pos - closest_point)**2)

            # Use a small distance threshold (e.g., 20 pixels normalized)
            # The square root operation is avoided by squaring the threshold
            if distance_sq < (20.0)**2:
                lane_blocked = True
                break

        return lane_blocked

    def _score_pass(self, passer: Dict[str, Any], receiver: Dict[str, Any], opponents: List[Dict[str, Any]], velocity_vector: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Calculates score and classification for a single pass option.
        """
        passer_pos = np.array(passer['projection'])
        receiver_pos = np.array(receiver['projection'])
        distance = np.linalg.norm(passer_pos - receiver_pos)

        # --- 1. Tactical Value (Angle to Goal) ---
        angle_to_goal = self._calculate_angle_to_goal(passer_pos, receiver_pos)
        # Score is higher for smaller angles (more direct path to goal)
        angle_score = 1.0 - (angle_to_goal / 180.0)

        # --- 2. Risk Assessment (Opponent Density & Lane Block) ---
        lane_blocked = self._check_lane_blocked(
            passer_pos, receiver_pos, opponents)
        block_penalty = 0.5 if lane_blocked else 0.0

        opponent_density = sum(1 for opp in opponents if point_distance(
            opp['projection'], receiver['projection']) < self.density_radius)
        # Max 0.4 penalty for high density
        density_penalty = min(0.4, opponent_density * 0.1)

        # --- 3. Pass Classification ---
        pass_type = 'long'
        if distance < self.max_short_dist:
            pass_type = 'short'
        # FIX HERE: Change self.max_long_pass_dist to self.max_long_dist
        elif distance < self.max_long_dist:
            # Check for through ball (ahead of the last defender/goal-side)
            # Simplification: if receiver is goal-side of passer and distance is medium
            if receiver_pos[0] > passer_pos[0] and distance > self.max_short_dist * 1.5:
                pass_type = 'through'
            else:
                pass_type = 'medium'
        # FIX HERE: Change self.max_long_pass_dist to self.max_long_dist
        elif distance > self.max_long_dist:
            # Check for switch (if pass is roughly diagonal/wide)
            if abs(passer_pos[1] - receiver_pos[1]) > FIELD_H * 0.5:
                pass_type = 'switch'

        # --- 4. Final Score Calculation ---
        # Base score weights (can be tuned)
        score = (angle_score * 0.6) + (1.0 - density_penalty) * 0.4

        # Apply penalties
        score -= block_penalty

        # Normalize score to [0, 1]
        final_score = np.clip(score, 0.0, 1.0)

        # Map 'medium' to 'switch' or 'through' based on context if needed, otherwise keep 'medium'
        if pass_type == 'medium':
            pass_type = 'through' if receiver_pos[0] > passer_pos[0] else 'switch'

        return {
            'passer_id': passer['id'],
            'target_id': receiver['id'],
            'type': pass_type,
            'score': float(final_score),
            'distance': float(distance),
            'meta': {
                'angle_to_goal': float(angle_to_goal),
                'lane_blocked': lane_blocked,
                'opponent_density': opponent_density
            }
        }

    def rank_candidates(self, passer: Dict[str, Any], teammates: List[Dict[str, Any]],
                        opponents: List[Dict[str, Any]], topk: int = 3,
                        teammate_vels: Dict[int, Tuple[float, float]] = {}) -> List[Dict[str, Any]]:
        """
        Scores all teammates and returns the top k predictions.
        """
        pass_options = []
        for receiver in teammates:
            velocity = teammate_vels.get(receiver['id'])
            score_data = self._score_pass(
                passer, receiver, opponents, velocity)
            pass_options.append(score_data)

        # Sort by score descending and return topk
        pass_options.sort(key=lambda x: x['score'], reverse=True)
        return pass_options[:topk]
