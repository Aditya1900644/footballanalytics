from collections import deque
from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
from utils import point_distance

# Normalized field dimensions for distance calculation (e.g., [0, 528] x [0, 352])
# These constants MUST match the dimensions used in SpeedEstimator and ObjectPositionMapper
FIELD_W = 528
FIELD_H = 352


class PassTrigger:
    """
    Handles the frame-to-frame logic to detect a 'pass-release event'.
    A pass event is triggered by loss of possession or a sudden ball acceleration.
    """

    def __init__(self, buffer_size: int = 5,
                 speed_spike_threshold: float = 0.5,
                 away_movement_threshold: float = 1.0,
                 fps: float = 30.0):
        """
        Initializes the PassTrigger.

        Args:
            buffer_size (int): Number of frames to keep in history.
            speed_spike_threshold (float): Threshold (normalized units/frame) for ball acceleration.
            away_movement_threshold (float): Threshold (normalized units/frame) for ball moving away from owner.
            fps (float): Frames per second, used to normalize time-based checks.
        """
        # Buffer stores (frame_number, frame_objects_list)
        self.frame_buffer: deque[Tuple[int, List[Dict[str, Any]]]] = deque(
            maxlen=buffer_size)
        self.last_owner_id: Optional[int] = None
        self.last_ball_pos: Optional[Tuple[float, float]] = None
        self.fps = fps
        self.speed_spike_thresh = speed_spike_threshold
        self.away_move_thresh = away_movement_threshold

    def _get_ball_track(self, frame_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Helper to safely extract the ball track from the list."""
        # Note: assuming 'class' or 'type' is set to 'ball' for the ball
        return next((obj for obj in frame_objects if obj.get('class', '') == 'ball' or obj.get('type', '') == 'ball'), None)

    def _calculate_ball_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculates distance in normalized field units (pixels)."""
        # This uses the same field dimensions as the SpeedEstimator and Mapper
        # Assuming the 'projection' is already in pixel coordinates (0 to 528 / 0 to 352)
        return point_distance(pos1, pos2)

    def update(self, frame_num: int, current_frame_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Updates the buffer and checks for a pass-release event.

        Args:
            frame_num (int): Current frame number.
            current_frame_objects (List[Dict[str, Any]]): List of tracked objects for the current frame.

        Returns:
            Optional[Dict[str, Any]]: An event dictionary if a pass is detected, otherwise None.
        """
        event: Optional[Dict[str, Any]] = None
        current_owner_id = next(
            (obj['id'] for obj in current_frame_objects if obj.get('has_ball', False)), None)
        current_ball_track = self._get_ball_track(current_frame_objects)
        current_ball_pos = current_ball_track['projection'] if current_ball_track and 'projection' in current_ball_track else None

        # --- 1. Detect Loss of Possession (has_ball=T -> has_ball=F) ---
        if self.last_owner_id is not None and current_owner_id is None:
            # Last frame had an owner, current frame does not
            # This is the primary trigger for a pass
            event = {
                'frame': frame_num,
                'passer_id': self.last_owner_id,
                'reason': 'possession_loss'
            }

        # --- 2. Detect Ball Velocity Spike or Sharp Away Movement (Secondary Triggers) ---
        if event is None and self.last_owner_id is not None and current_ball_pos and self.last_ball_pos and len(self.frame_buffer) > 1:

            # Get the predicted passer's position from the previous frame (when they last had the ball)
            prev_frame_num, prev_frame_objects = self.frame_buffer[-1]
            last_owner_track = next(
                (obj for obj in prev_frame_objects if obj['id'] == self.last_owner_id), None)

            if last_owner_track and 'projection' in last_owner_track:
                last_owner_pos = last_owner_track['projection']

                # Distance ball moved between last frame and current frame
                ball_move_distance = self._calculate_ball_distance(
                    self.last_ball_pos, current_ball_pos)

                # Distance ball moved relative to the owner's position
                # Check if the ball is moving significantly away from the owner
                dist_from_owner_then = self._calculate_ball_distance(
                    self.last_ball_pos, last_owner_pos)
                dist_from_owner_now = self._calculate_ball_distance(
                    current_ball_pos, last_owner_pos)

                # Check for "Ball moving sharply away from the ball-owner's position"
                if (dist_from_owner_now - dist_from_owner_then) > self.away_move_thresh:
                    event = {
                        'frame': frame_num,
                        'passer_id': self.last_owner_id,
                        'reason': 'away_movement'
                    }

                # Check for "Ball velocity spike"
                # This requires calculating ball acceleration, which is complex. For simplicity and robustness,
                # we'll use a large, sudden change in velocity (speed spike)
                # We need the ball's previous speed, which is not readily available, so we use distance moved as a proxy for velocity spike.
                # The 'speed' value in tracks is for players, not the ball via the SpeedEstimator.
                # A direct distance check acts as a velocity spike detector:
                if ball_move_distance > self.speed_spike_thresh:
                    event = {
                        'frame': frame_num,
                        'passer_id': self.last_owner_id,
                        'reason': 'velocity_spike'
                    }

        # --- Update State for Next Frame ---
        self.frame_buffer.append((frame_num, current_frame_objects))

        # Only update last_owner_id if possession is confirmed in the current frame (T)
        if current_owner_id is not None:
            self.last_owner_id = current_owner_id

        # Always update ball position if available
        if current_ball_pos is not None:
            self.last_ball_pos = current_ball_pos

        return event
