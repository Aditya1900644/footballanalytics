from utils import point_distance, get_bbox_center
from .possession_tracking import PossessionTracker
from club_assignment import Club

from typing import Dict, Tuple, Any


class BallToPlayerAssigner:
    """Assigns the ball to a player if it fits the criteria"""

    def __init__(self,
                 club1: Club,
                 club2: Club,
                 max_ball_distance: float = 10.0,
                 grace_period: float = 4.0,
                 ball_grace_period: float = 2.0,
                 fps: int = 30,
                 max_ball_speed: float = 250.0,
                 speed_check_frames: int = 5,
                 penalty_point_distance: float = 15.0) -> None:
        """
        Initializes the BallToPlayerAssigner with necessary parameters.
        (Original __init__ remains largely unchanged, ensuring attributes match the old logic)
        """
        self.max_ball_distance = max_ball_distance
        self.grace_period_frames = int(grace_period * fps)
        self.ball_grace_period_frames = int(ball_grace_period * fps)
        self.max_ball_speed = max_ball_speed
        self.speed_check_frames = speed_check_frames
        self.possession_tracker = PossessionTracker(club1, club2)
        self.last_possession_frame = None
        self.last_player_w_ball = None
        self.last_possessing_team = -1  # Team name or -1
        self.ball_exists = False
        self.ball_lost_frame = None
        self.ball_history = []
        self.penalty_point_distance = penalty_point_distance

    def is_ball_movement_valid(self, ball_pos: Tuple[float, float], current_frame: int) -> bool:
        """
        Checks if the ball's movement is valid based on its previous position.
        (Original method remains here)
        """
        if not self.ball_history:
            return True

        last_ball_pos, last_frame = self.ball_history[-1]

        if current_frame - last_frame <= self.speed_check_frames:
            distance_moved = point_distance(ball_pos, last_ball_pos)

            if distance_moved > self.max_ball_speed:
                return False

        return True

    def assign(self, tracks: Dict[str, Any], current_frame: int, penalty_point_1_pos: Tuple[float, float], penalty_point_2_pos: Tuple[float, float]) -> Tuple[Dict[str, Any], int]:
        """
        Assigns the ball to the nearest player based on various criteria.
        Returns: Updated tracks and the ID of the player with the ball.
        """
        tracks = tracks.copy()
        player_w_ball = -1
        # Final entity to be credited for the frame: -1, Club1 Name, or Club2 Name
        possessing_entity = -1

        valid_ball_tracks = []
        best_ball_key = None
        best_ball_pos = None
        to_delete = []

        # --- 1. Ball Validation & Filtering ---
        if 'ball' in tracks and tracks['ball']:
            self.ball_exists = False

            for ball_key, ball_data in tracks['ball'].items():
                ball_pos = ball_data.get('projection')
                if ball_pos is None:
                    continue

                ball_bbox_center = get_bbox_center(ball_data['bbox'])

                is_near_penalty_point = False
                if penalty_point_1_pos is not None and point_distance(ball_bbox_center, penalty_point_1_pos) < self.penalty_point_distance:
                    is_near_penalty_point = True
                if penalty_point_2_pos is not None and point_distance(ball_bbox_center, penalty_point_2_pos) < self.penalty_point_distance:
                    is_near_penalty_point = True

                if not is_near_penalty_point and self.is_ball_movement_valid(ball_pos, current_frame):
                    valid_ball_tracks.append((ball_key, ball_pos))
                else:
                    to_delete.append(ball_key)

        # --- 2. Determine Current Possession ---

        # A. Valid Ball Found: Find the closest player
        if valid_ball_tracks:
            self.ball_exists = True
            self.ball_lost_frame = None  # Reset ball lost timer

            min_dis = self.max_ball_distance
            players = {**tracks.get('player', {}), **
                       tracks.get('goalkeeper', {})}

            for ball_key, ball_pos in valid_ball_tracks:
                for player_id, player in players.items():
                    player_pos = player.get('projection')
                    if player_pos is None:
                        continue

                    dis = point_distance(ball_pos, player_pos)

                    if dis <= min_dis:
                        min_dis = dis
                        player_w_ball = player_id
                        best_ball_key, best_ball_pos = ball_key, ball_pos

            # Update ball history
            if best_ball_key is not None:
                self.ball_history.append((best_ball_pos, current_frame))
                if len(self.ball_history) > self.speed_check_frames:
                    self.ball_history.pop(0)

            # Assign to player or use grace period
            if player_w_ball != -1 and 'club' in players[player_w_ball]:
                # Found a new player in range
                possessing_entity = players[player_w_ball]['club']
                self.last_player_w_ball = player_w_ball
                self.last_possession_frame = current_frame
                self.last_possessing_team = possessing_entity
            else:
                # No player in range, check grace period (player just lost the ball)
                if self.last_player_w_ball is not None and self.last_possession_frame is not None:
                    # Both must be non-None to perform arithmetic
                    elapsed_frames = current_frame - self.last_possession_frame

                    if elapsed_frames <= self.grace_period_frames:
                        # Retain possession via grace period
                        player_w_ball = self.last_player_w_ball
                        possessing_entity = self.last_possessing_team
                    else:
                        # Grace period expired, possession is neutral
                        self.last_player_w_ball = None
                        possessing_entity = -1
                else:
                    # Neutral ball, no previous owner, or incomplete history
                    possessing_entity = -1

        # B. No Valid Ball Found (Ball Lost/Not Detected)
        else:
            self.ball_exists = False
            if self.ball_lost_frame is None:
                self.ball_lost_frame = current_frame

            # Check ball lost grace period (player retains possession if ball disappears briefly)
            if self.last_player_w_ball is not None:
                elapsed_frames_since_ball_seen = current_frame - self.ball_lost_frame

                if elapsed_frames_since_ball_seen <= self.ball_grace_period_frames:
                    # Retain possession via ball lost grace period
                    player_w_ball = self.last_player_w_ball
                    possessing_entity = self.last_possessing_team
                else:
                    # Ball lost period expired, possession is neutral
                    self.last_player_w_ball = None
                    possessing_entity = -1
            else:
                # Neutral ball, no player had it previously
                possessing_entity = -1

        # --- 3. Final Track & Possession Update (Consolidated) ---

        # A. Record Possession (Fixes Redundancy)
        self.possession_tracker.add_possession(possessing_entity)

        # B. Set 'has_ball' flag on the tracks
        for track_type in ['player', 'goalkeeper']:
            for pid, track_data in tracks.get(track_type, {}).items():
                if pid == player_w_ball:
                    tracks[track_type][pid]['has_ball'] = True
                else:
                    tracks[track_type][pid]['has_ball'] = False

        # C. Filter Ball Tracks (Original logic)
        for bid in to_delete:
            del tracks['ball'][bid]

        ball_tracks_cpy = tracks['ball'].copy()

        if best_ball_key:
            for bid in ball_tracks_cpy.keys():
                if bid != best_ball_key:
                    del tracks['ball'][bid]

        return tracks, player_w_ball

    def get_ball_possessions(self) -> Any:
        """
        Returns the current ball possessions tracked by the possession tracker.
        (Original method remains here)
        """
        return self.possession_tracker.get_ball_possessions()
