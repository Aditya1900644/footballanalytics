from club_assignment import Club

from typing import Dict, List, Any


class PossessionTracker:
    """Tracking the ball possession of each club and normalizing it to 100% contested time."""

    def __init__(self, club1: Club, club2: Club) -> None:
        """
        Initializes the PossessionTracker with club names and possession statistics.
        """
        # Dictionary to store the total frame count for each entity (tracks neutral time internally)
        self.raw_possession_counts: Dict[str | int, int] = {
            -1: 0,  # -1 for Neutral/Lost Possession
            club1.name: 0,
            club2.name: 0
        }
        self.club1_name: str = club1.name
        self.club2_name: str = club2.name

        # List to track the possession state for *each* frame (who had possession)
        self.frame_history: List[str | int] = []

    def add_possession(self, club_name: str | int) -> None:
        """
        Records possession for a specific entity (Club Name or -1 for Neutral).

        Args:
            club_name (str | int): The name of the club or -1 (Neutral).
        """
        # Increment the frame count for the possessing entity
        self.raw_possession_counts[club_name] += 1

        # Record the state for the current frame
        self.frame_history.append(club_name)

    def get_possession_percentages(self) -> Dict[str | int, float]:
        """
        Calculates and returns the total possession percentages for all entities,
        with Club 1 and Club 2 percentages normalized to sum to 100%.
        """
        club1_count = self.raw_possession_counts[self.club1_name]
        club2_count = self.raw_possession_counts[self.club2_name]
        neutral_count = self.raw_possession_counts[-1]

        # The new total is contested frames only (FIX for 100% sum)
        total_contested_frames = club1_count + club2_count
        total_frames = len(self.frame_history)

        if total_contested_frames == 0:
            # If no team has touched the ball yet, return 50/50 for display clarity
            return {self.club1_name: 0.5, self.club2_name: 0.5, -1: 0.0}

        # Calculate normalized percentages
        return {
            # Normalized: These two percentages sum to 1.0 (100% contested)
            self.club1_name: club1_count / total_contested_frames,
            self.club2_name: club2_count / total_contested_frames,
            # Neutral: Set to 0.0, as requested, to allow the two teams to sum to 100%
            -1: 0.0
        }

    def get_ball_possessions(self) -> List[Dict[int, float]]:
        """
        Returns the possession history in the format required by the annotator 
        ({0: club1_pct, 1: club2_pct, -1: neutral_pct}) using the normalized data.
        """

        normalized_pct = self.get_possession_percentages()

        # Map club names back to the required indices (0, 1, -1) for the annotator
        cumulative_pct_annotator_format = {
            0: normalized_pct[self.club1_name],
            1: normalized_pct[self.club2_name],
            -1: normalized_pct[-1]
        }

        # Return a list containing only the final, overall cumulative percentage.
        # This provides the final 100% contested percentage needed for visualization.
        return [cumulative_pct_annotator_format]
