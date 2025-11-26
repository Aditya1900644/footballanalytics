import os
import json
from typing import List, Dict, Any, Optional

# Assuming AbstractWriter and _make_serializable (from TracksJsonWriter) are available
# or that the input data is already serializable.
# We'll use a simple approach here since the data is mostly primitive types.


class PassDataWriter:
    """
    A class to write pass prediction data to a JSON file.
    Designed to append records incrementally during processing.
    """

    def __init__(self, save_dir: str = 'output_videos', filename: str = 'predicted_passes.json') -> None:
        """
        Initializes the PassDataWriter.

        Args:
            save_dir (str): Directory to save JSON files.
            filename (str): Filename for pass predictions.
        """
        self.save_path = os.path.join(save_dir, filename)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Clear existing file at start
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
            # Initialize with an empty list structure
            with open(self.save_path, 'w') as f:
                json.dump([], f, indent=2)

    def write_event(self, event_data: Dict[str, Any]) -> None:
        """
        Appends a single pass event record to the JSON file.
        This allows for incremental saving without high memory usage.

        Args:
            event_data (Dict[str, Any]): The single pass prediction record to write.
        """
        # Note: This is a robust but potentially slow way to append to a JSON array.
        # For performance, one might save all to a list and write once at the end.
        # Given the FootballVideoProcessor stores to a list, we'll assume the list is
        # dumped once at the end of processing, as done in the existing processor code.
        # This class will be a placeholder, but the logic in the processor will handle the writing.
        pass

    def write_all(self, all_predictions: List[Dict[str, Any]]) -> None:
        """
        Writes all accumulated predictions to the file.
        """
        with open(self.save_path, 'w') as f:
            # We don't need _make_serializable as the prediction data uses basic Python types (int, float, str)
            json.dump(all_predictions, f, indent=2)

    def get_path(self) -> str:
        """Returns the path to the predictions JSON file."""
        return self.save_path
