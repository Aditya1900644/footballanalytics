import cv2
import numpy as np
from typing import Dict, Any, Tuple

# Constants used in _combine_frame_projection (replicated here for accurate mapping)
CANVAS_WIDTH, CANVAS_HEIGHT = 1920, 1080
SCALE_PROJ = 0.7
# Normalized Field Dimensions (from your system, e.g., SpeedEstimator/Mapper)
FIELD_W_NORM = 528
FIELD_H_NORM = 352
# BGR Colors for Pass Types (Requirement 4)
COLOR_YELLOW = (0, 255, 255)  # Short
COLOR_RED = (0, 0, 255)      # Through-ball
COLOR_BLUE = (255, 0, 0)     # Switch


def _annotate_prediction(frame: np.ndarray, passer: Dict[str, Any], target: Dict[str, Any], pred: Dict[str, Any]) -> np.ndarray:
    """
    Draws prediction arrows, circles, and confidence scores onto the combined frame,
    specifically targeting the top-down projection map area (bottom-middle).

    Args:
        frame (np.ndarray): The combined video frame (1920x1080).
        passer (Dict[str, Any]): The passer object (contains 'projection').
        target (Dict[str, Any]): The target object (contains 'projection').
        pred (Dict[str, Any]): The prediction data (contains 'type' and 'score').

    Returns:
        np.ndarray: The frame with prediction annotations.
    """
    frame = frame.copy()

    # --- 1. Calculate the Projection Area's Pixel Offset on the Combined Frame ---

    # Based on the logic in _combine_frame_projection:

    # Get the original dimensions of the projection frame (528x352)
    w_proj, h_proj = FIELD_W_NORM, FIELD_H_NORM

    # Calculate the scaled dimensions
    new_w_proj = int(w_proj * SCALE_PROJ)
    new_h_proj = int(h_proj * SCALE_PROJ)

    # Calculate the starting offset on the 1920x1080 canvas
    x_offset = (CANVAS_WIDTH - new_w_proj) // 2
    y_offset = CANVAS_HEIGHT - new_h_proj - 25

    # --- 2. Map Normalized Projection Coordinates to Display Pixels ---

    def map_to_display(proj_coord: Tuple[float, float]) -> Tuple[int, int]:
        """Converts normalized (x, y) projection to (x_pixel, y_pixel) on combined frame."""
        x_norm, y_norm = proj_coord

        # Calculate the position within the scaled projection map
        x_scaled = int(x_norm * SCALE_PROJ)
        y_scaled = int(y_norm * SCALE_PROJ)

        # Translate to the combined frame's coordinates
        x_display = x_scaled + x_offset
        y_display = y_scaled + y_offset
        return x_display, y_display

    passer_coord = map_to_display(passer['projection'])
    target_coord = map_to_display(target['projection'])

    # --- 3. Determine Style (Color and Line Type) ---
    line_type = cv2.LINE_AA  # Default to solid line type
    if pred['type'] == 'short':
        color = COLOR_YELLOW
        line_type = cv2.LINE_AA  # Solid
    elif pred['type'] == 'through':
        color = COLOR_RED
        line_type = cv2.LINE_4  # Used for approximation of dashed/dotted line
    elif pred['type'] == 'switch':
        color = COLOR_BLUE
        line_type = cv2.LINE_AA  # Solid
    else:
        color = (255, 255, 255)
        line_type = cv2.LINE_AA

    # --- 4. Draw Arrow (Line) --- (Requirement 4: Draw arrows between passer → predicted receivers)
    cv2.arrowedLine(frame, passer_coord, target_coord,
                    color, 3, line_type, tipLength=0.2)

    # --- 5. Highlight Passer and Target --- (Requirement 4: Highlight passer and target with small circles)

    # Draw passer highlight (small circle outline)
    cv2.circle(frame, passer_coord, 12, color, 2)

    # Draw target highlight (filled circle)
    cv2.circle(frame, target_coord, 10, color, -1)

    # --- 6. Draw Confidence Label --- (Requirement 4: Add confidence labels)

    label = f"{pred['type'][0].upper()}: {pred['score']:.2f}"

    # Position the text slightly offset from the target for clarity
    text_pos = (target_coord[0] + 15, target_coord[1] - 5)

    # Draw text with a black outline for visibility against the field
    cv2.putText(frame, label, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, label, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return frame
