import os
import argparse
from pathlib import Path
from utils import process_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
from passing_prediction.config import DEFAULTS
from passing_prediction.pass_data_writer import PassDataWriter
import numpy as np

def main():
    # Anchor base directory to the file location
    BASE_DIR = Path(__file__).resolve().parent

    # CLI Argument Parsing
    parser = argparse.ArgumentParser(description="Automated Football Match Video Analytics System")
    parser.add_argument("--input", type=str, 
                        default=str(BASE_DIR / "assets" / "sample_match.mp4"), 
                        help="Path to input video")
    parser.add_argument("--output", type=str, 
                        default=str(BASE_DIR / "output_videos" / "output.mp4"), 
                        help="Path to output video")
    parser.add_argument("--obj-model", type=str, 
                        default=str(BASE_DIR / "models" / "weights" / "object-detection.pt"), 
                        help="Path to YOLO object detection model")
    parser.add_argument("--kp-model", type=str, 
                        default=str(BASE_DIR / "models" / "weights" / "keypoints-detection.pt"), 
                        help="Path to YOLO keypoints detection model")
    parser.add_argument("--field-img", type=str, 
                        default=str(BASE_DIR / "assets" / "field_2d_v2.png"), 
                        help="Path to 2D field image")
    args = parser.parse_args()

    # Ensure output directories exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_tracks_dir = output_path.parent

    # Initialize Trackers
    obj_tracker = ObjectTracker(
        model_path=args.obj_model,
        conf=.5,
        ball_conf=.05
    )

    kp_tracker = KeypointsTracker(
        model_path=args.kp_model,
        conf=.3,
        kp_conf=.7,
    )

    # Define Clubs
    club1 = Club('Club1', (232, 247, 248), (6, 25, 21))
    club2 = Club('Club2', (172, 251, 145), (239, 156, 132))

    club_assigner = ClubAssigner(club1, club2)
    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    # Standard Pitch Keypoints (Top-Down)
    top_down_keypoints = np.array([
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],
        [32, 122], [32, 229], [64, 176], [96, 57], [96, 122], [96, 229], [96, 293],
        [263, 0], [263, 122], [263, 229], [263, 351],
        [431, 57], [431, 122], [431, 229], [431, 293], [463, 176],
        [495, 122], [495, 229], [527, 0], [527, 57], [527, 122], [527, 229],
        [527, 293], [527, 351], [210, 176], [317, 176]
    ])

    # Initialize Processor
    processor = FootballVideoProcessor(obj_tracker,
                                       kp_tracker,
                                       club_assigner,
                                       ball_player_assigner,
                                       top_down_keypoints,
                                       field_img_path=args.field_img,
                                       save_tracks_dir=str(save_tracks_dir),
                                       draw_frame_num=True
                                       )

    # Run Pipeline
    process_video(processor,
                  video_source=args.input,
                  output_video=args.output,
                  batch_size=10
                  )

if __name__ == '__main__':
    main()
