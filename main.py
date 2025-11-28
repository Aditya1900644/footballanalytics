import os
from utils import process_video
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
from passing_prediction.config import DEFAULTS
from passing_prediction.pass_data_writer import PassDataWriter
import numpy as np


def main():

    obj_tracker = ObjectTracker(
        model_path='models/weights/object-detection.pt',
        conf=.5,
        ball_conf=.05
    )

    kp_tracker = KeypointsTracker(
        model_path='models/weights/keypoints-detection.pt',
        conf=.3,
        kp_conf=.7,
    )

    club1 = Club('Club1',
                 (232, 247, 248),
                 (6, 25, 21)
                 )
    club2 = Club('Club2',
                 (172, 251, 145),
                 (239, 156, 132)
                 )

    club_assigner = ClubAssigner(club1, club2)

    ball_player_assigner = BallToPlayerAssigner(club1, club2)

    top_down_keypoints = np.array([
        [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [
            0, 351],             # 0-5 (left goal line)
        # 6-7 (left goal box corners)
        [32, 122], [32, 229],
        # 8 (left penalty dot)
        [64, 176],
        # 9-12 (left penalty box)
        [96, 57], [96, 122], [96, 229], [96, 293],
        # 13-16 (halfway line)
        [263, 0], [263, 122], [263, 229], [263, 351],
        # 17-20 (right penalty box)
        [431, 57], [431, 122], [431, 229], [431, 293],
        # 21 (right penalty dot)
        [463, 176],
        # 22-23 (right goal box corners)
        [495, 122], [495, 229],
        [527, 0], [527, 57], [527, 122], [527, 229], [
            527, 293], [527, 351],
        # 30-31 (center circle
        [210, 176], [317, 176]
    ])
    # 5b. Define Pass Predictor Configuration
    pass_cfg = DEFAULTS.copy()

    # 5c. Initialize the Pass Data Writer
    pass_data_writer = PassDataWriter(save_dir='output_videos')

    processor = FootballVideoProcessor(obj_tracker,
                                       kp_tracker,
                                       club_assigner,
                                       ball_player_assigner,
                                       top_down_keypoints,
                                       field_img_path='input_videos/field_2d_v2.png',  # Top-Down field image path
                                       save_tracks_dir='output_videos',
                                       # Whether or not to draw current frame number on
                                       draw_frame_num=True

                                       )

    process_video(processor,
                  video_source='input_videos/video2.mp4',
                  output_video='output_videos/testx.mp4',
                  batch_size=10
                  )


if __name__ == '__main__':
    main()
