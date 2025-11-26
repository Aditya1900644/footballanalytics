# annotation/football_video_processor.py
from .abstract_annotator import AbstractAnnotator
from .abstract_video_processor import AbstractVideoProcessor
from .object_annotator import ObjectAnnotator
from .keypoints_annotator import KeypointsAnnotator
from .projection_annotator import ProjectionAnnotator
from position_mappers import ObjectPositionMapper
from speed_estimation import SpeedEstimator
from .frame_number_annotator import FrameNumberAnnotator
from file_writing import TracksJsonWriter
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner
from ball_to_player_assignment import BallToPlayerAssigner
from utils import rgb_bgr_converter

# --- Custom Modules ---
from passing_prediction.pass_trigger import PassTrigger
from passing_prediction.pass_predictor import PassingPredictor
from passing_prediction.pass_data_writer import PassDataWriter
from passing_prediction.config import DEFAULTS
from passing_prediction.realtime_threat_estimator import RealTimeThreatEstimator

# --- NEW: Heatmap Analysis ---
from analysis.heatmap_generator import HeatmapGenerator
from collections import defaultdict

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import json
import os


# --- GLOBAL CONSTANTS ---
CANVAS_WIDTH, CANVAS_HEIGHT = 1920, 1080
SCALE_PROJ = 0.7
FIELD_W_NORM = 528
FIELD_H_NORM = 352

# BGR Colors
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)


class FootballVideoProcessor(AbstractAnnotator, AbstractVideoProcessor):
    def __init__(self, obj_tracker: ObjectTracker, kp_tracker: KeypointsTracker,
                 club_assigner: ClubAssigner, ball_to_player_assigner: BallToPlayerAssigner,
                 top_down_keypoints: np.ndarray, field_img_path: str,
                 save_tracks_dir: Optional[str] = None, draw_frame_num: bool = True) -> None:

        self.obj_tracker = obj_tracker
        self.obj_annotator = ObjectAnnotator()
        self.kp_tracker = kp_tracker
        self.kp_annotator = KeypointsAnnotator()
        self.club_assigner = club_assigner
        self.ball_to_player_assigner = ball_to_player_assigner
        self.projection_annotator = ProjectionAnnotator()
        self.obj_mapper = ObjectPositionMapper(top_down_keypoints)
        self.draw_frame_num = draw_frame_num
        if self.draw_frame_num:
            self.frame_num_annotator = FrameNumberAnnotator()

        self.save_tracks_dir = save_tracks_dir
        if save_tracks_dir:
            self.writer = TracksJsonWriter(save_tracks_dir)

        self.field_img_path = field_img_path  # Store path for heatmap gen
        field_image = cv2.imread(field_img_path)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_BGR2GRAY)
        field_image = cv2.cvtColor(field_image, cv2.COLOR_GRAY2BGR)

        self.speed_estimator = SpeedEstimator(
            field_image.shape[1], field_image.shape[0])

        self.frame_num = 0
        self.field_image = field_image

        # --- Advanced Analytics ---
        self.pass_cfg = DEFAULTS.copy()
        self.event_trigger = PassTrigger(fps=30.0)
        self.pass_predictor = PassingPredictor()
        self.pass_data_writer = PassDataWriter(
            save_dir=save_tracks_dir or "output_videos")
        self._predictions_out: List[Dict[str, Any]] = []

        self.threat_estimator = RealTimeThreatEstimator()
        self.current_threat = 0.0

        self._img_buffer: deque[np.ndarray] = deque(
            maxlen=self.pass_cfg["buffer_frames"])

        # --- NEW: Heatmap Data Accumulator ---
        self.player_positions = defaultdict(list)

        # Ensure output directories
        os.makedirs(save_tracks_dir or "output_videos", exist_ok=True)
        os.makedirs(os.path.join(save_tracks_dir or "output_videos",
                    "pass_clips"), exist_ok=True)

    # --- VISUALIZATION HELPERS ---
    def _annotate_prediction(self, frame: np.ndarray, passer: Dict[str, Any],
                             predictions: List[Dict[str, Any]],
                             all_frame_objects: List[Dict[str, Any]]) -> np.ndarray:
        frame = frame.copy()

        def map_to_display(proj_coord):
            x_norm, y_norm = proj_coord
            new_w_proj = int(FIELD_W_NORM * SCALE_PROJ)
            x_scaled = int(x_norm * SCALE_PROJ)
            y_scaled = int(y_norm * SCALE_PROJ)
            x_display = x_scaled + (CANVAS_WIDTH - new_w_proj) // 2
            y_display = y_scaled + \
                (CANVAS_HEIGHT - int(FIELD_H_NORM * SCALE_PROJ) - 25)
            return x_display, y_display

        passer_coord = map_to_display(passer['projection'])
        cv2.circle(frame, passer_coord, 12, COLOR_YELLOW, 2)

        for pred in predictions:
            target_player = next(
                (p for p in all_frame_objects if p["id"] == pred["target_id"]), None)
            if target_player is None:
                continue

            target_coord = map_to_display(target_player['projection'])

            if pred['type'] == 'short':
                color = COLOR_YELLOW
                current_line_type = cv2.LINE_AA
            elif pred['type'] == 'through':
                color = COLOR_RED
                current_line_type = cv2.LINE_4
            elif pred['type'] == 'switch':
                color = COLOR_BLUE
                current_line_type = cv2.LINE_AA
            else:
                color = (255, 255, 255)
                current_line_type = cv2.LINE_AA

            cv2.line(frame, passer_coord, target_coord,
                     color, 3, current_line_type)

            label = f"{pred['type'][0].upper()}: {pred['score']:.2f}"
            text_pos = (target_coord[0] + 15, target_coord[1] - 5)
            cv2.putText(frame, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            cv2.circle(frame, target_coord, 10, color, -1)
        return frame

    def _annotate_threat(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()
        bar_x, bar_y = 1550, 40
        bar_w, bar_h = 250, 25
        viz_threat = min(self.current_threat, 1.0)
        fill_w = int(bar_w * viz_threat)

        color = (0, 255, 0)
        if viz_threat > 0.2:
            color = (0, 165, 255)
        if viz_threat > 0.5:
            color = (0, 0, 255)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w,
                      bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w,
                      bar_y + bar_h), (255, 255, 255), 2)

        label = f"THREAT: {viz_threat * 100:.1f}%"
        cv2.putText(frame, label, (bar_x, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

    # --- MAIN LOOP ---
    def process(self, frames: List[np.ndarray], fps: float = 1e-6) -> List[np.ndarray]:
        self.cur_fps = max(fps, 1e-6)
        self.event_trigger.fps = self.cur_fps

        batch_obj_detections = self.obj_tracker.detect(frames)
        batch_kp_detections = self.kp_tracker.detect(frames)
        processed_frames = []

        for idx, (frame, object_detection, kp_detection) in enumerate(zip(frames, batch_obj_detections, batch_kp_detections)):
            obj_tracks = self.obj_tracker.track(object_detection)
            kp_tracks = self.kp_tracker.track(kp_detection)
            obj_tracks = self.club_assigner.assign_clubs(frame, obj_tracks)
            all_tracks = {'object': obj_tracks, 'keypoints': kp_tracks}
            all_tracks = self.obj_mapper.map(all_tracks)
            all_tracks['object'], _ = self.ball_to_player_assigner.assign(
                all_tracks['object'], self.frame_num,
                all_tracks['keypoints'].get(8, None),
                all_tracks['keypoints'].get(24, None)
            )
            all_tracks['object'] = self.speed_estimator.calculate_speed(
                all_tracks['object'], self.frame_num, self.cur_fps
            )

            if hasattr(self, "save_tracks_dir") and self.save_tracks_dir:
                self._save_tracks(all_tracks)

            # --- 1. COLLECT DATA ---
            frame_objects = []
            for track_type in ['player', 'goalkeeper', 'ball']:
                for obj_id, obj in all_tracks['object'].get(track_type, {}).items():
                    if 'projection' in obj:
                        # Store for Heatmap (Accumulate history)
                        if track_type != 'ball':
                            self.player_positions[obj_id].append(
                                obj['projection'])

                        pnorm = (float(obj['projection'][0]),
                                 float(obj['projection'][1]))
                        fo = {
                            "frame": self.frame_num,
                            "id": obj_id,
                            "class": track_type,
                            "team": obj.get("club") or obj.get("team"),
                            "projection": pnorm,
                            "has_ball": bool(obj.get("has_ball", False)),
                            "speed": float(obj.get("speed", 0.0))
                        }
                        frame_objects.append(fo)

            # 2. Analytics Updates
            event = self.event_trigger.update(self.frame_num, frame_objects)
            self.current_threat = self.threat_estimator.get_threat_score(
                frame_objects)

            annotated_frame = self.annotate(frame, all_tracks)
            self._img_buffer.append(annotated_frame.copy())

            # 3. Prediction Event Logic
            if event:
                passer_id = event["passer_id"]
                passer = next(
                    (p for p in frame_objects if p["id"] == passer_id), None)
                if passer is not None:
                    teammates = [p for p in frame_objects if p.get("team") == passer.get("team")
                                 and p.get("id") != passer.get("id") and p.get("class") in ['player', 'goalkeeper']]
                    opponents = [p for p in frame_objects if p.get("team") != passer.get("team")
                                 and p.get("class") in ['player', 'goalkeeper']]

                    preds = self.pass_predictor.rank_candidates(
                        passer, teammates, opponents, topk=3, teammate_vels={})
                    preds = [p for p in preds if p["score"]
                             >= self.pass_cfg["score_thresh"]]

                    if preds:
                        frame_to_annotate = self._img_buffer[-1].copy()
                        frame_to_annotate = self._annotate_prediction(
                            frame_to_annotate, passer, preds, frame_objects)
                        self._img_buffer[-1] = frame_to_annotate.copy()

                        for _ in range(self.pass_cfg["pause_frames"]):
                            processed_frames.append(frame_to_annotate.copy())

                        for p in preds:
                            p_record = {
                                "frame": int(event["frame"]),
                                "passer_id": int(p["passer_id"]),
                                "target_id": int(p["target_id"]),
                                "type": p.get("type"),
                                "score": float(p["score"]),
                                "distance": float(p["distance"]),
                                "meta": p.get("meta"),
                                "reason": event.get("reason", [])
                            }
                            self._predictions_out.append(p_record)

            processed_frames.append(self._img_buffer[-1])
            self.frame_num += 1

        # --- END OF LOOP ---
        # Note: We do NOT generate heatmaps here anymore to avoid 'AttributeError'
        # and performance issues during batch processing.

        self.pass_data_writer.write_all(self._predictions_out)
        print(f"Saved pass predictions to: {self.pass_data_writer.get_path()}")

        return processed_frames

    # --- NEW METHOD: Called by app.py after loop ---
    def generate_heatmaps(self):
        """
        Generates heatmaps for the top active players. 
        Called explicitly after video processing is complete.
        """
        print("Generating Heatmaps for Top Players...")

        # Sort players by duration (number of frames appeared)
        sorted_players = sorted(
            self.player_positions.items(), key=lambda x: len(x[1]), reverse=True)

        # Select Top 25 active IDs (Covers 11 vs 11 + subs/refs/tracking errors)
        top_active_players = sorted_players[:25]

        output_dir = os.path.join(
            self.save_tracks_dir or "output_videos", "heatmaps")
        os.makedirs(output_dir, exist_ok=True)

        hm_gen = HeatmapGenerator(self.field_img_path, output_dir)

        for p_id, positions in top_active_players:
            hm_gen.generate_single(p_id, positions)

        print(f"Heatmaps generated for {len(top_active_players)} players.")

    # --- ANNOTATE ---

    def annotate(self, frame: np.ndarray, tracks: Dict) -> np.ndarray:
        if self.draw_frame_num:
            frame = self.frame_num_annotator.annotate(
                frame, {'frame_num': self.frame_num})
        frame = self.kp_annotator.annotate(frame, tracks['keypoints'])
        frame = self.obj_annotator.annotate(frame, tracks['object'])
        projection_frame = self.projection_annotator.annotate(
            self.field_image, tracks['object'])
        combined_frame = self._combine_frame_projection(
            frame, projection_frame)
        combined_frame = self._annotate_possession(combined_frame)
        combined_frame = self._annotate_threat(combined_frame)
        return combined_frame

    def _combine_frame_projection(self, frame: np.ndarray, projection_frame: np.ndarray) -> np.ndarray:
        h_frame, w_frame, _ = frame.shape
        h_proj, w_proj, _ = projection_frame.shape
        scale_proj = SCALE_PROJ
        new_w_proj = int(w_proj * scale_proj)
        new_h_proj = int(h_proj * scale_proj)
        projection_resized = cv2.resize(
            projection_frame, (new_w_proj, new_h_proj))
        combined_frame = np.zeros(
            (CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        combined_frame[:h_frame, :w_frame] = frame
        x_offset = (CANVAS_WIDTH - new_w_proj) // 2
        y_offset = CANVAS_HEIGHT - new_h_proj - 25
        alpha = 0.75
        overlay = combined_frame[y_offset:y_offset +
                                 new_h_proj, x_offset:x_offset + new_w_proj]
        cv2.addWeighted(projection_resized, alpha,
                        overlay, 1 - alpha, 0, overlay)
        return combined_frame

    def _annotate_possession(self, frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()
        overlay = frame.copy()
        overlay_width, overlay_height = 500, 100
        gap_x, gap_y = 20, 20
        cv2.rectangle(overlay, (gap_x, gap_y), (gap_x +
                      overlay_width, gap_y + overlay_height), (0, 0, 0), -1)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        text_x, text_y = gap_x + 15, gap_y + 30
        cv2.putText(frame, 'Possession:', (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1)
        bar_x, bar_y = text_x, text_y + 25
        bar_width, bar_height = overlay_width - bar_x, 15
        possession_list = self.ball_to_player_assigner.get_ball_possessions()
        if not possession_list:
            possession = {-1: 1.0, 0: 0.0, 1: 0.0}
        else:
            possession = possession_list[-1]
        possession_club1 = possession[0]
        possession_club2 = possession[1]
        club1_width = int(bar_width * possession_club1)
        club2_width = int(bar_width * possession_club2)
        neutral_width = bar_width - club1_width - club2_width
        club1_color = rgb_bgr_converter(
            self.club_assigner.club1.player_jersey_color)
        club2_color = rgb_bgr_converter(
            self.club_assigner.club2.player_jersey_color)
        neutral_color = (128, 128, 128)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + club1_width,
                      bar_y + bar_height), club1_color, -1)
        cv2.rectangle(frame, (bar_x + club1_width, bar_y), (bar_x +
                      club1_width + neutral_width, bar_y + bar_height), neutral_color, -1)
        cv2.rectangle(frame, (bar_x + club1_width + neutral_width, bar_y),
                      (bar_x + bar_width, bar_y + bar_height), club2_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width,
                      bar_y + bar_height), (0, 0, 0), 2)
        self._display_possession_text(frame, club1_width, club2_width, neutral_width, bar_x, bar_y,
                                      f'{int(possession_club1 * 100)}%', f'{int(possession_club2 * 100)}%', club1_color, club2_color)
        return frame

    def _display_possession_text(self, frame: np.ndarray, club1_width: int, club2_width: int, neutral_width: int, bar_x: int, bar_y: int, possession_club1_text: str, possession_club2_text: str, club1_color: Tuple[int, int, int], club2_color: Tuple[int, int, int]) -> None:
        club1_text_x = bar_x + club1_width // 2 - 10
        club1_text_y = bar_y + 35
        cv2.putText(frame, possession_club1_text, (club1_text_x,
                    club1_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, possession_club1_text, (club1_text_x,
                    club1_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, club1_color, 1)
        club2_text_x = bar_x + club1_width + neutral_width + club2_width // 2 - 10
        club2_text_y = bar_y + 35
        cv2.putText(frame, possession_club2_text, (club2_text_x,
                    club2_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, possession_club2_text, (club2_text_x,
                    club2_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, club2_color, 1)

    def _save_tracks(self, all_tracks: Dict[str, Dict[int, np.ndarray]]) -> None:
        self.writer.write(self.writer.get_object_tracks_path(),
                          all_tracks['object'])
        self.writer.write(
            self.writer.get_keypoints_tracks_path(), all_tracks['keypoints'])
