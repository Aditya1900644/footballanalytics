import streamlit as st
import cv2
import numpy as np
import os
import json
import pandas as pd
import plotly.express as px
from collections import deque
import time

# --- Import Project Modules ---
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
from passing_prediction.config import DEFAULTS

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Football AI Command Center",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .stApp { background-color: #0E1117; }
        .stMetric { background-color: #262730; padding: 10px; border-radius: 10px; }
        div[data-testid="stExpander"] { background-color: #262730; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_videos"
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_tuple):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_tuple[0]), int(rgb_tuple[1]), int(rgb_tuple[2]))


@st.cache_resource
def load_models():
    obj_tracker = ObjectTracker(
        model_path='models/weights/object-detection.pt', conf=0.5, ball_conf=0.05)
    kp_tracker = KeypointsTracker(
        model_path='models/weights/keypoints-detection.pt', conf=0.3, kp_conf=0.7)
    return obj_tracker, kp_tracker


def auto_detect_team_colors(video_path, tracker):
    """
    Reads the first valid frame, detects players, masks out the green field, 
    and clusters jersey colors to automatically find Team 1 and Team 2.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "#FFFFFF", "#000000"  # Fallback

    # 1. Detect Objects
    results = tracker.detect([frame])
    result = results[0]

    player_crops = []

    # --- Masking Constants (Green Field) ---
    LOWER_GREEN = np.array([36, 25, 25])
    UPPER_GREEN = np.array([86, 255, 255])

    # 2. Extract & Mask Player Crops
    for box in result.boxes:
        class_id = int(box.cls[0])
        if class_id == 0:  # Person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1

            if w > 0 and h > 0:
                # Take upper half (Jersey area)
                crop = frame[y1:y1+int(h*0.5), x1:x1+w]

                if crop.size > 0:
                    # --- APPLY GREEN MASK ---
                    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv_crop, LOWER_GREEN, UPPER_GREEN)

                    # Keep only pixels that are NOT green (mask == 0)
                    pixels = crop.reshape(-1, 3)
                    mask_flat = mask.reshape(-1)
                    valid_pixels = pixels[mask_flat == 0]

                    if len(valid_pixels) > 0:
                        # Randomly sample pixels to save memory if crop is large
                        if len(valid_pixels) > 200:
                            indices = np.random.choice(
                                len(valid_pixels), 200, replace=False)
                            valid_pixels = valid_pixels[indices]

                        player_crops.append(valid_pixels)

    if len(player_crops) < 2:
        return "#E8F7F8", "#ACFB91"  # Fallback if detection fails

    # 3. Stack all valid pixels from all players
    data = np.vstack(player_crops)
    data = np.float32(data)

    # 4. K-Means Clustering (K=2 for two main teams)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _, labels, centers = cv2.kmeans(
        data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    # 5. Return Detected Colors
    color1 = rgb_to_hex(centers[0])
    color2 = rgb_to_hex(centers[1])

    return color1, color2


def main():
    st.title("⚽ Football Analysis AI")

    # Initialize Session State
    if 'team1_color' not in st.session_state:
        st.session_state['team1_color'] = "#E8F7F8"
    if 'team2_color' not in st.session_state:
        st.session_state['team2_color'] = "#ACFB91"
    if 'auto_detected' not in st.session_state:
        st.session_state['auto_detected'] = False

    with st.sidebar:
        st.header("⚙️ Match Configuration")
        uploaded_file = st.file_uploader(
            "Upload Match Video", type=['mp4', 'mov'])

        st.divider()
        st.subheader("Team Colors")

        # Color Pickers (Auto-updated via session_state)
        c1_hex = st.color_picker(
            "Team 1 Color", st.session_state['team1_color'])
        c2_hex = st.color_picker(
            "Team 2 Color", st.session_state['team2_color'])

        run_btn = st.button("🚀 Start Analysis", type="primary")

    if uploaded_file:
        # Use timestamp to create unique filename
        timestamp = int(time.time())
        unique_filename = f"{uploaded_file.name.split('.')[0]}_{timestamp}.mp4"
        input_path = os.path.join(INPUT_DIR, unique_filename)

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # --- AUTO DETECT TRIGGER ---

        if not st.session_state['auto_detected']:
            with st.spinner("🤖 Auto-detecting Jersey Colors (Filtering Grass)..."):
                obj_tracker, _ = load_models()
                detected_c1, detected_c2 = auto_detect_team_colors(
                    input_path, obj_tracker)

                st.session_state['team1_color'] = detected_c1
                st.session_state['team2_color'] = detected_c2
                st.session_state['auto_detected'] = True
                st.rerun()  # Refresh sidebar to show new colors

        output_video_path = os.path.join(
            OUTPUT_DIR, f"analyzed_{unique_filename}")
        threat_data_path = os.path.join(OUTPUT_DIR, "threat_data.json")

        if run_btn:
            obj_tracker, kp_tracker = load_models()

            club1 = Club('Team 1', hex_to_rgb(c1_hex), (255, 255, 255))
            club2 = Club('Team 2', hex_to_rgb(c2_hex), (255, 255, 255))

            club_assigner = ClubAssigner(club1, club2)
            ball_assigner = BallToPlayerAssigner(club1, club2)

            top_down_keypoints = np.array([
                [0, 0], [0, 57], [0, 122], [0, 229], [0, 293], [0, 351],
                [32, 122], [32, 229], [64, 176], [96, 57], [
                    96, 122], [96, 229], [96, 293],
                [263, 0], [263, 122], [263, 229], [263, 351],
                [431, 57], [431, 122], [431, 229], [431, 293], [463, 176],
                [495, 122], [495, 229], [527, 0], [
                    527, 57], [527, 122], [527, 229],
                [527, 293], [527, 351], [210, 176], [317, 176]
            ])

            processor = FootballVideoProcessor(
                obj_tracker, kp_tracker, club_assigner, ball_assigner,
                top_down_keypoints, field_img_path='input_videos/field_2d_v2.png',
                save_tracks_dir=OUTPUT_DIR
            )

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Inside app.py
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(f"DEBUG: Input FPS detected as: {fps}")
            print(f"DEBUG: Total Frames: {total_frames}")

            # Attempt AVC1 (H.264) for browser compatibility, fallback to MP4V
            try:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))

            status_text = st.empty()
            progress_bar = st.progress(0)
            frames_batch = []
            batch_size = 10
            frame_count = 0
            threat_history = []

            status_text.info(f"Processing {total_frames} frames...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames_batch.append(frame)

                if len(frames_batch) == batch_size:
                    processed_frames = processor.process(frames_batch, fps=fps)
                    for p_frame in processed_frames:
                        out.write(p_frame)
                        threat_history.append({
                            "frame": frame_count,
                            "time": frame_count / fps,
                            "threat": processor.current_threat * 100
                        })
                        frame_count += 1

                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing... {int(progress*100)}%")
                    frames_batch = []

            if frames_batch:
                processed_frames = processor.process(frames_batch, fps=fps)
                for p_frame in processed_frames:
                    out.write(p_frame)
                    threat_history.append(
                        {"frame": frame_count, "time": frame_count/fps, "threat": processor.current_threat * 100})
                    frame_count += 1

            cap.release()
            out.release()

            # Save data
            with open(threat_data_path, 'w') as f:
                json.dump(threat_history, f)

            # Generate Heatmaps
            processor.generate_heatmaps()

            status_text.success("Analysis Complete!")
            st.rerun()

        if os.path.exists(output_video_path):
            show_dashboard(output_video_path, threat_data_path)


def show_dashboard(video_path, threat_data_path):
    st.divider()

    if os.path.exists(threat_data_path):
        with open(threat_data_path, 'r') as f:
            threat_data = json.load(f)
        df_threat = pd.DataFrame(threat_data)
    else:
        df_threat = pd.DataFrame()

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("📺 Match Feed")
        st.video(video_path)

        if not df_threat.empty:
            st.subheader("📈 Match Momentum")
            fig = px.area(df_threat, x="time", y="threat",
                          labels={"time": "Time (s)", "threat": "Threat %"},
                          color_discrete_sequence=["#FF4B4B"])
            fig.update_layout(
                plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                font_color="white", height=250, margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.subheader("📊 Analytics Panel")
        tabs = st.tabs(["Match Stats", "Player Analysis"])

        with tabs[0]:
            pred_path = "output_videos/predicted_passes.json"
            if os.path.exists(pred_path):
                with open(pred_path, 'r') as f:
                    preds = json.load(f)
                st.metric("Total Predictions", len(preds))
                df_preds = pd.DataFrame(preds)
                if not df_preds.empty:
                    high_val = df_preds[df_preds['score'] > 0.7]
                    # Fixed deprecation warning using width instead of use_container_width
                    st.dataframe(high_val[['frame', 'type', 'score']],
                                 hide_index=True, width=1000)
            else:
                st.info("No pass predictions found.")

        with tabs[1]:
            st.markdown("##### Heatmap Selector")
            heatmap_files = [f for f in os.listdir(
                HEATMAP_DIR) if f.endswith('.png')]
            if heatmap_files:
                heatmap_files.sort(key=lambda x: int(
                    x.split('_')[2].split('.')[0]))
                selected_map = st.selectbox(
                    "Select Player", heatmap_files,
                    format_func=lambda x: f"Player {x.split('_')[2].split('.')[0]}"
                )

                # Fixed deprecation warning
                st.image(os.path.join(HEATMAP_DIR, selected_map),
                         caption=f"Activity Map: {selected_map}",
                         use_container_width=True)
            else:
                st.warning("No heatmaps generated yet.")


if __name__ == "__main__":
    main()
