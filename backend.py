# backend.py
import os
import shutil
import uuid
import json
import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- IMPORT YOUR MODULES ---
from tracking import ObjectTracker, KeypointsTracker
from club_assignment import ClubAssigner, Club
from ball_to_player_assignment import BallToPlayerAssigner
from annotation import FootballVideoProcessor
from analysis.heatmap_generator import HeatmapGenerator

app = FastAPI()

# Anchor base directory to the file location
BASE_DIR = Path(__file__).resolve().parent

# 1. Allow the Frontend to talk to us (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup Folders using Pathlib
UPLOAD_DIR = BASE_DIR / "input_videos"
OUTPUT_DIR = BASE_DIR / "output_videos"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# 3. Let the browser access the output videos/images
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Global Job Storage (In memory)
jobs = {}


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# --- THE WORKER FUNCTION (Runs in Background) ---


def run_analysis_task(job_id: str, input_path: str, c1_hex: str, c2_hex: str):
    print(f"[{job_id}] Starting Analysis...")
    try:
        # Setup Paths
        filename = os.path.basename(input_path)
        output_name = f"analyzed_{filename}"
        output_path = str(OUTPUT_DIR / output_name)

        # Initialize Models with anchored paths
        obj_tracker = ObjectTracker(
            str(BASE_DIR / 'models' / 'weights' / 'object-detection.pt'), 
            conf=0.5, ball_conf=0.05
        )
        kp_tracker = KeypointsTracker(
            str(BASE_DIR / 'models' / 'weights' / 'keypoints-detection.pt'), 
            conf=0.3, kp_conf=0.7
        )

        club1 = Club('Team 1', hex_to_rgb(c1_hex), (255, 255, 255))
        club2 = Club('Team 2', hex_to_rgb(c2_hex), (255, 255, 255))
        club_assigner = ClubAssigner(club1, club2)
        ball_assigner = BallToPlayerAssigner(club1, club2)

        # Standard Keypoints
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
            top_down_keypoints, field_img_path=str(BASE_DIR / 'assets' / 'field_2d_v2.png'),
            save_tracks_dir=str(OUTPUT_DIR)
        )

        # FORCE DISABLE PAUSE
        processor.pass_cfg["pause_frames"] = 0

        # Processing Loop
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Try codecs
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))

        batch = []
        batch_size = 10
        frame_count = 0
        threat_history = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(frame)

            if len(batch) == batch_size:
                processed = processor.process(batch, fps=fps)
                for p in processed:
                    out.write(p)
                    threat_history.append(
                        {"time": frame_count/fps, "value": processor.current_threat * 100})
                    frame_count += 1

                # Update Progress for Frontend
                jobs[job_id]["progress"] = int(
                    (frame_count / total_frames) * 100)
                batch = []

        if batch:
            processed = processor.process(batch, fps=fps)
            for p in processed:
                out.write(p)
                threat_history.append(
                    {"time": frame_count/fps, "value": processor.current_threat * 100})
                frame_count += 1

        cap.release()
        out.release()

        # Post-Processing
        processor.generate_heatmaps()

        # Save Results
        jobs[job_id]["result"] = {
            "video_url": f"http://localhost:8000/outputs/{output_name}",
            "threat_data": threat_history,
            "passes": processor._predictions_out
        }
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        print(f"[{job_id}] Complete.")

    except Exception as e:
        print(f"Error: {e}")
        jobs[job_id]["status"] = "failed"

# --- ENDPOINTS ---


@app.post("/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks, c1: str = "#E8F7F8", c2: str = "#ACFB91"):
    job_id = str(uuid.uuid4())
    path = str(UPLOAD_DIR / f"{job_id}_{file.filename}")
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {"status": "processing", "progress": 0}
    background_tasks.add_task(run_analysis_task, job_id, path, c1, c2)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]


@app.get("/heatmaps")
def get_heatmaps():
    files = [f for f in os.listdir(str(HEATMAP_DIR)) if f.endswith('.png')]
    return [{"id": f.split('_')[2].split('.')[0], "url": f"http://localhost:8000/outputs/heatmaps/{f}"} for f in files]
