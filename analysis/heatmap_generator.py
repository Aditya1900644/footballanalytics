# analysis/heatmap_generator.py
import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    def __init__(self, field_img_path: str, output_dir: str):
        """
        Initializes the generator with the base field image.
        """
        self.field_img = cv2.imread(field_img_path)
        if self.field_img is None:
            raise ValueError(f"Could not load field image at {field_img_path}")

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.field_h, self.field_w, _ = self.field_img.shape

    def generate_single(self, player_id: int, positions: list):
        """
        Generates and saves a heatmap for a single player ID based on their position history.
        """
        # Ignore players with too little data (less than 60 frames/2 seconds)
        if len(positions) < 60:
            return

        # 1. Prepare Data
        pos_arr = np.array(positions)
        x = pos_arr[:, 0]
        y = pos_arr[:, 1]

        # Clip coordinates to ensure they stay within field bounds
        x = np.clip(x, 0, self.field_w - 1)
        y = np.clip(y, 0, self.field_h - 1)

        # 2. Create 2D Histogram (The "Grid" of activity)
        # bins=[50, 50] determines resolution. Higher = finer detail.
        heatmap, xedges, yedges = np.histogram2d(
            x, y,
            bins=[50, 50],
            range=[[0, self.field_w], [0, self.field_h]]
        )

        # 3. Apply Gaussian Smoothing (The "Glow" effect)
        # sigma=2 controls the spread of the blur
        heatmap = gaussian_filter(heatmap, sigma=2)

        # 4. Normalize and Color
        heatmap = (heatmap / heatmap.max()) * 255
        heatmap = np.array(heatmap, dtype=np.uint8)

        # Resize histogram back to full field image size
        # Note: histogram2d is (x,y), cv2 resize is (width, height). We transpose (.T) to match.
        heatmap = cv2.resize(heatmap.T, (self.field_w, self.field_h))

        # Apply "JET" colormap (Blue=Low, Red=High)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 5. Blend with Field Image
        alpha = 0.5  # Transparency factor
        overlay = cv2.addWeighted(
            colored_heatmap, alpha, self.field_img, 1 - alpha, 0)

        # 6. Save
        save_path = os.path.join(
            self.output_dir, f"heatmap_player_{player_id}.png")
        cv2.imwrite(save_path, overlay)
        # print(f"Saved heatmap for Player {player_id}")
