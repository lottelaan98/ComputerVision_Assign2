"""
CHOICE 2 — Automatic HSV threshold selection (no built-in background subtractors)

Idea:
- Build a background model from background.avi (HSV average).
- For a set of sampled frames from video.avi, try many (H,S,V) threshold triples.
- For each triple, compute foreground masks with simple HSV absdiff + thresholding (your method).
- Score each triple using unsupervised quality heuristics:
    1) Foreground area should be reasonable (not near 0%, not near 100%).
    2) Mask should have little "salt-and-pepper" noise (few tiny connected components).
    3) Mask should be temporally stable (good IoU between consecutive frames).
- Choose thresholds with the best (lowest) score.

This produces per-camera thresholds without manual supervision.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict


# ------------------------- Config -------------------------

NUM_BG_FRAMES = 50

# Sample frames used to evaluate thresholds (keep small for speed)
NUM_EVAL_FRAMES = 30

# Grid-search ranges (coarse by default; you can refine later)
H_CANDIDATES = list(range(6, 31, 3))    # Hue threshold values to try
S_CANDIDATES = list(range(10, 71, 5))   # Saturation thresholds
V_CANDIDATES = list(range(10, 71, 5))   # Value thresholds

# Morphology for cleanup during scoring (helps compare "usable" masks)
OPEN_KERNEL = np.ones((3, 3), np.uint8)
DILATE_KERNEL = np.ones((3, 3), np.uint8)

# Foreground area expectations (unsupervised prior)
# Adjust if your subject appears small/large in image.
FG_AREA_MIN = 0.005   # 0.5% of pixels
FG_AREA_MAX = 0.35    # 35% of pixels

# Connected components noise penalty: components smaller than this are "specks"
SMALL_COMPONENT_AREA = 80

# Weighting of score terms (tune if needed)
W_AREA = 1.0
W_NOISE = 1.2
W_STABILITY = 1.0


# ------------------------- Helpers -------------------------

@dataclass
class Thresholds:
    h: int
    s: int
    v: int


def evenly_spaced_indices(total: int, count: int) -> List[int]:
    if total <= 0:
        return []
    if count <= 1:
        return [total // 2]
    return [int(i * (total - 1) / (count - 1)) for i in range(count)]


def create_background_average_hsv(video_path: str, num_frames: int = 50) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open background video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = evenly_spaced_indices(total_frames, num_frames)

    sum_img = None
    used = 0

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        sum_img = hsv if sum_img is None else (sum_img + hsv)
        used += 1

    cap.release()
    if used == 0:
        raise RuntimeError("No frames read to build background model.")

    bg = (sum_img / used).astype(np.uint8)
    return bg


def sample_video_frames_bgr(video_path: str, num_frames: int = 30) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open input video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = evenly_spaced_indices(total_frames, num_frames)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    if len(frames) < max(5, num_frames // 5):
        raise RuntimeError(f"Too few frames sampled from {video_path} ({len(frames)}).")

    return frames


def foreground_mask_hsv(frame_hsv: np.ndarray, bg_hsv: np.ndarray, thr: Thresholds) -> np.ndarray:
    diff = cv2.absdiff(frame_hsv, bg_hsv)
    h_diff, s_diff, v_diff = cv2.split(diff)

    h_mask = (h_diff > thr.h).astype(np.uint8)
    s_mask = (s_diff > thr.s).astype(np.uint8)
    v_mask = (v_diff > thr.v).astype(np.uint8)

    fg = ((h_mask + s_mask + v_mask) > 0).astype(np.uint8) * 255
    # Cleanup (same style as your pipeline)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, OPEN_KERNEL)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, DILATE_KERNEL)
    return fg


def count_small_components(mask: np.ndarray, small_area: int = 80) -> Tuple[int, int]:
    """
    Returns:
      - num_components (excluding background)
      - num_small_components (area < small_area)
    """
    # Ensure binary 0/255
    bin_mask = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    # label 0 is background
    num_components = max(0, num_labels - 1)

    if num_components == 0:
        return 0, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    num_small = int(np.sum(areas < small_area))
    return num_components, num_small


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def score_thresholds(frames_bgr: List[np.ndarray], bg_hsv: np.ndarray, thr: Thresholds) -> float:
    """
    Lower is better.
    """
    prev_mask = None

    area_penalties = []
    noise_penalties = []
    stability_penalties = []

    H, W = frames_bgr[0].shape[:2]
    total_px = H * W

    for frame in frames_bgr:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = foreground_mask_hsv(frame_hsv, bg_hsv, thr)

        fg_area = float((mask > 0).sum()) / float(total_px)

        # 1) Area penalty: encourage within [FG_AREA_MIN, FG_AREA_MAX]
        if fg_area < FG_AREA_MIN:
            area_pen = (FG_AREA_MIN - fg_area) / FG_AREA_MIN
        elif fg_area > FG_AREA_MAX:
            area_pen = (fg_area - FG_AREA_MAX) / (1.0 - FG_AREA_MAX)
        else:
            area_pen = 0.0
        area_penalties.append(area_pen)

        # 2) Noise penalty: many small blobs is bad
        num_cc, num_small = count_small_components(mask, SMALL_COMPONENT_AREA)
        # Normalize by image size: more pixels → allow a few blobs
        noise_pen = (num_small / max(1, num_cc + 1))
        noise_penalties.append(noise_pen)

        # 3) Temporal stability: consecutive IoU should not be tiny
        if prev_mask is not None:
            stability_penalties.append(1.0 - iou(prev_mask, mask))
        prev_mask = mask

    area_term = float(np.mean(area_penalties))
    noise_term = float(np.mean(noise_penalties))
    stab_term = float(np.mean(stability_penalties)) if stability_penalties else 0.0

    return W_AREA * area_term + W_NOISE * noise_term + W_STABILITY * stab_term


def find_best_thresholds(video_path: str, bg_hsv: np.ndarray) -> Thresholds:
    frames = sample_video_frames_bgr(video_path, NUM_EVAL_FRAMES)

    best_thr = None
    best_score = float("inf")

    # Coarse grid search
    for h in H_CANDIDATES:
        for s in S_CANDIDATES:
            for v in V_CANDIDATES:
                thr = Thresholds(h=h, s=s, v=v)
                sc = score_thresholds(frames, bg_hsv, thr)
                if sc < best_score:
                    best_score = sc
                    best_thr = thr

    assert best_thr is not None
    return best_thr


def save_thresholds(path: str, thr: Thresholds) -> None:
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write("H_THRESH", int(thr.h))
    fs.write("S_THRESH", int(thr.s))
    fs.write("V_THRESH", int(thr.v))
    fs.release()


def apply_and_save_masks(video_path: str, bg_hsv: np.ndarray, thr: Thresholds, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = foreground_mask_hsv(hsv, bg_hsv, thr)
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:04d}.png"), mask)
        idx += 1

        if idx % 50 == 0:
            print(f"  saved {idx} masks...")

    cap.release()


# ------------------------- Main (run per camera) -------------------------

if __name__ == "__main__":
    for cam_id in range(1, 5):
        cam_dir = f"data/cam{cam_id}"
        bg_video = os.path.join(cam_dir, "background.avi")
        in_video = os.path.join(cam_dir, "video.avi")
        out_dir = os.path.join(cam_dir, "foreground_masks_auto")
        thr_xml = os.path.join(cam_dir, "auto_thresholds.xml")

        print(f"\n=== Camera {cam_id} ===")
        print("Building background model...")
        bg_hsv = create_background_average_hsv(bg_video, NUM_BG_FRAMES)

        print("Searching for best thresholds (unsupervised)...")
        best = find_best_thresholds(in_video, bg_hsv)
        print(f"Best thresholds: H={best.h}, S={best.s}, V={best.v}")

        print("Saving thresholds...")
        save_thresholds(thr_xml, best)

        print("Generating masks with chosen thresholds...")
        apply_and_save_masks(in_video, bg_hsv, best, out_dir)

        print(f"Done cam{cam_id}. Thresholds saved to {thr_xml}, masks in {out_dir}")