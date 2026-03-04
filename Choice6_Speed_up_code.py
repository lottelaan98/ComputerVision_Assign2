"""
CHOICE 6 — Speed-ups via parallelization + a faster indexing trick

This file gives you drop-in utilities to speed up 3 places:
1) Lookup-table creation: build per-camera projection tables in parallel (ProcessPoolExecutor)
2) Background subtraction: process cam1..cam4 in parallel (ProcessPoolExecutor)
3) Voxel reconstruction (per frame): faster mask lookup using 1D flattened indexing

"""

import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple, List


# ============================================================
#  Common: camera config load
# ============================================================

def load_camera_parameters(xml_path: str):
    """Returns K, dist, rvec, tvec, R."""
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    d = fs.getNode("distortion_coefficients").mat()
    R = fs.getNode("rotation_matrix").mat()
    t = fs.getNode("translation_vector").mat()
    fs.release()

    t = t / 1000.0

    rvec, _ = cv2.Rodrigues(R)
    return K, d, rvec, t, R


# ============================================================
#  Parallel lookup-table building 
# ============================================================

def _build_lookup_for_cam(cam_id: int, voxels: np.ndarray) -> Tuple[int, Dict[str, np.ndarray], Tuple[int, int]]:
    """
    Worker: build lookup table for one camera.
    Returns (cam_id, lookup_dict, image_shape)
    """
    xml_path = f"data/cam{cam_id}/config.xml"
    K, d, rvec, tvec, R = load_camera_parameters(xml_path)

    # Read one frame for image size
    cap = cv2.VideoCapture(f"data/cam{cam_id}/video.avi")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read data/cam{cam_id}/video.avi")
    h, w = frame.shape[:2]

    # Project all voxels
    imgpts, _ = cv2.projectPoints(voxels, rvec, tvec, K, d)
    imgpts = imgpts.reshape(-1, 2)

    # Depth 
    Xc = voxels @ R.T + tvec.reshape(1, 3)
    depth = Xc[:, 2].astype(np.float32)

    pixels = imgpts.astype(np.int32)

    in_front = depth > 1e-6
    in_image = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )
    valid = in_front & in_image

    # speed for reconstruction:
    # Precompute a 1D linear pixel index pid = v*w + u so we can do mask_flat[pid]
    pid = (pixels[:, 1].astype(np.int64) * w + pixels[:, 0].astype(np.int64))

    lookup = {
        "pixels": pixels,     # (N,2)
        "valid": valid,       # (N,)
        "pid": pid,           # (N,) 
        "depth": depth,       # (N,)
        "w": np.int32(w),
        "h": np.int32(h),
    }
    return cam_id, lookup, (h, w)


def parallel_build_all_lookup_tables(voxels: np.ndarray, cam_ids=(1, 2, 3, 4), max_workers: int = 4):
    """
    Builds lookup tables for all cams in parallel.
    Returns: lookup_tables dict and image_shape (h,w) 
    """
    lookup_tables: Dict[int, Dict[str, np.ndarray]] = {}
    image_shape = None

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_build_lookup_for_cam, cam_id, voxels) for cam_id in cam_ids]
        for f in as_completed(futures):
            cam_id, lookup, shape = f.result()
            lookup_tables[cam_id] = lookup
            image_shape = shape if image_shape is None else image_shape
            print(f"[lookup] built cam{cam_id}")

    return lookup_tables, image_shape


# ============================================================
#  Parallel background subtraction
# ============================================================

def create_background_average_hsv(video_path: str, num_frames: int = 50) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open background video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Background video has 0 frames?")

    idxs = [int(i * (total - 1) / max(1, (num_frames - 1))) for i in range(num_frames)]

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
        raise RuntimeError("Could not read any frames for background model.")

    return (sum_img / used).astype(np.uint8)


def background_subtraction_hsv(frame_bgr: np.ndarray, bg_hsv: np.ndarray,
                              H_THRESH: int, S_THRESH: int, V_THRESH: int) -> np.ndarray:
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    diff = cv2.absdiff(frame_hsv, bg_hsv)
    h_diff, s_diff, v_diff = cv2.split(diff)

    fg = ((h_diff > H_THRESH) | (s_diff > S_THRESH) | (v_diff > V_THRESH)).astype(np.uint8) * 255

    # light cleanup (same as your earlier pipeline)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    return fg


def _process_cam_background_and_masks(cam_id: int,
                                     num_bg_frames: int,
                                     H_THRESH: int, S_THRESH: int, V_THRESH: int,
                                     out_folder_name: str = "foreground_masks_fast") -> int:
    """
    Worker: build background model and produce masks for cam_id.
    """
    cam_dir = f"data/cam{cam_id}"
    bg_video = os.path.join(cam_dir, "background.avi")
    in_video = os.path.join(cam_dir, "video.avi")
    out_dir = os.path.join(cam_dir, out_folder_name)
    os.makedirs(out_dir, exist_ok=True)

    bg_hsv = create_background_average_hsv(bg_video, num_bg_frames)
    cv2.imwrite(os.path.join(out_dir, "background_model.png"), bg_hsv)

    cap = cv2.VideoCapture(in_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open {in_video}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg = background_subtraction_hsv(frame, bg_hsv, H_THRESH, S_THRESH, V_THRESH)
        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:04d}.png"), fg)
        frame_idx += 1

    cap.release()
    print(f"[bg] cam{cam_id}: wrote {frame_idx} masks to {out_dir}")
    return cam_id


def parallel_background_subtraction_all_cams(num_bg_frames=50,
                                            H_THRESH=15, S_THRESH=30, V_THRESH=30,
                                            cam_ids=(1, 2, 3, 4),
                                            max_workers=4,
                                            out_folder_name="foreground_masks_fast"):
    """
    Runs background model + mask extraction for all cams in parallel.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_process_cam_background_and_masks,
                      cam_id, num_bg_frames, H_THRESH, S_THRESH, V_THRESH, out_folder_name)
            for cam_id in cam_ids
        ]
        for f in as_completed(futures):
            _ = f.result()
    print("[bg] All cameras done.")


# ============================================================
#  Faster voxel reconstruction per frame, 1D mask indexing
# ============================================================

def load_masks_for_frame(frame_name: str, folder_name="foreground_masks_fast") -> Dict[int, np.ndarray]:
    masks = {}
    for cam_id in range(1, 5):
        path = f"data/cam{cam_id}/{folder_name}/{frame_name}"
        mask = cv2.imread(path, 0)
        if mask is None:
            raise RuntimeError(f"Missing mask: {path}")
        masks[cam_id] = mask
    return masks


def reconstruct_voxels_fast(masks: Dict[int, np.ndarray],
                            lookup_tables: Dict[int, Dict[str, np.ndarray]],
                            voxels: np.ndarray) -> np.ndarray:
    """
    Same logic as your reconstruct_voxels, but faster:
    - uses mask_flat[pid] instead of mask[v,u] indexing
    - avoids repeated np.where(valid) overhead by using boolean masks directly
    """
    voxel_on = np.ones(len(voxels), dtype=bool)

    for cam_id in range(1, 5):
        lookup = lookup_tables[cam_id]
        valid = lookup["valid"]
        pid = lookup["pid"]
        w = int(lookup["w"])
        h = int(lookup["h"])

        mask = masks[cam_id]
        if mask.shape[0] != h or mask.shape[1] != w:
            raise RuntimeError(f"Mask shape mismatch cam{cam_id}: {mask.shape} vs expected {(h,w)}")

        mask_flat = mask.reshape(-1)
        cam_visible = np.zeros(len(voxels), dtype=bool)

        # Only check valid projections:
        cam_visible[valid] = (mask_flat[pid[valid]] > 0)

        voxel_on &= cam_visible
        if not voxel_on.any():
            break

    return voxels[voxel_on]


# ============================================================
#  Example main
# ============================================================

def create_voxel_grid():
    # match your earlier ranges (edit as needed)
    x_range = np.arange(-1.0, 1.0, 0.03)
    y_range = np.arange(-1.0, 1.0, 0.03)
    z_range = np.arange(0.0, 2.0, 0.03)
    voxels = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3).astype(np.float32)
    return voxels


if __name__ == "__main__":
    # parallel background subtraction (per camera)
    # Comment out if you already produced masks.
    parallel_background_subtraction_all_cams(
        num_bg_frames=50,
        H_THRESH=15, S_THRESH=30, V_THRESH=30,
        out_folder_name="foreground_masks_fast",
        max_workers=4
    )

    # Build voxel grid + parallel lookup tables (per camera)
    voxels = create_voxel_grid()
    lookup_tables, image_shape = parallel_build_all_lookup_tables(voxels, max_workers=4)
    print("[lookup] all cameras done. image_shape=", image_shape)

    # Test fast reconstruction for a few frames
    mask_folder = "data/cam1/foreground_masks_fast"
    frame_files = sorted([f for f in os.listdir(mask_folder) if f.startswith("frame_") and f.endswith(".png")])

    for frame_name in frame_files[:5]:
        masks = load_masks_for_frame(frame_name, folder_name="foreground_masks_fast")
        active = reconstruct_voxels_fast(masks, lookup_tables, voxels)
        print(frame_name, "active voxels:", len(active))