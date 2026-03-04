"""
CHOICE 3 — Coloring the voxel model with occlusion reasoning (depth-aware)

1) Project all active voxels -> pixel coordinates (u,v)
2) Compute each voxel's depth in that camera: Zc from camera coordinates Xc = R*X + t
3) For each pixel, keep ONLY the voxel with the smallest depth (closest to camera).
   That voxel is the visible surface point at that pixel for that camera.
4) Sample the camera frame color at (u,v) and accumulate into that voxel's color.
5) Combine colors across cameras (average over cameras where voxel is visible).
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List


# ------------------------- Camera IO -------------------------

@dataclass
class CamParams:
    K: np.ndarray        # (3,3)
    d: np.ndarray        # (k,1) or (1,k)
    rvec: np.ndarray     # (3,1)
    tvec: np.ndarray     # (3,1)
    R: np.ndarray        # (3,3)


def load_camera_parameters(xml_path: str) -> CamParams:
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    d = fs.getNode("distortion_coefficients").mat()
    R = fs.getNode("rotation_matrix").mat()
    t = fs.getNode("translation_vector").mat()
    fs.release()

    # If your t is in mm (common in calibration), convert to meters:
    # If you already calibrated in meters, remove this.
    t = t / 1000.0

    rvec, _ = cv2.Rodrigues(R)
    return CamParams(K=K, d=d, rvec=rvec, tvec=t, R=R)


# ------------------------- Voxel grid / lookup -------------------------

def create_voxel_grid(
    x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, z_min=0.0, z_max=2.0, step=0.03
) -> np.ndarray:
    x_range = np.arange(x_min, x_max, step)
    y_range = np.arange(y_min, y_max, step)
    z_range = np.arange(z_min, z_max, step)
    voxels = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3).astype(np.float32)
    return voxels


def build_lookup_table(
    voxels: np.ndarray, cam: CamParams, image_shape: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """
    Returns:
      pixels: (N,2) projected integer pixel coords
      valid:  (N,)  boolean in-image and in-front-of-camera
      depth:  (N,)  float depth Zc (camera coords)
    """
    # Project to image
    imgpts, _ = cv2.projectPoints(voxels, cam.rvec, cam.tvec, cam.K, cam.d)
    imgpts = imgpts.reshape(-1, 2)

    h, w = image_shape

    # Depth from camera coordinates Xc = R*X + t
    # (N,3) = (N,3) @ (3,3).T + t.T
    Xc = voxels @ cam.R.T + cam.tvec.reshape(1, 3)
    depth = Xc[:, 2]  # Zc

    in_front = depth > 1e-6
    in_image = (
        (imgpts[:, 0] >= 0) & (imgpts[:, 0] < w) &
        (imgpts[:, 1] >= 0) & (imgpts[:, 1] < h)
    )
    valid = in_front & in_image

    pixels = imgpts.astype(np.int32)
    return {"pixels": pixels, "valid": valid, "depth": depth.astype(np.float32)}


def build_all_lookup_tables(voxels: np.ndarray) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, CamParams], Tuple[int, int]]:
    """
    Builds per-camera lookup tables and loads CamParams.
    Returns lookup_tables, cams, image_shape
    """
    lookup_tables: Dict[int, Dict[str, np.ndarray]] = {}
    cams: Dict[int, CamParams] = {}

    image_shape = None

    for cam_id in range(1, 5):
        xml_path = f"data/cam{cam_id}/config.xml"
        cam = load_camera_parameters(xml_path)
        cams[cam_id] = cam

        cap = cv2.VideoCapture(f"data/cam{cam_id}/video.avi")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read data/cam{cam_id}/video.avi")

        image_shape = frame.shape[:2]  # (h,w)

        lookup_tables[cam_id] = build_lookup_table(voxels, cam, image_shape)
        print(f"Lookup+depth built for cam{cam_id}")

    assert image_shape is not None
    return lookup_tables, cams, image_shape


# ------------------------- Reconstruction (from your earlier code) -------------------------

def load_masks_for_frame(frame_name: str) -> Dict[int, np.ndarray]:
    masks = {}
    for cam_id in range(1, 5):
        path = f"data/cam{cam_id}/foreground_masks/{frame_name}"
        mask = cv2.imread(path, 0)
        if mask is None:
            raise RuntimeError(f"Missing mask {path}")
        masks[cam_id] = mask
    return masks


def reconstruct_voxel_indices(foreground_masks: Dict[int, np.ndarray],
                              lookup_tables: Dict[int, Dict[str, np.ndarray]],
                              voxels: np.ndarray) -> np.ndarray:
    """
    Returns indices of voxels that survive all silhouettes.
    """
    voxel_on = np.ones(len(voxels), dtype=bool)

    for cam_id in range(1, 5):
        pixels = lookup_tables[cam_id]["pixels"]
        valid = lookup_tables[cam_id]["valid"]
        mask = foreground_masks[cam_id]

        cam_visible = np.zeros(len(voxels), dtype=bool)

        valid_idx = np.where(valid)[0]
        px = pixels[valid_idx]  # (M,2)

        cam_visible[valid_idx] = mask[px[:, 1], px[:, 0]] > 0
        voxel_on &= cam_visible

    return np.where(voxel_on)[0]


# ------------------------- Coloring with occlusion reasoning -------------------------

def get_frame_bgr(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def visible_surface_voxels_for_camera(active_idx: np.ndarray,
                                      lookup: Dict[str, np.ndarray],
                                      image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Occlusion reasoning:
    Among ACTIVE voxels, for each pixel, only the closest voxel (min depth) is visible.

    Returns:
      visible_idx: subset of active_idx that are visible from this camera
    """
    h, w = image_shape

    pixels = lookup["pixels"][active_idx]     # (A,2)
    valid = lookup["valid"][active_idx]       # (A,)
    depth = lookup["depth"][active_idx]       # (A,)

    # keep only valid
    keep = np.where(valid)[0]
    if keep.size == 0:
        return np.array([], dtype=np.int64)

    px = pixels[keep]
    d = depth[keep]
    idx = active_idx[keep]

    # pixel id for grouping
    pid = px[:, 1].astype(np.int64) * w + px[:, 0].astype(np.int64)

    # Sort by (pid, depth) so the first occurrence of each pid is the closest voxel
    order = np.lexsort((d, pid))  # primary pid, secondary depth
    pid_sorted = pid[order]
    idx_sorted = idx[order]

    # Take first in each pid group
    first = np.ones_like(pid_sorted, dtype=bool)
    first[1:] = pid_sorted[1:] != pid_sorted[:-1]
    visible_idx = idx_sorted[first]
    return visible_idx


def colorize_voxels_depth_aware(active_idx: np.ndarray,
                               voxels: np.ndarray,
                               lookup_tables: Dict[int, Dict[str, np.ndarray]],
                               image_shape: Tuple[int, int],
                               frames_bgr: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      colored_voxels: (M,3) voxel XYZ in world coords
      colors_rgb:     (M,3) uint8 colors in RGB (averaged over cameras where visible)
    """
    # Accumulators per voxel index (in full voxel grid)
    color_sum = np.zeros((len(voxels), 3), dtype=np.float32)
    color_cnt = np.zeros((len(voxels), 1), dtype=np.float32)

    for cam_id in range(1, 5):
        frame = frames_bgr[cam_id]
        lookup = lookup_tables[cam_id]

        # determine which active voxels are actually visible (not occluded) from this camera
        vis_idx = visible_surface_voxels_for_camera(active_idx, lookup, image_shape)
        if vis_idx.size == 0:
            continue

        px = lookup["pixels"][vis_idx]  # (V,2)
        # sample BGR then convert to RGB
        bgr = frame[px[:, 1], px[:, 0], :].astype(np.float32)
        rgb = bgr[:, ::-1]  # BGR->RGB

        color_sum[vis_idx] += rgb
        color_cnt[vis_idx] += 1.0

    # Keep voxels that got at least one visible color
    has_color = (color_cnt[:, 0] > 0)
    final_idx = np.where(has_color & np.isin(np.arange(len(voxels)), active_idx))[0]

    if final_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    avg_rgb = (color_sum[final_idx] / color_cnt[final_idx]).clip(0, 255).astype(np.uint8)
    return voxels[final_idx], avg_rgb


# ------------------------- Engine mapping (example) -------------------------

def world_to_engine_colored(voxels_world: np.ndarray, colors_rgb: np.ndarray) -> List[List[int]]:
    """
    Example output format (adjust to your renderer):
      [[vx, vy, vz, r, g, b], ...]
    """
    out = []
    for (x, y, z), (r, g, b) in zip(voxels_world, colors_rgb):
        vx = int((x + 1) * 64)
        vz = int((y + 1) * 64)
        vy = int(z * 32)
        if 0 <= vx < 128 and 0 <= vy < 64 and 0 <= vz < 128:
            out.append([vx, vy, vz, int(r), int(g), int(b)])
    return out


# ------------------------- Example main -------------------------

if __name__ == "__main__":
    # 1) Build voxel grid + lookup tables (with depth)
    voxels = create_voxel_grid(step=0.03)
    lookup_tables, cams, image_shape = build_all_lookup_tables(voxels)

    # 2) Iterate frames (based on mask files like your earlier pipeline)
    mask_folder = "data/cam1/foreground_masks"
    frame_files = sorted([f for f in os.listdir(mask_folder) if f.startswith("frame_") and f.endswith(".png")])

    # Choose a few frames for testing
    for frame_name in frame_files[:10]:
        frame_idx = int(frame_name.replace("frame_", "").replace(".png", ""))

        # Load silhouettes
        masks = load_masks_for_frame(frame_name)

        # Reconstruct active voxels (indices)
        active_idx = reconstruct_voxel_indices(masks, lookup_tables, voxels)
        print(frame_name, "active voxels:", active_idx.size)

        # Load actual camera frames for coloring
        frames_bgr = {}
        for cam_id in range(1, 5):
            frames_bgr[cam_id] = get_frame_bgr(f"data/cam{cam_id}/video.avi", frame_idx)

        # Color with occlusion reasoning
        colored_vox_world, colors_rgb = colorize_voxels_depth_aware(
            active_idx=active_idx,
            voxels=voxels,
            lookup_tables=lookup_tables,
            image_shape=image_shape,
            frames_bgr=frames_bgr
        )

        print("  colored voxels:", len(colored_vox_world))

        # Convert to engine coords + attach colors (example)
        engine_voxels_colored = world_to_engine_colored(colored_vox_world, colors_rgb)
        print("  engine colored voxels:", len(engine_voxels_colored))

        # Create blank image
        vis = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

        # Project colored voxels to cam1 for visualization
        lookup = lookup_tables[1]
        px = lookup["pixels"][active_idx]

        for i, (u, v) in enumerate(px):
            if 0 <= u < image_shape[1] and 0 <= v < image_shape[0]:
                vis[v, u] = colors_rgb[i]

        cv2.imwrite(f"data/choice3_debug/colored_debug_{frame_idx:04d}.png", vis)
