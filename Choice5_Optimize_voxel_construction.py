import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

# ============================================================
# Camera + Voxel grid
# ============================================================

def load_camera_parameters(xml_path: str):
    """Load K, dist, rvec, tvec, and also rotation matrix R (for depth if needed later)."""
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    d = fs.getNode("distortion_coefficients").mat()
    R = fs.getNode("rotation_matrix").mat()
    t = fs.getNode("translation_vector").mat()
    fs.release()

    # If your t was saved in mm, convert to meters. If already meters, remove this.
    t = t / 1000.0

    rvec, _ = cv2.Rodrigues(R)
    return K, d, rvec, t, R


@dataclass
class GridSpec:
    x_min: float = -1.0
    x_max: float =  1.0
    y_min: float = -1.0
    y_max: float =  1.0
    z_min: float =  0.0
    z_max: float =  2.0
    step: float  =  0.03


def create_voxel_grid(spec: GridSpec) -> np.ndarray:
    x = np.arange(spec.x_min, spec.x_max, spec.step, dtype=np.float32)
    y = np.arange(spec.y_min, spec.y_max, spec.step, dtype=np.float32)
    z = np.arange(spec.z_min, spec.z_max, spec.step, dtype=np.float32)
    vox = np.array(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T
    return vox.astype(np.float32)


# ============================================================
# Inverse Look-up table (pixel -> list of voxels)
# (This is exactly the "iterate over pixels in LUT" approach.)
# ============================================================

@dataclass
class InverseLUT:
    """
    Compressed mapping from pixel id -> contiguous range in voxel_idx_sorted

    unique_pid: sorted unique pixel-ids that are valid
    offsets:    offsets into voxel_idx_sorted, length len(unique_pid)+1
    voxel_idx_sorted: all voxel indices, grouped by pid
    w,h: image dimensions
    """
    unique_pid: np.ndarray
    offsets: np.ndarray
    voxel_idx_sorted: np.ndarray
    w: int
    h: int


def build_inverse_lut_for_cam(voxels: np.ndarray, cam_id: int) -> InverseLUT:
    xml_path = f"data/cam{cam_id}/config.xml"
    K, d, rvec, tvec, _R = load_camera_parameters(xml_path)

    cap = cv2.VideoCapture(f"data/cam{cam_id}/video.avi")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read data/cam{cam_id}/video.avi")
    h, w = frame.shape[:2]

    # Project all voxels to this camera
    imgpts, _ = cv2.projectPoints(voxels, rvec, tvec, K, d)
    imgpts = imgpts.reshape(-1, 2)
    px = imgpts.astype(np.int32)

    # Valid in-image
    valid = (
        (px[:, 0] >= 0) & (px[:, 0] < w) &
        (px[:, 1] >= 0) & (px[:, 1] < h)
    )

    voxel_idx = np.nonzero(valid)[0].astype(np.int32)
    pid = (px[valid, 1].astype(np.int64) * w + px[valid, 0].astype(np.int64))

    # Sort by pid so all voxels for the same pixel are contiguous
    order = np.argsort(pid, kind="mergesort")
    pid_sorted = pid[order]
    voxel_idx_sorted = voxel_idx[order]

    # Unique pid + offsets (CSR-like)
    unique_pid, start_idx = np.unique(pid_sorted, return_index=True)
    offsets = np.empty(len(unique_pid) + 1, dtype=np.int64)
    offsets[:-1] = start_idx
    offsets[-1] = len(pid_sorted)

    return InverseLUT(
        unique_pid=unique_pid,
        offsets=offsets,
        voxel_idx_sorted=voxel_idx_sorted,
        w=w,
        h=h
    )


def build_all_inverse_luts(voxels: np.ndarray, cam_ids=(1, 2, 3, 4)) -> Dict[int, InverseLUT]:
    inv = {}
    for cam_id in cam_ids:
        inv[cam_id] = build_inverse_lut_for_cam(voxels, cam_id)
        print(f"[LUT] inverse LUT built for cam{cam_id} (unique pixels: {len(inv[cam_id].unique_pid)})")
    return inv


# ============================================================
# Incremental voxel reconstruction using XOR masks
# ============================================================

def load_mask(cam_id: int, frame_name: str, masks_folder="foreground_masks") -> np.ndarray:
    path = f"data/cam{cam_id}/{masks_folder}/{frame_name}"
    m = cv2.imread(path, 0)
    if m is None:
        raise RuntimeError(f"Missing mask: {path}")
    return m


def pid_ranges_for_pixels(inv: InverseLUT, changed_pid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a list of pixel-ids (changed_pid), find for each pid its [start,end) range
    in inv.voxel_idx_sorted using binary search on inv.unique_pid.

    Returns arrays start, end aligned with changed_pid after filtering only existing pid.
    """
    # Locate in unique_pid
    pos = np.searchsorted(inv.unique_pid, changed_pid)
    ok = (pos < len(inv.unique_pid)) & (inv.unique_pid[pos] == changed_pid)
    pos = pos[ok]
    start = inv.offsets[pos]
    end = inv.offsets[pos + 1]
    return ok, start, end


def incremental_reconstruction_sequence(
    frame_files: List[str],
    inv_luts: Dict[int, InverseLUT],
    num_voxels: int,
    masks_folder="foreground_masks",
    refresh_every: int = 50
) -> Dict[str, np.ndarray]:
    """
    Maintains per-camera visibility and a global count.
    Only updates voxels for pixels that changed (XOR) between frames. :contentReference[oaicite:3]{index=3}

    Returns: dict frame_name -> active voxel indices (count == 4)
    """
    cam_ids = sorted(inv_luts.keys())
    C = len(cam_ids)

    # Visible flags per cam for all voxels (boolean)
    visible = {cam_id: np.zeros(num_voxels, dtype=bool) for cam_id in cam_ids}
    # Count of how many cameras see each voxel as foreground (0..C)
    count = np.zeros(num_voxels, dtype=np.uint8)

    prev_masks = {cam_id: None for cam_id in cam_ids}
    results: Dict[str, np.ndarray] = {}

    for fi, frame_name in enumerate(frame_files):
        do_refresh = (fi == 0) or (refresh_every > 0 and fi % refresh_every == 0)

        for cam_id in cam_ids:
            inv = inv_luts[cam_id]
            mask = load_mask(cam_id, frame_name, masks_folder=masks_folder)
            if mask.shape[:2] != (inv.h, inv.w):
                raise RuntimeError(f"Mask size mismatch cam{cam_id}: {mask.shape} vs {(inv.h, inv.w)}")

            if do_refresh or prev_masks[cam_id] is None:
                # Full update for this camera: compute visibility for ALL pixels in LUT.
                # We still do it pixel-driven (fast), but it's a refresh.
                mask_flat = (mask.reshape(-1) > 0)

                # Reset this camera's visibility and update global counts accordingly:
                old_vis = visible[cam_id]
                if old_vis.any():
                    count[old_vis] -= 1
                visible[cam_id] = np.zeros(num_voxels, dtype=bool)

                # For each LUT pixel, if foreground -> mark all voxels mapped to it
                fg_pid = np.nonzero(mask_flat)[0].astype(np.int64)

                ok, start, end = pid_ranges_for_pixels(inv, fg_pid)
                fg_pid = fg_pid[ok]

                for s, e in zip(start, end):
                    vlist = inv.voxel_idx_sorted[s:e]
                    visible[cam_id][vlist] = True

                count[visible[cam_id]] += 1

            else:
                # Incremental update: only changed pixels (XOR) :contentReference[oaicite:4]{index=4}
                prev = prev_masks[cam_id]
                changed = cv2.bitwise_xor(prev, mask)
                ys, xs = np.nonzero(changed)
                if ys.size > 0:
                    changed_pid = (ys.astype(np.int64) * inv.w + xs.astype(np.int64))

                    ok, start, end = pid_ranges_for_pixels(inv, changed_pid)
                    changed_pid = changed_pid[ok]
                    start = start  # already filtered by ok inside pid_ranges
                    end = end

                    mask_flat = mask.reshape(-1)
                    for pid, s, e in zip(changed_pid, start, end):
                        new_fg = mask_flat[pid] > 0
                        vlist = inv.voxel_idx_sorted[s:e]

                        old_vals = visible[cam_id][vlist]
                        if new_fg:
                            # Only those previously false will increment count
                            to_add = vlist[~old_vals]
                            visible[cam_id][to_add] = True
                            count[to_add] += 1
                        else:
                            # Only those previously true will decrement count
                            to_sub = vlist[old_vals]
                            visible[cam_id][to_sub] = False
                            count[to_sub] -= 1

            prev_masks[cam_id] = mask

        active_idx = np.nonzero(count == C)[0].astype(np.int32)
        results[frame_name] = active_idx
        print(f"{frame_name}: active voxels = {len(active_idx)}  (refresh={do_refresh})")

    return results


# ============================================================
# Example run
# ============================================================

if __name__ == "__main__":
    spec = GridSpec(step=0.03)
    voxels = create_voxel_grid(spec)

    # 1) Build inverse LUT once (pixel -> voxels), per camera :contentReference[oaicite:5]{index=5}
    inv_luts = build_all_inverse_luts(voxels, cam_ids=(1, 2, 3, 4))

    # 2) Collect frame names from one camera’s masks folder (must exist for all cams)
    masks_folder = "foreground_masks"  # change if you used another folder name
    mask_dir = f"data/cam1/{masks_folder}"
    frame_files = sorted([f for f in os.listdir(mask_dir) if f.startswith("frame_") and f.endswith(".png")])

    # 3) Run incremental reconstruction
    results = incremental_reconstruction_sequence(
        frame_files=frame_files,
        inv_luts=inv_luts,
        num_voxels=len(voxels),
        masks_folder=masks_folder,
        refresh_every=50  # periodic full refresh to avoid drift
    )

    # Example: get active voxels for one frame
    some_frame = frame_files[0]
    active_idx = results[some_frame]
    active_voxels = voxels[active_idx]
    print("Example active_voxels shape:", active_voxels.shape)