import cv2
import numpy as np
import os

def load_camera_parameters(xml_path):
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)

    K = fs.getNode("camera_matrix").mat()
    d = fs.getNode("distortion_coefficients").mat()
    r = fs.getNode("rotation_matrix").mat()
    t = fs.getNode("translation_vector").mat()
    fs.release()
    return K, d, r, t

def create_voxel_grid():
    x_range = np.arange(-1.0, 1.0, 0.03)
    y_range = np.arange(-1.0, 1.0, 0.03)
    z_range = np.arange(0.0, 2.0, 0.03)

    voxels = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
    return voxels.astype(np.float32)

def build_lookup_table(voxels, K, d, rvec, tvec, image_shape):
    imgpts, _ = cv2.projectPoints(
        voxels,
        rvec,
        tvec,
        K,
        d
    )

    imgpts = imgpts.reshape(-1, 2)

    h, w = image_shape

    # Check if in image
    valid_mask = (
        (imgpts[:, 0] >= 0) & (imgpts[:, 0] < w) &
        (imgpts[:, 1] >= 0) & (imgpts[:, 1] < h)
    )

    projected_pixels = imgpts.astype(np.int32)

    return projected_pixels, valid_mask


def build_all_lookup_tables(voxels):
    lookup_tables = {}

    for cam_id in range(1, 5):

        xml_path = f"data/cam{cam_id}/config.xml"
        K, d, rvec, tvec = load_camera_parameters(xml_path)

        # Read one frame to get image size
        cap = cv2.VideoCapture(f"data/cam{cam_id}/video.avi")
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise Exception(f"Could not read video for cam{cam_id}")

        image_shape = frame.shape[:2]

        projected_pixels, valid_mask = build_lookup_table(
            voxels, K, d, rvec, tvec, image_shape
        )

        lookup_tables[cam_id] = {
            "pixels": projected_pixels,
            "valid": valid_mask
        }

        print(f"Lookup table built for cam{cam_id}")

    return lookup_tables

def reconstruct_voxels(foreground_masks, lookup_tables, voxels):
    """
    foreground_masks: dict {1: mask1, 2: mask2, ...}
    """

    voxel_on = np.ones(len(voxels), dtype=bool)

    for cam_id in range(1, 5):
        pixels = lookup_tables[cam_id]["pixels"]
        valid = lookup_tables[cam_id]["valid"]
        mask = foreground_masks[cam_id]

        cam_visible = np.zeros(len(voxels), dtype=bool)

        valid_indices = np.where(valid)[0]
        px = pixels[valid_indices]

        cam_visible[valid_indices] = (
            mask[px[:, 1], px[:, 0]] == 255
        )

        voxel_on &= cam_visible

    return voxels[voxel_on]

def load_masks_for_frame(frame_name):
    masks = {}

    for cam_id in range(1, 5):
        path = f"data/cam{cam_id}/foreground_masks/{frame_name}"
        mask = cv2.imread(path, 0)

        if mask is None:
            raise Exception(f"Missing {frame_name} in cam{cam_id}")

        masks[cam_id] = mask

    return masks

if __name__ == "__main__":

    voxels = create_voxel_grid()
    lookup_tables = build_all_lookup_tables(voxels)

    cam1_files = sorted([
        f for f in os.listdir("data/cam1/foreground_masks")
        if f.startswith("frame_") and f.endswith(".png")
    ])

    for frame_name in cam1_files:

        masks = load_masks_for_frame(frame_name)
        active_voxels = reconstruct_voxels(voxels, lookup_tables, masks)

        print(frame_name, "-> active voxels:", len(active_voxels))
