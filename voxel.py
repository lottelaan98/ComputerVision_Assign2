import cv2
import numpy as np
import os

def load_camera_parameters(xml_path):
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    d = fs.getNode("distortion_coefficients").mat()
    R = fs.getNode("rotation_matrix").mat()
    t = fs.getNode("translation_vector").mat()
    t = t/1000
    fs.release()

    rvec, _ = cv2.Rodrigues(R)
    return K, d, rvec, t


def create_voxel_grid(x_range=(-3.5, 3.5), y_range=(0, 2), z_range=(-3.5, 3.5), step=0.03):
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    z = np.arange(z_range[0], z_range[1], step)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    voxels = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
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

def load_masks_for_frame(frame_name):
    masks = {}
    for cam_id in range(1, 5):
        #can change the folder name if needed for different versions of masks
        path = f"data/cam{cam_id}/foreground_masks/{frame_name}"
        mask = cv2.imread(path, 0)
        #print(mask.shape)
        

        if mask is None:
            raise Exception(f"Missing {frame_name} in cam{cam_id}")

        masks[cam_id] = mask

    return masks

def reconstruct_voxels(foreground_masks, lookup_tables, voxels):

    votes = np.zeros(len(voxels), dtype=int)

    for cam_id in range(1, 5):

        pixels = lookup_tables[cam_id]["pixels"]
        valid = lookup_tables[cam_id]["valid"]
        mask = foreground_masks[cam_id]

        valid_indices = np.where(valid)[0]
        px = pixels[valid_indices]

        visible = mask[px[:, 1], px[:, 0]] > 0

        votes[valid_indices] += visible.astype(int)

    voxel_on = votes >= 3  

    return voxels[voxel_on]

# def remove_floor_voxels(voxels, threshold=0.05):
#     return voxels[voxels[:, 2] > threshold]

def world_to_engine(voxels, x_range=(-3.5,3.5), y_range=(0,2), z_range=(-3.5,3.5)):
    engine_voxels = []
    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    for x, y, z in voxels:
        vx = int((x - x_min) / (x_max - x_min) * 127)
        vy = int((y - y_min) / (y_max - y_min) * 63)
        vz = int((z - z_min) / (z_max - z_min) * 127)
        if 0 <= vx < 128 and 0 <= vy < 64 and 0 <= vz < 128:
            engine_voxels.append([vx, vy, vz])
    return engine_voxels

if __name__ == "__main__":

    voxels = create_voxel_grid()
    lookup_tables = build_all_lookup_tables(voxels)
    for cam_id in range(1, 5):
        valid = lookup_tables[cam_id]["valid"]
        print(f"Cam{cam_id} valid projections:", np.sum(valid))
    
    #can change the folder name if needed for different versions of masks
    mask_folder = "data/cam1/foreground_masks"
    frame_files = sorted([f for f in os.listdir(mask_folder) if f.startswith("frame_") and f.endswith(".png")])

    for frame_name in frame_files:

        masks = load_masks_for_frame(frame_name)

        active_voxels = reconstruct_voxels(masks, lookup_tables, voxels)

        # Cleanup
        # active_voxels = remove_floor_voxels(active_voxels)

        print(frame_name, "active voxels:", len(active_voxels))
        #engine_voxels = world_to_engine(active_voxels)


