import cv2
import numpy as np
import os

def load_camera_parameters(xml_path):
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)

    K = fs.getNode("camera_matrix").mat()
    d = fs.getNode("distortion_coefficients").mat()
    R = fs.getNode("rotation_matrix").mat()
    t = fs.getNode("translation_vector").mat()
    t = t / 1000.0

    fs.release()
    rvec, _ = cv2.Rodrigues(R)
    return K, d, rvec, t

    # # Convert mm → meters
    # t = t / 1000.0

    # # Invert extrinsics
    # R_inv = R.T
    # t_inv = -R_inv @ t

    # rvec, _ = cv2.Rodrigues(R_inv)

    # return K, d, rvec, t_inv

def create_voxel_grid(x_range=(-3.5,3.5), y_range=(0,2), z_range=(-3.5,3.5), step=0.03):
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    z = np.arange(z_range[0], z_range[1], step)

    voxels = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
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
        path = f"data/cam{cam_id}/foreground_masks_auto/{frame_name}"
        mask = cv2.imread(path, 0)
        #print(mask.shape)
        

        if mask is None:
            raise Exception(f"Missing {frame_name} in cam{cam_id}")

        masks[cam_id] = mask

    return masks

def reconstruct_voxels(foreground_masks, lookup_tables, voxels):
    """
    foreground_masks: dict {1: mask1, 2: mask2, ...}
    """

    voxel_on = np.ones(len(voxels), dtype=bool)
    
    for cam_id in range(1, 5):
        pixels = lookup_tables[cam_id]["pixels"]
        valid = lookup_tables[cam_id]["valid"]
        mask = foreground_masks[cam_id]
        #print(np.unique(mask))

        #print(pixels.shape, valid.shape, mask.shape)
        #print(f"Cam{cam_id} valid projections:", np.sum(valid))
        

        cam_visible = np.zeros(len(voxels), dtype=bool)

        #print(f"Cam{cam_id} foreground hits:", np.sum(cam_visible))

        valid_indices = np.where(valid)[0]
        px = pixels[valid_indices]

        #cam_visible[valid_indices] = (mask[px[:, 1], px[:, 0]] == 255)
        cam_visible[valid_indices] = mask[px[:, 1], px[:, 0]] > 0

        voxel_on &= cam_visible
        #voxel_on = cam_visible
        # break

    return voxels[voxel_on]

# def remove_floor_voxels(voxels, threshold=0.05):
#     return voxels[voxels[:, 2] > threshold]

# def world_to_engine(voxels):
#     engine_voxels = []

#     for x, y, z in voxels:

#         vx = int((x + 1) * 64)
#         vz = int((y + 1) * 64)
#         vy = int(z * 32)

#         if 0 <= vx < 128 and 0 <= vy < 64 and 0 <= vz < 128:
#             engine_voxels.append([vx, vy, vz])

#     return engine_voxels


def world_to_engine(voxels, x_range=(-1, 1), y_range=(0, 2), z_range=(-1, 1)):
    engine_voxels = []

    x_min, x_max = x_range
    y_min, y_max = y_range
    z_min, z_max = z_range

    # Scale to voxel grid
    for x, y, z in voxels:
        vx = int((x - x_min) / (x_max - x_min) * 127)  # 0–127
        vy = int((y - y_min) / (y_max - y_min) * 63)   # 0–63
        vz = int((z - z_min) / (z_max - z_min) * 127)  # 0–127

        if 0 <= vx < 128 and 0 <= vy < 64 and 0 <= vz < 128:
            engine_voxels.append([vx, vy, vz])

    return engine_voxels

if __name__ == "__main__":

    voxels = create_voxel_grid()
    lookup_tables = build_all_lookup_tables(voxels)
    # for cam_id in range(1, 5):
    #     valid = lookup_tables[cam_id]["valid"]
    #     print(f"Cam{cam_id} valid projections:", np.sum(valid))

    
    mask_folder = "data/cam1/foreground_masks_auto"
    frame_files = sorted([f for f in os.listdir(mask_folder) if f.startswith("frame_") and f.endswith(".png")])

    for frame_name in frame_files:

        masks = load_masks_for_frame(frame_name)

        active_voxels = reconstruct_voxels(masks, lookup_tables, voxels)

        # Cleanup
        # active_voxels = remove_floor_voxels(active_voxels)

        print(frame_name, "active voxels:", len(active_voxels))
        #engine_voxels = world_to_engine(active_voxels)
