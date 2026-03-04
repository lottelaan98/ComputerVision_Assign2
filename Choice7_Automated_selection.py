"""
CHOICE 7 — Automated selection and rejection of low-quality calibration frames

This file gives you drop-in utilities to improve calibration quality by **automatically selecting only high-quality frames** from your videos:

1) **Blurriness check**: Frames that are too blurry are rejected.
2) **Checkerboard coverage check**: Frames where the checkerboard is too small in the image are rejected.
3) **Border proximity check**: Frames where corners are too close to image borders are rejected.
4) **Random frame selection with filtering**: Loops randomly through frames until a sufficient number of valid frames are found.
"""

import cv2
import numpy as np
import os
import random

CHECKERBOARD_SIZE = (8, 6)
SQUARE_SIZE = 115
MAX_ATTEMPTS = 200
REQUIRED_DETECTIONS = 20
BLUR_THRESHOLD = 100
COVERAGE_THRESHOLD = 0.05
BORDER_MARGIN = 20

def create_object_points():
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0],
                           0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp

def is_blurry(image, threshold=BLUR_THRESHOLD):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def board_coverage(corners, image_shape):
    h, w = image_shape[:2]
    x_coords = corners[:, 0, 0]
    y_coords = corners[:, 0, 1]
    width = x_coords.max() - x_coords.min()
    height = y_coords.max() - y_coords.min()
    coverage = (width * height) / (w * h)
    return coverage

def too_close_to_border(corners, image_shape, margin=BORDER_MARGIN):
    h, w = image_shape[:2]
    for corner in corners:
        x, y = corner[0]
        if x < margin or x > w - margin:
            return True
        if y < margin or y > h - margin:
            return True
    return False


#  Intrinsics Calibration 
def calibrate_camera_random(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

    objp = create_object_points()
    objpoints = []
    imgpoints = []

    attempts = 0
    while attempts < MAX_ATTEMPTS and len(objpoints) < REQUIRED_DETECTIONS:
        frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            attempts += 1
            continue

        if is_blurry(frame):
            attempts += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            if too_close_to_border(corners, frame.shape):
                attempts += 1
                continue

            if board_coverage(corners, frame.shape) < COVERAGE_THRESHOLD:
                attempts += 1
                continue

            corners = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS +
                          cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"Accepted frame {len(objpoints)}/{REQUIRED_DETECTIONS}")

        attempts += 1

    cap.release()

    if len(objpoints) < 5:
        raise RuntimeError("Not enough valid checkerboard detections for intrinsics")

    ret, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return K, dist


#  Manual Corner Selection for Extrinsics 
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))


def get_manual_corners(image):
    global clicked_points
    clicked_points = []

    window_name = "Manual Corner Selection"
    clone = image.copy()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = clone.copy()
        for p in clicked_points:
            cv2.circle(display, p, 10, (0, 0, 255), -1)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if len(clicked_points) == 4 or key == 27:
            break

    cv2.destroyAllWindows()
    return np.array(clicked_points, dtype=np.float32)


def interpolate_corners(outer_corners):
    tl, tr, br, bl = outer_corners
    cols, rows = CHECKERBOARD_SIZE
    corners = []
    for r in range(rows):
        alpha = r / (rows - 1)
        left = (1 - alpha) * tl + alpha * bl
        right = (1 - alpha) * tr + alpha * br
        for c in range(cols):
            beta = c / (cols - 1)
            point = (1 - beta) * left + beta * right
            corners.append(point)
    return np.array(corners, dtype=np.float32)

if __name__ == "__main__":

    for cam_id in range(1, 5):
        cam_dir = f"data/cam{cam_id}"
        intrinsics_video = os.path.join(cam_dir, "intrinsics.avi")
        checkerboard_video = os.path.join(cam_dir, "checkerboard.avi")
        output_config = os.path.join(cam_dir, "config.xml")

        # ---- Intrinsics ----
        print(f"Calibrating intrinsics for cam{cam_id}...")
        K, dist = calibrate_camera_random(intrinsics_video)
        print("Camera matrix:\n", K)
        print("Distortion coefficients:\n", dist)

        # ---- Extrinsics ----
        cap = cv2.VideoCapture(checkerboard_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read checkerboard frame")

        print("Click the four outer corners (TL, TR, BR, BL) of the checkerboard.")
        outer = get_manual_corners(frame)
        if len(outer) != 4:
            raise RuntimeError("You must select exactly 4 corners")

        img_corners = interpolate_corners(outer)
        obj_points = create_object_points()

        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            raise RuntimeError("solvePnP failed for extrinsics")

        R, _ = cv2.Rodrigues(rvec)
        print("Rotation matrix:\n", R)
        print("Translation vector:\n", tvec)

        # Save calibration 
        fs = cv2.FileStorage(output_config, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", dist)
        fs.write("rotation_matrix", R)
        fs.write("translation_vector", tvec)
        fs.release()
        print(f"Calibration written to {output_config}")