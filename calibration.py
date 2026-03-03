import cv2
import numpy as np
import random

CHECKERBOARD_SIZE = (8, 6)
SQUARE_SIZE = 115
MAX_ATTEMPTS = 200
REQUIRED_DETECTIONS = 20


def create_object_points():
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0],
                           0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
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
            print(f"Found checkerboard ({len(objpoints)}/{REQUIRED_DETECTIONS})")

        attempts += 1

    cap.release()

    if len(objpoints) < 5:
        raise RuntimeError("Not enough checkerboard detections")

    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return mtx, dist


# ================= Manual Corner Selection =================

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

        if len(clicked_points) == 4:
            break

        if cv2.waitKey(1) & 0xFF == 27:
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


# ================= Main =================

if __name__ == "__main__":

    for cam_id in range(1, 5):

        cam_dir = f"data/cam{cam_id}"

        intrinsics_video = f"{cam_dir}/intrinsics.avi"
        checkerboard_video = f"{cam_dir}/checkerboard.avi"
        output_config = f"{cam_dir}/config.xml"

        # ---- Intrinsics ----
        K, dist = calibrate_camera_random(intrinsics_video)

        print("Camera matrix:\n", K)
        print("Distortion coefficients:\n", dist)

        # ---- Load checkerboard frame ----
        cap = cv2.VideoCapture(checkerboard_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Could not read checkerboard frame")

        # ---- Manual corner selection ----
        print("Click 4 corners: TL, TR, BR, BL")
        outer = get_manual_corners(frame)

        if len(outer) != 4:
            raise RuntimeError("You must click exactly 4 points")

        img_corners = interpolate_corners(outer)
        obj_points = create_object_points()

        # ---- SolvePnP ----
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_corners,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            raise RuntimeError("solvePnP failed")

        R, _ = cv2.Rodrigues(rvec)

        print("Rotation matrix:\n", R)
        print("Translation vector:\n", tvec)

        # ---- Visualization ----
        s = SQUARE_SIZE
        axes = np.float32([[0, 0, 0], [3*s, 0, 0],
                           [0, 3*s, 0], [0, 0, -3*s]])

        imgpts, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)

        origin = tuple(imgpts[0])
        cv2.line(frame, origin, tuple(imgpts[1]), (0, 0, 255), 3)
        cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 3)
        cv2.line(frame, origin, tuple(imgpts[3]), (255, 0, 0), 3)

        cv2.imshow("World origin check", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(f"{cam_dir}/extrinsics_check.png", frame)

        # ---- Save ----
        fs = cv2.FileStorage(output_config, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", dist)
        fs.write("rotation_matrix", R)
        fs.write("translation_vector", tvec)
        fs.release()

        print(f"Calibration saved to {output_config}")