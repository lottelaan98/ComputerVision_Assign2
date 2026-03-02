import cv2
import numpy as np
import random

CHECKERBOARD_SIZE = (8, 6)  # number of inner corners (columns = 8, rows=6)
SQUARE_SIZE = 115   # real world square size in mm
MAX_ATTEMPTS = 200      # how many random frames to try
REQUIRED_DETECTIONS = 20  # how many valid checkerboards

def create_object_points():
    # creates an array of shape (48, 3) filled with zeros
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    # np.mgrid build a grid of coordinates, and reshapes into a list of (x,y) points
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0],
                           0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    # these fill the first two columns (X,Y). Z stays 0
    objp *= SQUARE_SIZE
    # returns the 3D corner coordinates on the board plane.
    return objp

def calibrate_camera_random(video_path):
    # opens the video intrinsics.avi
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video")

    # reads how many frames exist in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

    # objp = 3D points for one chessboard view
    objp = create_object_points()
    # ojbpoints = list of 3D points. One set per detected frame.
    objpoints = []
    # imgpoints = list of detected 2D image corners. One set per detected frame.
    imgpoints = []

    # counts for random tries
    attempts = 0

    # loop: try random frames until enough boards detected or max attempts reached
    while attempts < MAX_ATTEMPTS and len(objpoints) < REQUIRED_DETECTIONS:
        # pick a random frame index and jump to it
        frame_idx = random.randint(0, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # read the frame. If failed, skip.
        ret, frame = cap.read()
        if not ret:
            attempts += 1
            continue

        # convert to grayscale, cause chessboard detection works on gray.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # try to find 8x6 corner pattern
        # flags improve detection under different lighting 
        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            # refine corners for better accuracy
            corners = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )

            # store 3D-2D correspondences for calibration
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"Found checkerboard ({len(objpoints)}/{REQUIRED_DETECTIONS})")

        attempts += 1

    cap.release()

    # need at least a few views to calibrate
    if len(objpoints) < 5:
        raise RuntimeError("Not enough checkerboard detections for calibration")

    # mtx is intrinsic camera matrix K
    # dist is distortion coefficients
    # rvecs, tvecs is pose of board in each used frame
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # return intrinsics only
    return mtx, dist

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback to collect user clicks.
    """
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))


def get_manual_corners(image):
    """
    Allows the user to manually click the four outer checkerboard corners.
    """
    global clicked_points
    clicked_points = []

    clone = image.copy()

    cv2.namedWindow("Manual Corner Selection: From top left, to top right, to bottom right, to bottom left", cv2.WINDOW_NORMAL)

    cv2.setMouseCallback("Manual Corner Selection: From top left, to top right, to bottom right, to bottom left", mouse_callback)

    while True:
        display = clone.copy()
        for p in clicked_points:
            cv2.circle(display, p, 10, (0, 0, 255), -1)

        cv2.imshow("Manual Corner Selection: From top left, to top right, to bottom right, to bottom left", display)

        if len(clicked_points) == 4:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return np.array(clicked_points, dtype=np.float32)

def interpolate_corners(outer_corners):
    """
    Linearly interpolates all inner checkerboard corners from four outer corners.
    """
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


def compute_extrinsics(image, K, dist):
    """
    Computes camera extrinsics (R, t) using manual checkerboard corners.
    """
    # FORCE manual selection
    outer = get_manual_corners(image)
    img_corners = interpolate_corners(outer)

    obj_points = create_object_points()

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
    return R, tvec

def get_checkerboard_frame(video_path, frame_idx=100):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read checkerboard frame")

    return frame

def save_camera_config(path, K, dist, R, t):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)

    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist)
    fs.write("rotation_matrix", R)
    fs.write("translation_vector", t)

    fs.release()

if __name__ == "__main__":

    # -------- paths --------
    # hier moet nog een for loop voor alle 4 de camera's, maar voor nu even 1 per keer
    for cam_id in range(1, 5):
        cam_dir = f"data/cam{cam_id}"

        intrinsics_path = f"{cam_dir}/intrinsics.xml"
        intrinsics_video = f"{cam_dir}/intrinsics.avi"
        checkerboard_video = f"{cam_dir}/checkerboard.avi"
        output_config = f"{cam_dir}/config.xml"

        # -------- load intrinsics --------
        K, dist = calibrate_camera_random(intrinsics_video)

        print("Camera matrix:\n", K)
        print("Distortion coefficients:\n", dist)

        # -------- get checkerboard frame --------
        cap = cv2.VideoCapture(checkerboard_video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)   # change if blurred
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Could not read checkerboard frame")

        # -------- manual corner selection --------
        print("Click 4 corners: top-left top-right bottom-right bottom-left")
        outer = get_manual_corners(frame)
        img_corners = interpolate_corners(outer)

        # -------- object points --------
        obj_points = create_object_points()

        # -------- solve PnP (extrinsics) --------
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

        # -------- visualization check --------
        s = SQUARE_SIZE
        axes = np.float32([[0,0,0],[3*s,0,0],[0,3*s,0],[0,0,-3*s]])

        imgpts, _ = cv2.projectPoints(axes, rvec, tvec, K, dist)
        imgpts = imgpts.reshape(-1, 2).astype(int)

            # Draw axes
        o = tuple(imgpts[0].astype(int))
        cv2.line(frame, o, tuple(imgpts[1].astype(int)), (0, 0, 255), 3)
        cv2.line(frame, o, tuple(imgpts[2].astype(int)), (0, 255, 0), 3)
        cv2.line(frame, o, tuple(imgpts[3].astype(int)), (255, 0, 0), 3)

        #cv2.circle(frame, tuple(imgpts[0]), 10, (0, 0, 255), -1)
        cv2.imshow("World origin check", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(f"{cam_dir}/extrinsics_check.png", frame)

        # -------- save final config --------
        fs = cv2.FileStorage(output_config, cv2.FILE_STORAGE_WRITE)

        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", dist)
        fs.write("rotation_matrix", R)
        fs.write("translation_vector", tvec)

        fs.release()

        print(f"Calibration saved to {output_config}")