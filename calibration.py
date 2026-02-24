import cv2
import numpy as np
import glob
import os

CHECKERBOARD_SIZE = (8, 6)
SQUARE_SIZE = 115

def get_frames(video_path):
    cap = cv2.VideoCapture(f"{video_path}/intrinsics.avi")
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()

    total = len(frames)
    start = total // 4
    end = 3 * total // 4
    return frames[start:end]

def create_object_points():
    """
    Creates 3D object points for the checkerboard in world coordinates.
    """
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0],
                           0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp

def calibrate_camera(frames):
    objp = create_object_points()

    objpoints = []
    imgpoints = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist
    else:
        raise ValueError("No checkerboard corners found in the provided frames.")
    



if __name__ == "__main__":
    video_path = "cam1"
    frames = get_frames(video_path)
    print(f"Extracted {len(frames)} frames for calibration.")
    # mtx, dist = calibrate_camera(frames)
    # print("Camera Matrix:\n", mtx)
    # print("Distortion Coefficients:\n", dist)