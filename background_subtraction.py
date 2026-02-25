import cv2
import numpy as np
import os


BACKGROUND_VIDEO = "data/cam1/background.avi"
INPUT_VIDEO = "data/cam1/video.avi"
OUTPUT_DIR = "data/cam1/foreground_masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

H_THRESH = 15      
S_THRESH = 30      
V_THRESH = 30      

NUM_BG_FRAMES = 50

def create_background_average(video_path, num_frames=50):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sum_img = None
    frames_used = 0

    for i in range(num_frames):
        idx = int(i * total_frames / num_frames)  # evenly spaced
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        if sum_img is None:
            sum_img = frame
        else:
            sum_img += frame
        frames_used += 1

    cap.release()
    if frames_used == 0:
        raise RuntimeError("No frames read from background video!")

    bg_model = (sum_img / frames_used).astype(np.uint8)
    return bg_model

def background_subtraction(frame_hsv, bg_hsv):
    # absolute difference
    diff = cv2.absdiff(frame_hsv, bg_hsv)

    h_diff, s_diff, v_diff = cv2.split(diff)

    # threshold each channel
    h_mask = (h_diff > H_THRESH).astype(np.uint8)
    s_mask = (s_diff > S_THRESH).astype(np.uint8)
    v_mask = (v_diff > V_THRESH).astype(np.uint8)

    fg_mask = ((h_mask + s_mask + v_mask) > 0).astype(np.uint8) * 255
    return fg_mask

def process_video(input_video, bg_model, output_dir):
    cap = cv2.VideoCapture(input_video)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fg_mask = background_subtraction(frame_hsv, bg_model)

        # optional: morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        out_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, fg_mask)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    print("Foreground extraction done.")


if __name__ == "__main__":
    print("Creating background model...")
    bg_model = create_background_average(BACKGROUND_VIDEO, NUM_BG_FRAMES)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "background_model.png"), bg_model)
    print("Background model saved.")

    print("Processing input video...")
    process_video(INPUT_VIDEO, bg_model, OUTPUT_DIR)