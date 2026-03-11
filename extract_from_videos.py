# py -3.10 "c:\Users\jugen\OneDrive\Desktop\project\Project Phase 2\extract_from_videos.py

# py -3.10 "c:/Users/jugen/OneDrive/Desktop/project/Project Phase 2/extract_from_videos.py
# py -3.10 "c:/Users/jugen/OneDrive/Desktop/project/Project Phase 2/train_model.py
# py -3.10 "c:/Users/jugen/OneDrive/Desktop/project/Project Phase 2/live_proctoring.py



import os
import cv2
import csv
import numpy as np
import mediapipe as mp

# -----------------------------
# Dataset Path
# -----------------------------
DATASET_PATH = "dataset"
OUTPUT_CSV = "video_proctoring_data.csv"

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -----------------------------
# EAR Function
# -----------------------------
def compute_ear(p1, p2, p3, p4, p5, p6):
    return (
        np.linalg.norm(p2 - p6) +
        np.linalg.norm(p3 - p5)
    ) / (2.0 * max(np.linalg.norm(p1 - p4), 1e-6))

# -----------------------------
# Check Dataset Folder
# -----------------------------
if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")

# -----------------------------
# CSV Setup
# -----------------------------
with open(OUTPUT_CSV, "w", newline="") as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "face_present",
        "face_count",
        "yaw",
        "pitch",
        "roll",
        "left_EAR",
        "right_EAR",
        "gaze_direction",
        "mouth_ratio",
        "face_area",
        "label"
    ])

    # -----------------------------
    # Process Videos
    # -----------------------------
    for label in ["normal", "suspicious"]:

        folder = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        for video_file in os.listdir(folder):

            video_path = os.path.join(folder, video_file)
            if not os.path.isfile(video_path):
                continue

            print("Processing:", video_file)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Skipping unreadable video: {video_path}")
                continue

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Sample 1 frame every 10 frames (Optimization)
                if frame_count % 10 != 0:
                    continue

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                detection_results = face_detection.process(rgb)
                mesh_results = face_mesh.process(rgb)

                # -----------------------------
                # Default Feature Values
                # -----------------------------
                face_present = 1 if detection_results.detections else 0
                face_count = len(detection_results.detections) if detection_results.detections else 0

                yaw = 0.0
                pitch = 0.0
                roll = 0.0
                left_EAR = 0.0
                right_EAR = 0.0
                gaze_direction = 0
                mouth_ratio = 0.0
                face_area = 0.0

                # -----------------------------
                # Extract Landmarks
                # -----------------------------
                if mesh_results.multi_face_landmarks and detection_results.detections:

                    landmarks = mesh_results.multi_face_landmarks[0]

                    # -------- FACE AREA --------
                    bbox = detection_results.detections[0].location_data.relative_bounding_box
                    face_area = bbox.width * bbox.height

                    # -------- YAW --------
                    nose = landmarks.landmark[1]
                    left_cheek = landmarks.landmark[234]
                    right_cheek = landmarks.landmark[454]

                    nose_x = nose.x * w
                    left_x = left_cheek.x * w
                    right_x = right_cheek.x * w

                    face_center_x = (left_x + right_x) / 2
                    face_width = max((right_x - left_x), 1e-6)

                    yaw = ((nose_x - face_center_x) / face_width) * 70

                    # -------- PITCH --------
                    left_eye_y = landmarks.landmark[33].y * h
                    right_eye_y = landmarks.landmark[263].y * h
                    nose_y = nose.y * h

                    eye_center_y = (left_eye_y + right_eye_y) / 2
                    face_height = max((nose_y - eye_center_y), 1e-6)

                    pitch = ((nose_y - eye_center_y) / face_height - 0.5) * 60

                    # -------- ROLL --------
                    left_eye = landmarks.landmark[33]
                    right_eye = landmarks.landmark[263]

                    dy = (right_eye.y - left_eye.y) * h
                    dx = (right_eye.x - left_eye.x) * w
                    roll = np.degrees(np.arctan2(dy, dx))

                    # -------- Helper Function --------
                    def get_point(idx):
                        return np.array([
                            landmarks.landmark[idx].x * w,
                            landmarks.landmark[idx].y * h
                        ])

                    # -------- EAR --------
                    left_EAR = compute_ear(
                        get_point(33), get_point(160), get_point(158),
                        get_point(133), get_point(153), get_point(144)
                    )

                    right_EAR = compute_ear(
                        get_point(362), get_point(385), get_point(387),
                        get_point(263), get_point(373), get_point(380)
                    )

                    # -------- GAZE DIRECTION --------
                    if yaw < -15:
                        gaze_direction = 1   # Left
                    elif yaw > 15:
                        gaze_direction = 2   # Right
                    elif pitch > 15:
                        gaze_direction = 3   # Up
                    elif pitch < -15:
                        gaze_direction = 4   # Down
                    else:
                        gaze_direction = 0   # Center

                    # -------- MOUTH RATIO --------
                    top_lip = get_point(13)
                    bottom_lip = get_point(14)
                    left_mouth = get_point(78)
                    right_mouth = get_point(308)

                    vertical = np.linalg.norm(top_lip - bottom_lip)
                    horizontal = max(np.linalg.norm(left_mouth - right_mouth), 1e-6)

                    mouth_ratio = vertical / horizontal

                # -----------------------------
                # Write Row
                # -----------------------------
                writer.writerow([
                    face_present,
                    face_count,
                    round(yaw, 2),
                    round(pitch, 2),
                    round(roll, 2),
                    round(left_EAR, 3),
                    round(right_EAR, 3),
                    gaze_direction,
                    round(mouth_ratio, 3),
                    round(face_area, 3),
                    label.capitalize()
                ])

            cap.release()

print("Feature extraction completed successfully!")
