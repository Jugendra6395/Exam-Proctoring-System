import cv2
import numpy as np
import mediapipe as mp
import joblib

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("proctoring_model.pkl")
encoder = joblib.load("label_encoder.pkl")

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
    max_num_faces=1,
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
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

WINDOW_NAME = "Live Proctoring"
STOP_BTN_TOP_LEFT = (500, 10)
STOP_BTN_BOTTOM_RIGHT = (620, 50)
stop_requested = False


def on_mouse(event, x, y, flags, param):
    global stop_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        if (
            STOP_BTN_TOP_LEFT[0] <= x <= STOP_BTN_BOTTOM_RIGHT[0]
            and STOP_BTN_TOP_LEFT[1] <= y <= STOP_BTN_BOTTOM_RIGHT[1]
        ):
            stop_requested = True


cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)

print("Starting Live Proctoring... Click STOP or press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detection_results = face_detection.process(rgb)
    mesh_results = face_mesh.process(rgb)

    # Default feature values
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

    if mesh_results.multi_face_landmarks and detection_results.detections:

        landmarks = mesh_results.multi_face_landmarks[0]

        # Face area
        bbox = detection_results.detections[0].location_data.relative_bounding_box
        face_area = bbox.width * bbox.height

        # Yaw
        nose = landmarks.landmark[1]
        left_cheek = landmarks.landmark[234]
        right_cheek = landmarks.landmark[454]

        nose_x = nose.x * w
        left_x = left_cheek.x * w
        right_x = right_cheek.x * w

        face_center_x = (left_x + right_x) / 2
        face_width = max((right_x - left_x), 1e-6)

        yaw = ((nose_x - face_center_x) / face_width) * 70

        # Pitch
        left_eye_y = landmarks.landmark[33].y * h
        right_eye_y = landmarks.landmark[263].y * h
        nose_y = nose.y * h

        eye_center_y = (left_eye_y + right_eye_y) / 2
        face_height = max((nose_y - eye_center_y), 1e-6)

        pitch = ((nose_y - eye_center_y) / face_height - 0.5) * 60

        # Roll
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]

        dy = (right_eye.y - left_eye.y) * h
        dx = (right_eye.x - left_eye.x) * w
        roll = np.degrees(np.arctan2(dy, dx))

        def get_point(idx):
            return np.array([
                landmarks.landmark[idx].x * w,
                landmarks.landmark[idx].y * h
            ])

        # EAR
        left_EAR = compute_ear(
            get_point(33), get_point(160), get_point(158),
            get_point(133), get_point(153), get_point(144)
        )

        right_EAR = compute_ear(
            get_point(362), get_point(385), get_point(387),
            get_point(263), get_point(373), get_point(380)
        )

        # Gaze Direction
        if yaw < -15:
            gaze_direction = 1
        elif yaw > 15:
            gaze_direction = 2
        elif pitch > 15:
            gaze_direction = 3
        elif pitch < -15:
            gaze_direction = 4
        else:
            gaze_direction = 0

        # Mouth ratio
        top_lip = get_point(13)
        bottom_lip = get_point(14)
        left_mouth = get_point(78)
        right_mouth = get_point(308)

        vertical = np.linalg.norm(top_lip - bottom_lip)
        horizontal = max(np.linalg.norm(left_mouth - right_mouth), 1e-6)

        mouth_ratio = vertical / horizontal

    # Prepare feature array
    features = np.array([[
        face_present,
        face_count,
        yaw,
        pitch,
        roll,
        left_EAR,
        right_EAR,
        gaze_direction,
        mouth_ratio,
        face_area
    ]])

    # Predict
    prediction = model.predict(features)
    predicted_label = encoder.inverse_transform(prediction)[0]

    # Display result
    color = (0, 255, 0) if predicted_label == "Normal" else (0, 0, 255)

    cv2.putText(
        frame,
        f"Status: {predicted_label}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.rectangle(frame, STOP_BTN_TOP_LEFT, STOP_BTN_BOTTOM_RIGHT, (0, 0, 255), -1)
    cv2.putText(
        frame,
        "STOP",
        (STOP_BTN_TOP_LEFT[0] + 25, STOP_BTN_TOP_LEFT[1] + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(WINDOW_NAME, frame)

    if stop_requested or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
