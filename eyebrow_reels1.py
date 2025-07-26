import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# -------------------------------
# 🔹 CONFIG
# -------------------------------
COOLDOWN = 1.5  # seconds between gestures
last_swipe = time.time()
status_text = "Calibrating... Keep your face neutral."

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(0)

baseline_eyebrow = []
baseline_eye = []
calibrated = False

print("Keep your face neutral for 5 seconds...")

start_calibration = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Eyebrow vs Eye (for eyebrow raise)
        brow_y = face_landmarks.landmark[65].y
        eye_y = face_landmarks.landmark[159].y
        eyebrow_raise = (eye_y - brow_y) * 100

        # Eye openness (for blink detection)
        top_eye = face_landmarks.landmark[159].y
        bottom_eye = face_landmarks.landmark[145].y
        eye_open = (bottom_eye - top_eye) * 100

        # ✅ CALIBRATION (First 5 seconds)
        if not calibrated:
            baseline_eyebrow.append(eyebrow_raise)
            baseline_eye.append(eye_open)
            if time.time() - start_calibration > 5:
                baseline_eyebrow = np.mean(baseline_eyebrow)
                baseline_eye = np.mean(baseline_eye)
                calibrated = True
                status_text = "Calibrated ✅ Eyebrow UP = Next | Blink = Previous"
                print(f"Baseline Eyebrow:{baseline_eyebrow:.2f}, Eye:{baseline_eye:.2f}")

        else:
            if time.time() - last_swipe > COOLDOWN:
                # Eyebrow raised (Next Reel)
                if eyebrow_raise - baseline_eyebrow > 2.0:
                    status_text = "👆 Next Reel"
                    pyautogui.press("down")
                    last_swipe = time.time()

                # Eye Blink (Previous Reel)
                elif eye_open < baseline_eye * 0.5:  # eye almost closed
                    status_text = "👇 Previous Reel"
                    pyautogui.press("up")
                    last_swipe = time.time()
            else:
                status_text = "Waiting..."

        # Display on screen
        cv2.putText(frame, status_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Eyebrow:{eyebrow_raise:.2f} Eye:{eye_open:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Eyebrow + Blink Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
