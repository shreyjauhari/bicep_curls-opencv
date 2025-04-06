import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
from tkinter import *
from PIL import Image, ImageTk
import time

# ---------- Angle Calculation ----------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = acos(np.clip(np.dot((a - b), (c - b)) /
                           (np.linalg.norm(a - b) * np.linalg.norm(c - b)), -1.0, 1.0))
    return degrees(radians)

# ---------- Smooth Landmarks ----------
def smooth_landmarks(prev_lm, curr_lm, alpha=0.2):
    return [
        (int(prev[0] * (1 - alpha) + curr[0] * alpha),
         int(prev[1] * (1 - alpha) + curr[1] * alpha))
        for prev, curr in zip(prev_lm, curr_lm)
    ]

# ---------- Accurate Synchronized Curl Counter ----------
class CurlCounter:
    def __init__(self):
        self.left_down = False
        self.right_down = False
        self.left_up_time = None
        self.right_up_time = None
        self.last_count_time = 0
        self.count = 0
        self.sync_threshold = 0.4  # seconds
        self.cooldown = 1.0        # seconds
        self.min_angle = 35
        self.max_angle = 155

    def update(self, left_angle, right_angle):
        current_time = time.time()

        if not (0 < left_angle < 180) or not (0 < right_angle < 180):
            return self.count

        if left_angle > self.max_angle:
            self.left_down = True
        if right_angle > self.max_angle:
            self.right_down = True

        if self.left_down and left_angle < self.min_angle:
            self.left_up_time = current_time
        if self.right_down and right_angle < self.min_angle:
            self.right_up_time = current_time

        if self.left_up_time and self.right_up_time:
            if abs(self.left_up_time - self.right_up_time) < self.sync_threshold:
                if current_time - self.last_count_time > self.cooldown:
                    self.count += 1
                    self.last_count_time = current_time
                    print(f"✅ Accurate Synchronized Curl Count: {self.count}")
                    self.left_down = self.right_down = False
                    self.left_up_time = self.right_up_time = None

        return self.count

# ---------- ESC Button Area (Top-Left) ----------
exit_button = (20, 20, 120, 60)

# ---------- Initialize MediaPipe ----------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1)

counter = CurlCounter()
cap = cv2.VideoCapture(0)
prev_landmarks = None

# ---------- Tkinter GUI ----------
root = Tk()
root.title("Synchronized Bicep Curl Counter")

video_label = Label(root)
video_label.grid(row=0, column=0, padx=10, pady=10)

count_label = Label(root, text="Curls: 0", font=("Helvetica", 20))
count_label.grid(row=1, column=0)

# ---------- Exit Function ----------
def exit_app():
    cap.release()
    root.destroy()

# ---------- Keyboard ESC Key Bind ----------
def key_event(event):
    if event.keysym == 'Escape':
        print("🧑‍💻 ESC key pressed! Exiting...")
        exit_app()

root.bind('<Escape>', key_event)

# ---------- Main Frame Update ----------
def update_frame():
    global prev_landmarks

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb)
    hands_results = hands.process(rgb)

    if pose_results.pose_landmarks:
        raw_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in pose_results.pose_landmarks.landmark]
        landmarks = smooth_landmarks(prev_landmarks, raw_landmarks) if prev_landmarks else raw_landmarks
        prev_landmarks = landmarks

        try:
            l_shoulder, l_elbow, l_wrist = landmarks[11], landmarks[13], landmarks[15]
            r_shoulder, r_elbow, r_wrist = landmarks[12], landmarks[14], landmarks[16]

            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

            counter.update(l_angle, r_angle)

            cv2.putText(frame, f'L: {int(l_angle)}', (l_elbow[0] + 10, l_elbow[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f'R: {int(r_angle)}', (r_elbow[0] + 10, r_elbow[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        except IndexError:
            pass

    # Draw ESC Button
    x, y, w_btn, h_btn = exit_button
    cv2.rectangle(frame, (x - 2, y - 2), (x + w_btn + 2, y + h_btn + 2), (0, 0, 0), 4)
    cv2.rectangle(frame, (x, y), (x + w_btn, y + h_btn), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "ESC", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Gesture ESC
    if hands_results.multi_hand_landmarks:
        for handLms in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            index_tip = handLms.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            if x < cx < x + w_btn and y < cy < y + h_btn:
                print("👋 Exit via hand gesture")
                exit_app()

    # Update Tkinter image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Update curl count label
    count_label.configure(text=f"Curls: {counter.count}")

    root.after(10, update_frame)

# ---------- Start ----------
update_frame()
root.mainloop()
