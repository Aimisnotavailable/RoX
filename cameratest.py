# test_mediapipe_camera.py
import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("cap.isOpened:", cap.isOpened())

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cap.read failed")
            break
        print("frame.shape:", frame.shape)
        # convert BGR->RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        print("multi_hand_landmarks:", bool(res.multi_hand_landmarks))
        out = frame.copy()
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_drawing.draw_landmarks(out, lm, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("MP Test", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
