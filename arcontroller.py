# arcontroller.py
import threading
import time
import glm
import cv2
import math
import numpy as np
from settings import *
from scripts.ar import AR
from src.handstateaction import HandActionState

class ARController:
    def __init__(self, engine):
        self.engine = engine
        self.ar = AR(WIN_RES)
        self.cap = cv2.VideoCapture(0)
        
        self.running = True
        
        self.left_state = HandActionState()
        self.right_state = HandActionState()
        
        self._raw_left_landmarks = []
        self._raw_right_landmarks = []
        self._raw_left_pinch = False
        self._raw_right_pinch = False
        
        self.smooth_left_landmarks = []
        self.smooth_right_landmarks = []
        self.ema_alpha = 0.1
        
        self.smooth_left_pos = None  
        self.smooth_right_pos = None
        self.ar_mouse_pos = None
        self.ar_right_click = False
        self._hand_type_left = "REAL"
        self._hand_type_right = "REAL"
        
        # Gesture states
        self.last_rotate_pos = None
        self.last_zoom_dist = None          # for two‑hand zoom

        get_logger_info('AR', 'THREAD FOR AR SYSTEM INITIALIZED')
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()

    def _tracking_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            ar_data = self.ar.update(frame)
            if ar_data is None:
                continue

            self._raw_left_landmarks = ar_data["POSITION_DATA"].get("LEFT", [])
            self._raw_right_landmarks = ar_data["POSITION_DATA"].get("RIGHT", [])
            self._raw_left_pinch = ar_data["CLICK_FLAG"].get("LEFT", False)
            self._raw_right_pinch = ar_data["CLICK_FLAG"].get("RIGHT", False)
            self._hand_type_left = ar_data['FRAME_TYPE'].get("LEFT", "REAL")
            self._hand_type_right = ar_data['FRAME_TYPE'].get("RIGHT", "REAL")

    def _apply_ema(self, current_smooth, raw_new):
        if not raw_new or len(raw_new) < 21:
            return []
        if not current_smooth or len(current_smooth) < 21:
            return [glm.vec3(p[0], p[1], p[2]) for p in raw_new]
        smoothed = []
        for i in range(21):
            target = glm.vec3(raw_new[i][0], raw_new[i][1], raw_new[i][2])
            new_pos = (target * self.ema_alpha) + (current_smooth[i] * (1.0 - self.ema_alpha))
            smoothed.append(new_pos)
        return smoothed

    def update(self):
        now = time.time()

        self.smooth_left_landmarks = self._apply_ema(self.smooth_left_landmarks, self._raw_left_landmarks)
        self.smooth_right_landmarks = self._apply_ema(self.smooth_right_landmarks, self._raw_right_landmarks)

        l_index = self.smooth_left_landmarks[8] if len(self.smooth_left_landmarks) > 8 else None
        r_index = self.smooth_right_landmarks[8] if len(self.smooth_right_landmarks) > 8 else None

        if l_index:
            l_pixel = (l_index.x * WIN_RES[0], l_index.y * WIN_RES[1])
            l_norm  = (l_index.x, l_index.y)
        else:
            l_pixel = None
            l_norm  = None

        if r_index:
            r_pixel = (r_index.x * WIN_RES[0], r_index.y * WIN_RES[1])
            r_norm  = (r_index.x, r_index.y)
        else:
            r_pixel = None
            r_norm  = None

        self.left_state.update(self._raw_left_pinch, l_norm, now)
        self.right_state.update(self._raw_right_pinch, r_norm, now)

        self.smooth_left_pos = l_index
        self.smooth_right_pos = r_index

        if self.right_state.current_pos is not None:
            self.ar_mouse_pos = (self.right_state.current_pos[0] * WIN_RES[0],
                                self.right_state.current_pos[1] * WIN_RES[1])
        else:
            self.ar_mouse_pos = None
        self.ar_right_click = self.right_state.active

        # Gesture dispatch
        world = self.engine.scene.world
        l_active = self.left_state.active
        r_active = self.right_state.active

        # --- Two‑hand pinch: zoom ---
        if l_active and r_active and l_pixel and r_pixel:
            current_dist = math.hypot(l_pixel[0] - r_pixel[0], l_pixel[1] - r_pixel[1])
            if self.last_zoom_dist is not None:
                delta_zoom = (current_dist - self.last_zoom_dist) * 0.005
                world.world_scale += delta_zoom
                world.world_scale = max(0.1, min(10.0, world.world_scale))
            self.last_zoom_dist = current_dist
            self.last_rotate_pos = None

        # --- Left hand only: rotate ---
        elif l_active and not r_active and l_pixel:
            self.last_zoom_dist = None
            if self.last_rotate_pos is not None:
                dx = l_pixel[0] - self.last_rotate_pos[0]
                dy = l_pixel[1] - self.last_rotate_pos[1]
                world.world_yaw   += dx * 0.01
                world.world_pitch += dy * 0.01
            self.last_rotate_pos = l_pixel

        else:
            self.last_zoom_dist = None
            self.last_rotate_pos = None