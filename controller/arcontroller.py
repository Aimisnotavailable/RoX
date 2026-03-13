# arcontroller.py
import threading
import time
import glm
import cv2
import math
import numpy as np
from settings import *
from controller.ar import AR
from controller.gesture_detector import GestureManager, PinchDetector, TwoFingerUpDetector
from scripts.logger import get_logger_info

class ARController:
    def __init__(self, engine):
        self.engine = engine
        self.ar = AR(WIN_RES)
        self.cap = cv2.VideoCapture(0)
        
        self.running = True
        
        # Gesture manager
        self.gesture_manager = GestureManager()
        # Add detectors (with optional hold frames)
        self.gesture_manager.add_detector(PinchDetector('LEFT', hold_frames=15))   # 15 frames ~0.5s at 30fps
        self.gesture_manager.add_detector(PinchDetector('RIGHT', hold_frames=15))
        self.gesture_manager.add_detector(TwoFingerUpDetector('RIGHT', hold_frames=None))
        
        # Raw data from AR thread
        self._raw_left_landmarks = []
        self._raw_right_landmarks = []
        self._raw_left_pinch = False      # will be updated from gesture events
        self._raw_right_pinch = False
        self._hand_type_left = "REAL"
        self._hand_type_right = "REAL"
        
        # Smoothed landmarks and positions
        self.smooth_left_landmarks = []
        self.smooth_right_landmarks = []
        self.ema_alpha = 0.1
        
        self.smooth_left_pos = None  
        self.smooth_right_pos = None
        self.ar_mouse_pos = None
        self.ar_right_click = False
        
        # Pinch state tracking (for drag delta and start positions)
        self.pinch_active_left = False
        self.pinch_active_right = False
        self.pinch_start_left = None   # (x_norm, y_norm) at start
        self.pinch_start_right = None
        self.pinch_current_left = None
        self.pinch_current_right = None
        
        # Gesture states for zoom/rotate
        self.last_rotate_pos = None
        self.last_zoom_dist = None

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
            # Note: ar_data["CLICK_FLAG"] is now always False; we ignore it.
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

        # Smooth landmarks (same as before)
        self.smooth_left_landmarks = self._apply_ema(self.smooth_left_landmarks, self._raw_left_landmarks)
        self.smooth_right_landmarks = self._apply_ema(self.smooth_right_landmarks, self._raw_right_landmarks)

        # Convert to tuples for gesture detectors (only if hand is REAL)
        left_tuples = []
        right_tuples = []

        if self._hand_type_left == "REAL" and self.smooth_left_landmarks:
            left_tuples = [(v.x, v.y, v.z) for v in self.smooth_left_landmarks]
        if self._hand_type_right == "REAL" and self.smooth_right_landmarks:
            right_tuples = [(v.x, v.y, v.z) for v in self.smooth_right_landmarks]

        # Process gestures only for real hands
        events = self.gesture_manager.process_both(left_tuples, right_tuples, now)

        # Reset pinch flags (they will be set by events if real hand is pinched)
        self._raw_left_pinch = False
        self._raw_right_pinch = False

        # If a hand is ghost, force its pinch state to False
        if self._hand_type_left != "REAL":
            self.pinch_active_left = False
            self.pinch_start_left = None
            self.pinch_current_left = None
        if self._hand_type_right != "REAL":
            self.pinch_active_right = False
            self.pinch_start_right = None
            self.pinch_current_right = None

        # Handle gesture events (only from real hands)
        for ev in events:
            if ev.gesture_name == 'pinch':
                if ev.hand == 'LEFT':
                    if ev.event_type == 'START':
                        self.pinch_active_left = True
                        self._raw_left_pinch = True
                        self.pinch_start_left = self.smooth_left_pos  # store norm pos
                    elif ev.event_type == 'UPDATE':
                        self._raw_left_pinch = True
                        self.pinch_current_left = self.smooth_left_pos
                    elif ev.event_type == 'END':
                        self.pinch_active_left = False
                        self._raw_left_pinch = False
                        self.pinch_start_left = None
                    elif ev.event_type == 'HOLD':
                        get_logger_info('AR', 'Left pinch HOLD detected')
                        # Example: toggle interaction mode
                        self.engine.scene.world.voxel_handler.switch_mode()
                elif ev.hand == 'RIGHT':
                    if ev.event_type == 'START':
                        self.pinch_active_right = True
                        self._raw_right_pinch = True
                        self.pinch_start_right = self.smooth_right_pos
                    elif ev.event_type == 'UPDATE':
                        self._raw_right_pinch = True
                        self.pinch_current_right = self.smooth_right_pos
                    elif ev.event_type == 'END':
                        self.pinch_active_right = False
                        self._raw_right_pinch = False
                        self.pinch_start_right = None
                    elif ev.event_type == 'HOLD':
                        get_logger_info('AR', 'Right pinch HOLD detected')
                        # Example: open radial menu (placeholder)
            elif ev.gesture_name == 'twofingerup' and ev.hand == 'RIGHT':
                if ev.event_type == 'START':
                    get_logger_info('AR', 'Two-finger up START')
                    # Could start some mode
                elif ev.event_type == 'UPDATE':
                    # Use ev.value (average height) for something
                    pass
                elif ev.event_type == 'END':
                    pass

        # Update smoothed positions (index tip)
        self.smooth_left_pos = self.smooth_left_landmarks[8] if len(self.smooth_left_landmarks) > 8 else None
        self.smooth_right_pos = self.smooth_right_landmarks[8] if len(self.smooth_right_landmarks) > 8 else None

        # Set AR mouse position and click (for crosshair and voxel placement)
        if self.smooth_right_pos is not None:
            self.ar_mouse_pos = (self.smooth_right_pos.x * WIN_RES[0], self.smooth_right_pos.y * WIN_RES[1])
        else:
            self.ar_mouse_pos = None
        self.ar_right_click = self.pinch_active_right

        # Gesture dispatch for world manipulation (original logic, now using pinch_active flags)
        world = self.engine.scene.world
        l_active = self.pinch_active_left
        r_active = self.pinch_active_right

        # Get pixel positions for gestures
        l_pixel = (self.smooth_left_pos.x * WIN_RES[0], self.smooth_left_pos.y * WIN_RES[1]) if self.smooth_left_pos else None
        r_pixel = (self.smooth_right_pos.x * WIN_RES[0], self.smooth_right_pos.y * WIN_RES[1]) if self.smooth_right_pos else None

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