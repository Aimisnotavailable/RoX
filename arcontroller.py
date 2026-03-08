# arcontroller.py
import threading
import time
import glm
import cv2
from settings import *
from scripts.ar import AR

class ARController:
    def __init__(self, engine):
        self.engine = engine
        self.ar = AR(WIN_RES)
        self.running = True
        
        # --- Raw Thread-Safe Data (Producer) ---
        self._raw_left_landmarks = []
        self._raw_right_landmarks = []
        self._raw_left_pinch = False
        self._raw_right_pinch = False
        
        # --- Smoothed Engine Data (Consumer) ---
        self.smooth_left_landmarks = []
        self.smooth_right_landmarks = []
        
        # Engine integration points (Index fingers)
        self.smooth_left_pos = None  
        self.smooth_right_pos = None
        self.ar_mouse_pos = None
        self.ar_right_click = False
        
        self.cap = cv2.VideoCapture(0)
        # --- EMA Smoothing & Gestures ---
        # EMA Alpha: 1.0 = raw input (no smooth), 0.1 = extremely heavy smoothing
        # 0.45 is the sweet spot for fast AR tracking without jitter
        self.ema_alpha = 0.45  
        
        self.last_zoom_dist = None
        self.last_left_pinch_pos = None

        get_logger_info('AR', f'THREAD FOR AR SYSTEM INITIALIZED')
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()

    def _tracking_loop(self):
        while self.running:
            
            # Safely extract raw data dumped by your Optical Flow script
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            ar_data = None
            if ret:
                ar_data = self.ar.update(frame)

            if ar_data is None:
               continue 
            
            
            # Extract full 21 landmarks instead of just point 8
            self._raw_left_landmarks = ar_data["POSITION_DATA"].get("LEFT", [])
            self._raw_right_landmarks = ar_data["POSITION_DATA"].get("RIGHT", [])
            
            self._raw_left_pinch = ar_data["CLICK_FLAG"].get("LEFT", False)
            self._raw_right_pinch = ar_data["CLICK_FLAG"].get("RIGHT", False)
            
            time.sleep(0.016)

    def _apply_ema(self, current_smooth, raw_new):
        """Applies Exponential Moving Average to an array of 21 landmarks."""
        if not raw_new or len(raw_new) < 21:
            return []
            
        # If we just detected the hand, snap directly to raw positions
        if not current_smooth or len(current_smooth) < 21:
            return [glm.vec2(p[0], p[1]) for p in raw_new]
            
        # Apply EMA filter to each joint independently
        smoothed = []
        for i in range(21):
            target = glm.vec2(raw_new[i][0], raw_new[i][1])
            new_pos = (target * self.ema_alpha) + (current_smooth[i] * (1.0 - self.ema_alpha))
            smoothed.append(new_pos)
            
        return smoothed

    def update(self):
        # 1. APPLY EMA TO FULL SKELETONS
        self.smooth_left_landmarks = self._apply_ema(self.smooth_left_landmarks, self._raw_left_landmarks)
        self.smooth_right_landmarks = self._apply_ema(self.smooth_right_landmarks, self._raw_right_landmarks)

        # 2. EXTRACT INDEX TIPS FOR ENGINE CONTROLS (Landmark 8)
        if len(self.smooth_left_landmarks) > 8:
            self.smooth_left_pos = self.smooth_left_landmarks[8]
        else:
            self.smooth_left_pos = None

        if len(self.smooth_right_landmarks) > 8:
            self.smooth_right_pos = self.smooth_right_landmarks[8]
            self.ar_mouse_pos = (self.smooth_right_pos.x, self.smooth_right_pos.y)
            self.ar_right_click = self._raw_right_pinch
        else:
            self.smooth_right_pos = None
            self.ar_mouse_pos = None
            self.ar_right_click = False

        # 3. GESTURE CONTROLS (Rotation & Zoom)
        world = self.engine.scene.world
        is_left_pinched = self._raw_left_pinch and self.smooth_left_pos is not None
        is_right_pinched = self._raw_right_pinch and self.smooth_right_pos is not None

        if is_left_pinched and is_right_pinched:
            current_dist = glm.distance(self.smooth_left_pos, self.smooth_right_pos)
            if self.last_zoom_dist is not None:
                delta_zoom = (current_dist - self.last_zoom_dist) * 0.005
                world.world_scale += delta_zoom
                world.world_scale = max(0.1, min(10.0, world.world_scale))
            self.last_zoom_dist = current_dist
            self.last_left_pinch_pos = None 
            
        elif is_left_pinched and not is_right_pinched:
            self.last_zoom_dist = None 
            if self.last_left_pinch_pos is not None:
                delta_rot = self.smooth_left_pos - self.last_left_pinch_pos
                world.world_yaw += delta_rot.x * 0.005
                world.world_pitch += delta_rot.y * 0.005
            self.last_left_pinch_pos = glm.vec2(self.smooth_left_pos)
            
        else:
            self.last_zoom_dist = None
            self.last_left_pinch_pos = None