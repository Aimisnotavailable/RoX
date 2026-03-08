import threading
import cv2
import math
from settings import *
from scripts.ar import AR 

class ARController:
    def __init__(self, engine):
        self.engine = engine
        # Initialize your custom AR system (Mediapipe + Ghost Frames)
        self.ar_system = AR(screen_dim=WIN_RES)
        self.cap = cv2.VideoCapture("demo\demo.mp4")
        
        # --- Smoothing (EMA) Logic ---
        # 0.3 means 30% new data, 70% old data. 
        # Lower = smoother but more lag. Higher = raw jitter.
        self.smooth_alpha = 0.3 
        self.smoothed_landmarks = {"LEFT": None, "RIGHT": None}
        
        # Shared data
        self.is_running = True
        self.current_action_label = "IDLE"
        self.ar_data = None 
        self.frame = None

        # State tracking for gestures
        self.last_rotate_pos = None
        self.last_zoom_dist = None
        
        # HUD Helpers
        self.l_pos, self.r_pos = None, None
        self.l_click, self.r_click = False, False
        
        # Threading setup
        print('[AR] Starting Ghost-Frame Tracking Thread...')
        self.thread = threading.Thread(target=self._ar_loop, daemon=True)
        self.thread.start()

    def _apply_ema(self, label, new_landmarks):
        """Reduces violent shaking by averaging movement over time."""
        if not new_landmarks or len(new_landmarks) < 21:
            self.smoothed_landmarks[label] = None
            return None
        
        # If first time seeing hand, don't smooth
        if self.smoothed_landmarks[label] is None:
            self.smoothed_landmarks[label] = new_landmarks
            return new_landmarks

        prev_lms = self.smoothed_landmarks[label]
        smoothed = []
        
        for i in range(21):
            # EMA Formula: (New * Alpha) + (Previous * (1 - Alpha))
            px = (new_landmarks[i][0] * self.smooth_alpha) + (prev_lms[i][0] * (1 - self.smooth_alpha))
            py = (new_landmarks[i][1] * self.smooth_alpha) + (prev_lms[i][1] * (1 - self.smooth_alpha))
            smoothed.append((px, py))
            
        self.smoothed_landmarks[label] = smoothed
        return smoothed

    def _ar_loop(self):
        """Threaded camera processing loop."""
        while self.is_running:
            success, frame = self.cap.read()
            if not success: continue
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            self.frame = frame 
            
            # Update the AR Ghost-Frame system
            # This returns the raw dict with POSITION_DATA, CLICK_FLAG, etc.
            raw_data = self.ar_system.update(frame)
            
            # Apply EMA smoothing to the landmarks to stop the shaking
            for label in ["LEFT", "RIGHT"]:
                raw_lms = raw_data["POSITION_DATA"][label]
                raw_data["POSITION_DATA"][label] = self._apply_ema(label, raw_lms)
            
            self.ar_data = raw_data

    def update(self):
        """Main Engine Update: Translates Hand Data into Voxel Actions."""
        if not self.ar_data: 
            return
        
        data = self.ar_data
        self.current_action_label = "IDLE"

        # 1. Extract Hand positions (using Index Tip - Landmark 8)
        # We use the smoothed landmarks for steady control
        l_lms = data["POSITION_DATA"]["LEFT"]
        r_lms = data["POSITION_DATA"]["RIGHT"]
        
        self.l_click = data["CLICK_FLAG"]["LEFT"]
        self.r_click = data["CLICK_FLAG"]["RIGHT"]
        
        # Get coordinates for the HUD cursor / Voxel interaction
        self.l_pos = l_lms[8] if l_lms else None
        self.r_pos = r_lms[8] if r_lms else None

        # --- Update Engine Interface ---
        # Map right hand to the mouse for building
        self.engine.ar_right_click = self.r_click
        if self.r_pos:
            # Scale normalized 0.0-1.0 back to screen pixels for the raycaster
            self.engine.ar_mouse_pos = (self.r_pos[0] * WIN_RES[0], self.r_pos[1] * WIN_RES[1])

        # --- GESTURE LOGIC ---

        # 1. WORLD ROTATION (Left Hand Pinch)
        if self.l_click and self.l_pos and not self.r_click:
            self.current_action_label = "ROTATING"
            if self.last_rotate_pos is None:
                self.last_rotate_pos = self.l_pos
            
            # Calculate delta movement
            dx = self.l_pos[0] - self.last_rotate_pos[0]
            dy = self.l_pos[1] - self.last_rotate_pos[1]
            
            # Update World angles in Scene
            self.engine.scene.world.world_yaw += dx * 2.0
            self.engine.scene.world.world_pitch += dy * 2.0
            self.last_rotate_pos = self.l_pos
        else:
            self.last_rotate_pos = None

        # 2. WORLD ZOOM (Double Hand Pinch)
        if self.l_click and self.r_click and self.l_pos and self.r_pos:
            self.current_action_label = "ZOOMING"
            # Distance between hands
            dist = math.hypot(self.l_pos[0] - self.r_pos[0], self.l_pos[1] - self.r_pos[1])
            
            if self.last_zoom_dist is None:
                self.last_zoom_dist = dist
            
            zoom_delta = (dist - self.last_zoom_dist) * 2.0
            self.engine.scene.world.world_scale = max(0.1, self.engine.scene.world.world_scale + zoom_delta)
            self.last_zoom_dist = dist
        else:
            self.last_zoom_dist = None

        # 3. BUILDING (Right Hand Pinch)
        if self.r_click and not self.l_click:
            self.current_action_label = "BUILDING"