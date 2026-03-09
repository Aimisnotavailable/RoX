# arcontroller.py
import threading
import cv2
import time
import glm
import numpy as np
from settings import *
from scripts.ar import AR

# Import your existing state manager!
from src.handstateaction import HandActionState 

class ARController:
    def __init__(self, engine):
        self.engine = engine
        self.ar = AR(WIN_RES)
        self.cap = cv2.VideoCapture(0)
        
        self.running = True
        
        # --- SMART HAND STATE MACHINES (Debouncing) ---
        self.left_state = HandActionState()
        self.right_state = HandActionState()
        
        # --- Raw Thread-Safe Data ---
        self._raw_left_landmarks = []
        self._raw_right_landmarks = []
        self._raw_left_pinch = False
        self._raw_right_pinch = False
        
        # --- Smoothed Engine Data ---
        self.smooth_left_landmarks = []
        self.smooth_right_landmarks = []
        self.ema_alpha = 0.45  # Smoothing factor
        
        # --- Public Engine Properties (HUD & Voxel Handler) ---
        self.smooth_left_pos = None  
        self.smooth_right_pos = None
        self.ar_mouse_pos = None
        self.ar_right_click = False
        
        # Gesture states
        self.last_rotate_pos = None
        self.last_zoom_dist = None

        get_logger_info('AR', f'THREAD FOR AR SYSTEM INITIALIZED')
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()

    def _tracking_loop(self):
        """Asynchronous daemon thread for computer vision."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1) # Mirror Mode
            
            ar_data = self.ar.update(frame)
            
            if ar_data is None:
                continue

            # Extract full 21 landmarks safely
            self._raw_left_landmarks = ar_data["POSITION_DATA"].get("LEFT", [])
            self._raw_right_landmarks = ar_data["POSITION_DATA"].get("RIGHT", [])
            
            # Extract raw, noisy pinch flags
            self._raw_left_pinch = ar_data["CLICK_FLAG"].get("LEFT", False)
            self._raw_right_pinch = ar_data["CLICK_FLAG"].get("RIGHT", False)

    def _apply_ema(self, current_smooth, raw_new):
        """Applies Exponential Moving Average to the 21-point skeleton."""
        if not raw_new or len(raw_new) < 21:
            return []
            
        if not current_smooth or len(current_smooth) < 21:
            return [glm.vec2(p[0], p[1]) for p in raw_new]
            
        smoothed = []
        for i in range(21):
            target = glm.vec2(raw_new[i][0], raw_new[i][1])
            new_pos = (target * self.ema_alpha) + (current_smooth[i] * (1.0 - self.ema_alpha))
            smoothed.append(new_pos)
            
        return smoothed

    def update(self):
        """Fast-tick loop running on the main Engine thread."""
        now = time.time()

        # 1. APPLY EMA TO SKELETONS (For HUD Rendering)
        self.smooth_left_landmarks = self._apply_ema(self.smooth_left_landmarks, self._raw_left_landmarks)
        self.smooth_right_landmarks = self._apply_ema(self.smooth_right_landmarks, self._raw_right_landmarks)

        # 2. EXTRACT SMOOTH INDEX TIPS
        l_index = self.smooth_left_landmarks[8] if len(self.smooth_left_landmarks) > 8 else None
        r_index = self.smooth_right_landmarks[8] if len(self.smooth_right_landmarks) > 8 else None

        l_pos_tuple = (l_index.x, l_index.y) if l_index else None
        r_pos_tuple = (r_index.x, r_index.y) if r_index else None

        # 3. UPDATE HAND STATE MACHINES (The Debounce Magic!)
        self.left_state.update(self._raw_left_pinch, l_pos_tuple, now)
        self.right_state.update(self._raw_right_pinch, r_pos_tuple, now)

        # 4. EXPOSE STABLE DATA TO ENGINE
        self.smooth_left_pos = l_index
        self.smooth_right_pos = r_index
        
        # Voxel Handler strictly respects the debounced 'active' state now
        self.ar_mouse_pos = self.right_state.current_pos 
        self.ar_right_click = self.right_state.active  

        # 5. GESTURE CONTROLS (World Rotation & Zoom)
        world = self.engine.scene.world
        
        # Use the highly stable debounced states for gestures
        l_active = self.left_state.active
        r_active = self.right_state.active

        # Double Pinch = Zoom
        if l_active and r_active and l_index and r_index:
            current_dist = glm.distance(l_index, r_index)
            if self.last_zoom_dist is not None:
                delta_zoom = (current_dist - self.last_zoom_dist) * 0.005
                world.world_scale += delta_zoom
                world.world_scale = max(0.1, min(10.0, world.world_scale)) # Clamp scale
            
            self.last_zoom_dist = current_dist
            self.last_rotate_pos = None # Interrupt rotation
            
        # Left Pinch Only = Rotate World
        elif l_active and not r_active and l_index:
            self.last_zoom_dist = None
            
            if self.last_rotate_pos is not None:
                delta_rot = l_index - self.last_rotate_pos
                world.world_yaw += delta_rot.x * 0.005
                world.world_pitch += delta_rot.y * 0.005
                
            self.last_rotate_pos = glm.vec2(l_index)
            
        else:
            self.last_zoom_dist = None
            self.last_rotate_pos = None