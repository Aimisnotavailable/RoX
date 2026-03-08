import threading
import cv2
import math
from settings import *
from scripts.ar import AR 

class ARController:
    def __init__(self, engine):
        self.engine = engine
        self.ar_system = AR(screen_dim=WIN_RES)
        self.cap = cv2.VideoCapture(0)
        
        # --- Smoothing Logic ---
        # Because we are now smoothing in the FAST engine loop, 
        # this alpha needs to be much lower (e.g., 0.05 to 0.1)
        self.smooth_alpha = 0.1 
        self.current_lms = {"LEFT": None, "RIGHT": None} # Tracks the display position
        
        # Shared data
        self.is_running = True
        self.current_action_label = "IDLE"
        self.ar_data = None 
        self.frame = None

        self.last_rotate_pos = None
        self.last_zoom_dist = None
        
        self.l_pos, self.r_pos = None, None
        self.l_click, self.r_click = False, False
        
        print('[AR] Starting Ghost-Frame Tracking Thread...')
        self.thread = threading.Thread(target=self._ar_loop, daemon=True)
        self.thread.start()

    def _ar_loop(self):
        """Threaded camera processing loop (Slow Tick ~30fps)."""
        while self.is_running:
            success, frame = self.cap.read()
            if not success: continue
            
            frame = cv2.flip(frame, 1)
            self.frame = frame 
            
            # Just dump the RAW target data. Do NOT smooth here.
            self.ar_data = self.ar_system.update(frame)

    def update(self):
        """Main Engine Update (Fast Tick ~60+fps): Smooths towards target data."""
        if not self.ar_data: 
            return
        
        data = self.ar_data
        self.current_action_label = "IDLE"

        # 1. Smooth interpolation in the FAST engine loop
        for label in ["LEFT", "RIGHT"]:
            target_lms = data["POSITION_DATA"][label]
            
            # If no hand is detected, drop the current tracking
            if not target_lms or len(target_lms) < 21:
                self.current_lms[label] = None
                continue

            # First time seeing the hand, snap to it
            if self.current_lms[label] is None:
                self.current_lms[label] = target_lms
            else:
                # Interpolate (Lerp) towards the target position every fast frame
                smoothed = []
                for i in range(21):
                    px = self.current_lms[label][i][0] + (target_lms[i][0] - self.current_lms[label][i][0]) * self.smooth_alpha
                    py = self.current_lms[label][i][1] + (target_lms[i][1] - self.current_lms[label][i][1]) * self.smooth_alpha
                    smoothed.append((px, py))
                self.current_lms[label] = smoothed

        l_lms = self.current_lms["LEFT"]
        r_lms = self.current_lms["RIGHT"]
        
        self.l_click = data["CLICK_FLAG"]["LEFT"]
        self.r_click = data["CLICK_FLAG"]["RIGHT"]
        
        self.l_pos = l_lms[8] if l_lms else None
        self.r_pos = r_lms[8] if r_lms else None

        # --- Update Engine Interface (INPUT FALLBACK LOGIC) ---
        if self.r_pos:
            self.engine.ar_right_click = self.r_click
            self.engine.ar_mouse_pos = (self.r_pos[0] * WIN_RES[0], self.r_pos[1] * WIN_RES[1])
        else:
            # Explicitly turn off AR inputs if right hand is not visible
            self.engine.ar_right_click = False
            self.engine.ar_mouse_pos = None

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