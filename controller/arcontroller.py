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
        # Add detectors
        # In __init__:
        self.gesture_manager.add_detector(PinchDetector('LEFT', hold_frames=15))   # 0.5s at 30fps
        self.gesture_manager.add_detector(PinchDetector('RIGHT', hold_frames=5))
        self.gesture_manager.add_detector(TwoFingerUpDetector('LEFT', hold_frames=None, require_others_down=True))
        self.gesture_manager.add_detector(TwoFingerUpDetector('RIGHT', hold_frames=None, require_others_down=True))
        
        # Raw data from AR thread
        self._raw_left_landmarks = []
        self._raw_right_landmarks = []
        self._raw_left_pinch = False
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
        
        # Pinch state tracking
        self.pinch_active_left = False
        self.pinch_active_right = False
        self.pinch_hold_emitted = {'LEFT': False, 'RIGHT': False}
        self.pinch_start_left = None
        self.pinch_start_right = None
        self.pinch_current_left = None
        self.pinch_current_right = None
        self.last_mode_toggle_time = 0

        # Two-finger-up state
        self.two_finger_up_left_active = False
        self.two_finger_up_left_pos = None
        self.two_finger_up_right_active = False
        self.two_finger_up_right_pos = None
        
        # Radial menu (now on left hand)
        self.radial_menu_active = False
        self.radial_menu_center = None
        
        # Legacy zoom tracking (two-hand pinch)
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

        # Smooth landmarks
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

        # Reset pinch flags (will be set by events)
        self._raw_left_pinch = False
        self._raw_right_pinch = False

        # If a hand is ghost, force its pinch state to False
        if self._hand_type_left != "REAL":
            self.pinch_active_left = False
            self.pinch_hold_emitted['LEFT'] = False
            self.pinch_start_left = None
            self.pinch_current_left = None
        if self._hand_type_right != "REAL":
            self.pinch_active_right = False
            self.pinch_hold_emitted['RIGHT'] = False
            self.pinch_start_right = None
            self.pinch_current_right = None

        # Handle gesture events
        for ev in events:
            if ev is None:
                continue

            if ev.gesture_name == 'pinch':
                if ev.hand == 'LEFT':
                    if ev.event_type == 'START':
                        self.pinch_active_left = True
                        self._raw_left_pinch = True
                        self.pinch_hold_emitted['LEFT'] = False
                        self.pinch_start_left = self.smooth_left_pos
                        # Close any existing menu if we start a new pinch
                        if self.radial_menu_active:
                            self.close_radial_menu()
                    elif ev.event_type == 'UPDATE' and ev.value is not None:
                        self._raw_left_pinch = True
                        self.pinch_current_left = self.smooth_left_pos
                    elif ev.event_type == 'HOLD':
                        # Hold threshold reached – open radial menu immediately
                        self.pinch_hold_emitted['LEFT'] = True
                        get_logger_info('AR', 'Left pinch HOLD detected – opening menu')
                        self.open_radial_menu(self.smooth_left_pos)
                    elif ev.event_type == 'END':
                        if self.radial_menu_active:
                            get_logger_info('DEBUG', 'Left pinch END with menu active – executing selection')
                            self.execute_radial_selection()
                        else:
                            get_logger_info('DEBUG', f'Left pinch END, hold_emitted={self.pinch_hold_emitted["LEFT"]}')
                            if not self.pinch_hold_emitted['LEFT']:
                                get_logger_info('DEBUG', 'Left quick pinch – toggling mode')
                                self.engine.scene.world.voxel_handler.switch_mode()
                        self.pinch_active_left = False
                        self._raw_left_pinch = False
                        self.pinch_start_left = None
                        # Do not reset radial_menu_active here – it is reset in execute or close
                elif ev.hand == 'RIGHT':
                    if ev.event_type == 'START':
                        self.pinch_active_right = True
                        self._raw_right_pinch = True
                        self.pinch_hold_emitted['RIGHT'] = False
                        self.pinch_start_right = self.smooth_right_pos
                    elif ev.event_type == 'UPDATE' and ev.value is not None:
                        self._raw_right_pinch = True
                        self.pinch_current_right = self.smooth_right_pos
                    elif ev.event_type == 'HOLD':
                        self.pinch_hold_emitted['RIGHT'] = True
                        get_logger_info('AR', 'Right pinch HOLD detected')
                        # Optional: could do something else
                    elif ev.event_type == 'END':
                        # Quick pinch if no hold
                        if not self.pinch_hold_emitted['RIGHT']:
                            self.engine.scene.world.voxel_handler.set_voxel()
                        self.pinch_active_right = False
                        self._raw_right_pinch = False
                        self.pinch_start_right = None

            elif ev.gesture_name == 'two_finger_up':
                if ev.hand == 'LEFT':
                    if ev.event_type == 'START' and ev.value is not None:
                        self.two_finger_up_left_active = True
                        self.two_finger_up_left_pos = ev.value
                    elif ev.event_type == 'UPDATE' and ev.value is not None and self.two_finger_up_left_pos is not None:
                        dx = ev.value[0] - self.two_finger_up_left_pos[0]
                        dy = ev.value[1] - self.two_finger_up_left_pos[1]
                        world = self.engine.scene.world
                        world.world_yaw += dx * 2.0
                        world.world_pitch += dy * 2.0
                        world.world_pitch = max(-1.5, min(1.5, world.world_pitch))
                        self.two_finger_up_left_pos = ev.value
                    elif ev.event_type == 'END':
                        self.two_finger_up_left_active = False
                        self.two_finger_up_left_pos = None
                elif ev.hand == 'RIGHT':
                    if ev.event_type == 'START' and ev.value is not None:
                        self.two_finger_up_right_active = True
                        self.two_finger_up_right_pos = ev.value
                    elif ev.event_type == 'UPDATE' and ev.value is not None and self.two_finger_up_right_pos is not None:
                        dx = ev.value[0] - self.two_finger_up_right_pos[0]
                        dy = ev.value[1] - self.two_finger_up_right_pos[1]
                        if self.engine.player.mode == "FPS":
                            self.engine.player.fps_camera.rotate_yaw(dx * 2.0)
                            self.engine.player.fps_camera.rotate_pitch(dy * 2.0)
                        self.two_finger_up_right_pos = ev.value
                    elif ev.event_type == 'END':
                        self.two_finger_up_right_active = False
                        self.two_finger_up_right_pos = None

        # Update smoothed positions (index tip)
        self.smooth_left_pos = self.smooth_left_landmarks[8] if len(self.smooth_left_landmarks) > 8 else None
        self.smooth_right_pos = self.smooth_right_landmarks[8] if len(self.smooth_right_landmarks) > 8 else None

        # Set AR mouse position and click for building (right hand)
        if self.smooth_right_pos is not None:
            self.ar_mouse_pos = (self.smooth_right_pos.x * WIN_RES[0], self.smooth_right_pos.y * WIN_RES[1])
        else:
            self.ar_mouse_pos = None
        self.ar_right_click = self.pinch_active_right

        # --- Two‑hand pinch: zoom (highest priority) ---
        # If both hands are pinched, override any other gesture that might conflict.
        if self.pinch_active_left and self.pinch_active_right:
            # If menu was open, close it
            if self.radial_menu_active:
                self.close_radial_menu()
            # Compute zoom based on distance between hands
            if self.smooth_left_pos is not None and self.smooth_right_pos is not None:
                l_pixel = (self.smooth_left_pos.x * WIN_RES[0], self.smooth_left_pos.y * WIN_RES[1])
                r_pixel = (self.smooth_right_pos.x * WIN_RES[0], self.smooth_right_pos.y * WIN_RES[1])
                current_dist = math.hypot(l_pixel[0] - r_pixel[0], l_pixel[1] - r_pixel[1])
                if self.last_zoom_dist is not None:
                    delta_zoom = (current_dist - self.last_zoom_dist) * 0.005
                    self.engine.scene.world.world_scale += delta_zoom
                    self.engine.scene.world.world_scale = max(0.1, min(10.0, self.engine.scene.world.world_scale))
                self.last_zoom_dist = current_dist
            else:
                self.last_zoom_dist = None
        else:
            self.last_zoom_dist = None

    def open_radial_menu(self, hand_pos):
        """Activate radial menu at hand position (left hand)."""
        if hand_pos is None:
            return
        screen_x = hand_pos.x * WIN_RES[0]
        screen_y = hand_pos.y * WIN_RES[1]
        self.radial_menu_active = True
        self.radial_menu_center = (screen_x, screen_y)
        self.engine.scene.hud.radial_menu.activate(self.radial_menu_center)
        get_logger_info('AR', f'Radial menu opened at ({screen_x:.0f}, {screen_y:.0f})')

    def close_radial_menu(self):
        """Force close the radial menu without selection."""
        if self.radial_menu_active:
            self.engine.scene.hud.radial_menu.deactivate()
            self.radial_menu_active = False
            get_logger_info('AR', 'Radial menu closed')

    def execute_radial_selection(self):
        selected = self.engine.scene.hud.radial_menu.selected_index
        if selected < 0:
            get_logger_info('AR', 'Radial menu closed with no selection')
            self.close_radial_menu()
            return
        option = self.engine.scene.hud.radial_menu.options[selected]
        voxel_id = option["voxel_id"]
        get_logger_info('AR', f'Radial menu selected: {option["name"]} (ID {voxel_id})')
        self.engine.scene.world.voxel_handler.new_voxel_id = voxel_id
        self.close_radial_menu()