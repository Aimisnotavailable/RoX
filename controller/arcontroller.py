# controller/arcontroller.py
import threading
import time
import glm
import cv2
import math
import numpy as np
from collections import deque
from settings import *
from controller.ar import AR
from controller.gesture_detector import (
    GestureManager, PinchDetector, TwoFingerUpDetector,
    OpenPalmDetector, PointDetector, FistDetector
)
from scripts.logger import get_logger_info

TOP_MENU = [
    {"name": "BLOCKS", "color": (200,200,200), "submenu": [
        {"name": "SAND",   "color": (230,210,180), "voxel_id": 1},
        {"name": "GRASS",  "color": (100,200,100), "voxel_id": 2},
        {"name": "DIRT",   "color": (140,100,70),  "voxel_id": 3},
        {"name": "STONE",  "color": (160,160,170), "voxel_id": 4},
        {"name": "SNOW",   "color": (240,240,255), "voxel_id": 5},
        {"name": "LEAVES", "color": (80,160,80),   "voxel_id": 6},
        {"name": "WOOD",   "color": (180,140,100), "voxel_id": 7},
        {"name": "BACK",   "color": (100,100,100), "action": "back"}
    ]},
    {"name": "GRAB SIZE", "color": (200,200,200), "submenu": [
        {"name": "SIZE 1", "color": (200,200,200), "size": 1},
        {"name": "SIZE 3", "color": (200,200,200), "size": 3},
        {"name": "SIZE 5", "color": (200,200,200), "size": 5},
        {"name": "BACK",   "color": (100,100,100), "action": "back"}
    ]},
]

class ARController:
    def __init__(self, engine):
        self.engine = engine
        self.ar = AR(WIN_RES)
        self.cap = cv2.VideoCapture(0)
        
        self.running = True
        
        # Gesture manager
        self.gesture_manager = GestureManager()
        self.gesture_manager.add_detector(PinchDetector('LEFT', hold_frames=30))
        self.gesture_manager.add_detector(PinchDetector('RIGHT', hold_frames=20))
        self.gesture_manager.add_detector(TwoFingerUpDetector('LEFT', hold_frames=None, require_others_down=True))
        self.gesture_manager.add_detector(TwoFingerUpDetector('RIGHT', hold_frames=None, require_others_down=True))
        self.gesture_manager.add_detector(OpenPalmDetector('LEFT', hold_frames=15))
        self.gesture_manager.add_detector(PointDetector('LEFT', hold_frames=None))
        self.gesture_manager.add_detector(PointDetector('RIGHT', hold_frames=None))
        self.gesture_manager.add_detector(FistDetector('RIGHT', hold_frames=None))
        
        # Raw data
        self._raw_left_landmarks = []
        self._raw_right_landmarks = []
        self._raw_left_pinch = False
        self._raw_right_pinch = False
        self._hand_type_left = "REAL"
        self._hand_type_right = "REAL"
        
        # Smoothed landmarks
        self.smooth_left_landmarks = []
        self.smooth_right_landmarks = []
        self.ema_alpha = 0.1
        
        self.smooth_left_pos = None  
        self.smooth_right_pos = None
        self.ar_mouse_pos = None
        self.ar_right_click = False
        
        # Pinch state
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
        
        # Radial menu
        self.radial_menu_active = False
        self.radial_menu_center = None
        
        # Zoom tracking
        self.last_zoom_dist = None

        # Grab state
        self.grab_size = 1
        self.is_grabbing = False
        self.grabbed_region = None          # list of (voxel_id, offset)
        self.grabbed_region_center = None   # original center (world pos, integer)
        self.grabbed_region_offset = None   # list of (dx,dy,dz) relative to center
        self.grabbed_region_current_pos = None  # for ghost rendering

        # Click detection (left hand point) – simple z‑delta from start to end
        self.left_point_active = False
        self.left_point_start_z = None
        self.CLICK_Z_DELTA_THRESHOLD = -0.05   # negative = toward camera
        self.CLICK_COOLDOWN = 0.3
        self.last_click_time = 0

        # Open palm anti‑reopen
        self._open_palm_active = False
        self._open_palm_used_to_close = False

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

    def get_world_pos_from_hand_point_fps(self, hand_landmarks):
        """
        Returns the integer world position of the block under the index fingertip,
        using FPS raycasting from the camera through the hand's screen position.
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return None

        tip_norm = hand_landmarks[8]
        screen_x = tip_norm.x * WIN_RES[0]
        screen_y = tip_norm.y * WIN_RES[1]

        # Get ray from camera through screen point
        camera = self.engine.player
        x = (2.0 * screen_x) / WIN_RES[0] - 1.0
        y = 1.0 - (2.0 * screen_y) / WIN_RES[1]
        ray_clip = glm.vec4(x, y, -1.0, 1.0)
        ray_eye = glm.inverse(camera.m_proj) * ray_clip
        ray_eye = glm.vec4(ray_eye.x, ray_eye.y, -1.0, 0.0)
        ray_world = glm.inverse(camera.m_view) * ray_eye
        ray_dir = glm.normalize(glm.vec3(ray_world))
        ray_origin = camera.position

        # Transform to local world space
        vh = self.engine.scene.world.voxel_handler
        inv_model = glm.inverse(self.engine.scene.world.m_model)
        local_origin = glm.vec3(inv_model * glm.vec4(ray_origin, 1.0))
        local_dir = glm.normalize(glm.vec3(inv_model * glm.vec4(ray_dir, 0.0)))

        max_dist = 10.0
        step = 0.05
        pos = local_origin
        for _ in range(int(max_dist / step)):
            pos += local_dir * step
            block_x = int(math.floor(pos.x))
            block_y = int(math.floor(pos.y))
            block_z = int(math.floor(pos.z))
            voxel_id, _, _, _ = vh.get_voxel_id((block_x, block_y, block_z))
            if voxel_id != 0:
                return glm.ivec3(block_x, block_y, block_z)
        return None

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

        # Process gestures
        events = self.gesture_manager.process_both(left_tuples, right_tuples, now)

        # Reset pinch flags
        self._raw_left_pinch = False
        self._raw_right_pinch = False

        # Handle events
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
                    elif ev.event_type == 'UPDATE' and ev.value is not None:
                        self._raw_left_pinch = True
                        self.pinch_current_left = self.smooth_left_pos
                    elif ev.event_type == 'HOLD':
                        self.pinch_hold_emitted['LEFT'] = True
                    elif ev.event_type == 'END':

                        if not self.pinch_hold_emitted['LEFT']:
                            get_logger_info('DEBUG', 'Left quick pinch – toggling mode')
                            self.engine.scene.world.voxel_handler.switch_mode()
                        self.pinch_active_left = False
                        self._raw_left_pinch = False
                        self.pinch_start_left = None
                elif ev.hand == 'RIGHT':
                    if self.is_grabbing:
                        continue
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
                        get_logger_info('AR', 'Right pinch HOLD')
                    elif ev.event_type == 'END':
                        if not self.pinch_hold_emitted['RIGHT']:
                            self.engine.scene.world.voxel_handler.set_voxel()
                        self.pinch_active_right = False
                        self._raw_right_pinch = False
                        self.pinch_start_right = None

            elif ev.gesture_name == 'two_finger_up':
                get_logger_info('DEBUG', f'Two-finger-up {ev.hand} {ev.event_type} value={ev.value}')
                if ev.hand == 'LEFT':
                    if ev.event_type == 'START' and ev.value is not None:
                        self.two_finger_up_left_active = True
                        self.two_finger_up_left_pos = ev.value
                    elif ev.event_type == 'UPDATE' and ev.value is not None and self.two_finger_up_left_pos is not None:
                        dx = ev.value[0] - self.two_finger_up_left_pos[0]
                        dy = ev.value[1] - self.two_finger_up_left_pos[1]
                        get_logger_info('DEBUG', f'Left two-finger-up delta: {dx:.3f}, {dy:.3f}')
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
                        get_logger_info('DEBUG', f'Right two-finger-up delta: {dx:.3f}, {dy:.3f}')
                        if self.engine.player.mode == "FPS":
                            self.engine.player.fps_camera.rotate_yaw(dx * 2.0)
                            self.engine.player.fps_camera.rotate_pitch(dy * 2.0)
                        self.two_finger_up_right_pos = ev.value
                    elif ev.event_type == 'END':
                        self.two_finger_up_right_active = False
                        self.two_finger_up_right_pos = None

            elif ev.gesture_name == 'open_palm' and ev.hand == 'LEFT':
                if ev.event_type == 'START':
                    self._open_palm_active = True
                elif ev.event_type == 'HOLD' and not self.radial_menu_active and not self._open_palm_used_to_close:
                    if self.smooth_left_pos is not None:
                        self.open_radial_menu(self.smooth_left_pos)
                elif ev.event_type == 'END':
                    self._open_palm_active = False
                    self._open_palm_used_to_close = False

            elif ev.gesture_name == 'point' and ev.hand == 'LEFT':
                # Left hand point: navigate menu and detect quick forward jab
                if ev.event_type == 'START' and ev.value is not None:
                    self.left_point_active = True
                    self.left_point_start_z = ev.value[2]
                elif ev.event_type == 'UPDATE' and ev.value is not None and self.radial_menu_active:
                    screen_x = ev.value[0] * WIN_RES[0]
                    screen_y = ev.value[1] * WIN_RES[1]
                    self.engine.scene.hud.radial_menu.update_selection((screen_x, screen_y))
                elif ev.event_type == 'END' and ev.value is not None and self.left_point_active:
                    delta_z = ev.value[2] - self.left_point_start_z
                    if delta_z < self.CLICK_Z_DELTA_THRESHOLD and (now - self.last_click_time) > self.CLICK_COOLDOWN:
                        self.last_click_time = now
                        if self.radial_menu_active:
                            self.execute_radial_selection()
                    self.left_point_active = False
                    self.left_point_start_z = None

            elif ev.gesture_name == 'point' and ev.hand == 'RIGHT':
                # Right point (reserved, but we can ignore)
                pass

            elif ev.gesture_name == 'fist' and ev.hand == 'RIGHT':
                # Only allow grabbing in FPS mode
                if self.engine.player.mode != "FPS":
                    continue

                if ev.event_type == 'START' and not self.is_grabbing:
                    # Get block under right index tip
                    center = self.get_world_pos_from_hand_point_fps(self.smooth_right_landmarks)
                    if center is None:
                        get_logger_info('DEBUG', 'No block under hand to grab')
                        continue
                    vh = self.engine.scene.world.voxel_handler
                    # Temporarily set vh.voxel_world_pos to center for _start_grab to use
                    old_pos = vh.voxel_world_pos
                    vh.voxel_world_pos = center
                    self._start_grab(vh)
                    vh.voxel_world_pos = old_pos
                elif ev.event_type == 'UPDATE' and self.is_grabbing:
                    # Update grabbed region position to follow right hand index tip
                    if self.smooth_right_landmarks and len(self.smooth_right_landmarks) > 8:
                        tip = self.smooth_right_landmarks[8]
                        self.grabbed_region_current_pos = glm.vec3(tip)
                    elif self.smooth_right_pos is not None:
                        self.grabbed_region_current_pos = glm.vec3(self.smooth_right_pos)
                elif ev.event_type == 'END' and self.is_grabbing:
                    self._end_grab()

        # Update smoothed positions
        self.smooth_left_pos = self.smooth_left_landmarks[8] if len(self.smooth_left_landmarks) > 8 else None
        self.smooth_right_pos = self.smooth_right_landmarks[8] if len(self.smooth_right_landmarks) > 8 else None

        # AR mouse for building
        if self.smooth_right_pos is not None:
            self.ar_mouse_pos = (self.smooth_right_pos.x * WIN_RES[0], self.smooth_right_pos.y * WIN_RES[1])
        else:
            self.ar_mouse_pos = None
        self.ar_right_click = self.pinch_active_right

        # Two‑hand pinch zoom
        if self.pinch_active_left and self.pinch_active_right and not self.is_grabbing:
            if self.radial_menu_active:
                self.close_radial_menu()
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

        # Clean up pinch state for hands that are completely absent
        if not self.smooth_left_landmarks and self._hand_type_left == "REAL":
            self.pinch_active_left = False
            self.pinch_hold_emitted['LEFT'] = False
            self.pinch_start_left = None
            self.pinch_current_left = None

        if not self.smooth_right_landmarks and self._hand_type_right == "REAL":
            self.pinch_active_right = False
            self.pinch_hold_emitted['RIGHT'] = False
            self.pinch_start_right = None
            self.pinch_current_right = None

        # Update ghost region
        if hasattr(self.engine.scene, 'ghost_region'):
            self.engine.scene.ghost_region.visible = self.is_grabbing
            if self.is_grabbing and self.grabbed_region_current_pos is not None:
                self.engine.scene.ghost_region.position = self.grabbed_region_current_pos
                self.engine.scene.ghost_region.size = self.grab_size

    def _start_grab(self, vh):
        """Initiate grab of a cubic region centered at vh.voxel_world_pos."""
        center = vh.voxel_world_pos
        if center is None:
            get_logger_info('DEBUG', 'Grab start failed: center is None')
            return
        half = self.grab_size // 2
        min_corner = (center[0] - half, center[1] - half, center[2] - half)
        max_corner = (center[0] + half, center[1] + half, center[2] + half)

        region = []
        offsets = []
        for x in range(min_corner[0], max_corner[0] + 1):
            for y in range(min_corner[1], max_corner[1] + 1):
                for z in range(min_corner[2], max_corner[2] + 1):
                    voxel_id, idx, local, chunk = vh.get_voxel_id((x, y, z))
                    if chunk is not None and voxel_id != 0:
                        region.append((voxel_id, (x, y, z), chunk, idx))
                        offsets.append((x - center[0], y - center[1], z - center[2]))

        if not region:
            get_logger_info('DEBUG', 'No blocks to grab')
            return

        # Remove the blocks
        for voxel_id, pos, chunk, idx in region:
            chunk.voxels[int(idx)] = 0
            chunk.mesh.rebuild()

        self.grabbed_region = [(voxel_id, offset) for (voxel_id, pos, chunk, idx), offset in zip(region, offsets)]
        self.grabbed_region_center = center
        self.grabbed_region_offset = offsets
        self.grabbed_region_current_pos = glm.vec3(center)
        self.is_grabbing = True

        get_logger_info('AR', f'Grabbed region of size {self.grab_size} with {len(region)} blocks')

    def _end_grab(self):
        """Place the grabbed region at the current position (snapped to grid)."""
        if not self.is_grabbing or self.grabbed_region_current_pos is None:
            return

        target_center = glm.round(self.grabbed_region_current_pos)
        half = self.grab_size // 2
        min_corner = (target_center[0] - half, target_center[1] - half, target_center[2] - half)

        vh = self.engine.scene.world.voxel_handler
        target_positions = []
        for offset in self.grabbed_region_offset:
            wx = int(min_corner[0] + offset[0])
            wy = int(min_corner[1] + offset[1])
            wz = int(min_corner[2] + offset[2])
            target_positions.append((wx, wy, wz))

        # Verify emptiness
        placeable = True
        chunks_to_rebuild = set()
        for (voxel_id, offset), (wx, wy, wz) in zip(self.grabbed_region, target_positions):
            target_id, idx, _, chunk = vh.get_voxel_id((wx, wy, wz))
            if target_id != 0:
                placeable = False
                break
            if chunk is not None:
                chunks_to_rebuild.add(chunk)

        if placeable:
            # Place all blocks
            for (voxel_id, offset), (wx, wy, wz) in zip(self.grabbed_region, target_positions):
                target_id, idx, _, chunk = vh.get_voxel_id((wx, wy, wz))
                if chunk is not None:
                    chunk.voxels[int(idx)] = voxel_id
                else:
                    get_logger_info('ERROR', f'Placement: chunk is None at ({wx},{wy},{wz})')
            for chunk in chunks_to_rebuild:
                if chunk:
                    chunk.mesh.rebuild()
            get_logger_info('AR', f'Placed grabbed region at {target_center}')
        else:
            # Restore original
            for (voxel_id, offset), (wx, wy, wz) in zip(self.grabbed_region, target_positions):
                ox = int(self.grabbed_region_center[0] + offset[0])
                oy = int(self.grabbed_region_center[1] + offset[1])
                oz = int(self.grabbed_region_center[2] + offset[2])
                target_id, idx, _, chunk = vh.get_voxel_id((ox, oy, oz))
                if chunk is not None:
                    chunk.voxels[int(idx)] = voxel_id
                    chunk.mesh.rebuild()
                else:
                    get_logger_info('ERROR', f'Restore: chunk is None at ({ox},{oy},{oz})')
            get_logger_info('AR', 'Cannot place – restored original')

        self.is_grabbing = False
        self.grabbed_region = None
        self.grabbed_region_center = None
        self.grabbed_region_offset = None
        self.grabbed_region_current_pos = None

    def open_radial_menu(self, hand_pos):
        if hand_pos is None:
            return
        screen_x = hand_pos.x * WIN_RES[0]
        screen_y = hand_pos.y * WIN_RES[1]
        self.radial_menu_active = True
        self.radial_menu_center = (screen_x, screen_y)
        self.engine.scene.hud.radial_menu.activate(self.radial_menu_center, TOP_MENU)
        get_logger_info('AR', f'Radial menu at ({screen_x:.0f}, {screen_y:.0f})')

    def close_radial_menu(self):
        if self.radial_menu_active:
            self.engine.scene.hud.radial_menu.deactivate()
            self.radial_menu_active = False
            get_logger_info('AR', 'Radial menu closed')

    def execute_radial_selection(self):
        selected = self.engine.scene.hud.radial_menu.selected_index
        if selected < 0:
            return
        option = self.engine.scene.hud.radial_menu.current_options[selected]
        if "submenu" in option:
            self.engine.scene.hud.radial_menu.push_submenu(option["submenu"])
        elif option.get("action") == "back":
            self.engine.scene.hud.radial_menu.pop_submenu()
        elif "voxel_id" in option:
            self.engine.scene.world.voxel_handler.new_voxel_id = option["voxel_id"]
            self.close_radial_menu()
        elif "size" in option:
            self.grab_size = option["size"]
            self.close_radial_menu()