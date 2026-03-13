# scripts/ar.py
from scripts.arconfig import *
import math
import cv2
import pygame
import numpy as np
import mediapipe as mp
from collections import deque, namedtuple

# --- PINCH DETECTOR (unchanged, uses 2D) ---
HandState = namedtuple("HandState", ["pos_hist", "pinch_count", "is_pinched"])

class PinchDetector:
    def __init__(self):
        self.hands = {
            "LEFT":  HandState(deque(maxlen=HISTOGRAM_SIZE), 0, False),
            "RIGHT": HandState(deque(maxlen=HISTOGRAM_SIZE), 0, False),
        }

    @staticmethod
    def euclid(a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.hypot(dx, dy)

    def update(self, label, landmarks_norm):
        state = self.hands[label]
        state.pos_hist.append(landmarks_norm)

        p_thumb = landmarks_norm[THUMB_TIP_IDX]
        p_index = landmarks_norm[INDEX_TIP_IDX]
        p_wrist = landmarks_norm[WRIST_IDX]
        p_mcp   = landmarks_norm[MIDDLE_MCP_IDX]

        raw_dist   = self.euclid(p_thumb, p_index)
        hand_scale = self.euclid(p_wrist, p_mcp)
        rel_dist   = raw_dist / hand_scale if hand_scale > 0 else float('inf')

        pc, pinched = state.pinch_count, state.is_pinched

        if pinched:
            if rel_dist > PINCH_OFF_THRESH:
                pc -= 1
        else:
            if rel_dist < PINCH_ON_THRESH:
                pc += 1

        pc = max(-PINCH_FRAMES_REQ, min(PINCH_FRAMES_REQ, pc))

        if not pinched and pc >= PINCH_FRAMES_REQ:
            pinched = True
        elif pinched and pc <= -PINCH_FRAMES_REQ:
            pinched = False

        self.hands[label] = HandState(state.pos_hist, pc, pinched)

        return {
            "raw_dist":  raw_dist,
            "scale":     hand_scale,
            "rel_dist":  rel_dist,
            "is_pinched": pinched
        }

    def reset(self, label):
        if label in self.hands:
            state = self.hands[label]
            self.hands[label] = HandState(state.pos_hist, 0, False)


# --- AR CLASS (3D) ---
class AR:
    SOURCE_REAL = "real"
    SOURCE_GHOST = "ghost"

    def __init__(self, screen_dim=(1280, 720)):
        self.W = screen_dim[0]
        self.H = screen_dim[1]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7
        )

        # Each entry in the histogram:
        #   "pts": list of (x_norm, y_norm, z_norm) for 21 landmarks
        #   "source": "real" or "ghost"
        #   "frame": self.frame_count
        #   "gen_frame": for ghosts, the frame they were generated (for aging)
        self.position_histogram = {'LEFT': [], 'RIGHT': []}
        self.hands_tracker     = {'LEFT': 0,      'RIGHT': 0     }
        self.presence_counter = {'LEFT': 0, 'RIGHT': 0}
        self.presence_threshold_on = 2   
        self.presence_threshold_off = -2 
        self.pinch_absent_reset = max(3, HISTOGRAM_SIZE // 2)
        
        # --- GHOST KINEMATICS & TTL ---
        # Velocity is a 6‑element list: [vx, vy, vz, ax, ay, az]
        # where (ax, ay, az) is rotation axis * angle per frame.
        self.ghost_velocity = {'LEFT': [0.0]*6, 'RIGHT': [0.0]*6}
        self.KINEMATIC_FRICTION = 0.85 
        self.MAX_GHOST_TTL = 15  
        self.absent_reset_threshold = max(HISTOGRAM_SIZE, self.MAX_GHOST_TTL + 1) 
        self.ghost_ttl_counter = {'LEFT': 0, 'RIGHT': 0}
        self.ghost_age_default = 2
        
        # --- OPTICAL FLOW (still 2D pixel tracking) ---
        self.prev_gray = None
        self.lk_params = dict(
            winSize=(25, 25), 
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # Stores pixel coordinates of wrist and middle MCP
        self.lk_tracked_points = {'LEFT': None, 'RIGHT': None}

        self.detector = PinchDetector()
        self.frame_count = 0

        # ar_data now contains 3D points (normalized)
        self.ar_data = {
            "POSITION_DATA": {"LEFT": [], "RIGHT": []},   # list of (x,y,z)
            "SCALE":         {"LEFT": 1,    "RIGHT": 1},
            "FRAME_TYPE":    {"LEFT" : "REAL", "RIGHT" : "REAL"},
            "CLICK_DIST":    {"LEFT": 0,    "RIGHT": 0},
            "CLICK_FLAG":    {"LEFT": False,"RIGHT": False},
            "HAND_PRESENCE" : False
        }
        self.image = None
        self.debug = True

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def _log(self, level, msg, force=False):
        if not self.debug and not force:
            return
        get_logger_info(level, msg)

    # ------------------------------------------------------------------
    # Landmark validation (now checks z as well)
    # ------------------------------------------------------------------
    def _valid_landmark(self, lm):
        try:
            x = float(lm.x); y = float(lm.y); z = float(lm.z)
        except Exception:
            return False
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False
        return True

    # ------------------------------------------------------------------
    # Core function: convert landmarks to normalized 3D tuples and store
    # ------------------------------------------------------------------
    def calculate_hand_points(self, landmarks, label, is_generated=False):
        pts = []
        raw_landmarks = list(landmarks.landmark)

        for lm in raw_landmarks:
            try:
                lx = float(lm.x)
                ly = float(lm.y)
                lz = float(lm.z)
            except Exception:
                pts.append((0.0, 0.0, 0.0))
                continue

            # For both real and ghost, keep normalized coordinates.
            pts.append((lx, ly, lz))

        entry = {
            "pts": pts,
            "source": self.SOURCE_GHOST if is_generated else self.SOURCE_REAL,
            "frame": self.frame_count
        }
        
        if is_generated:
            gen_frame = getattr(landmarks, "_meta_gen_frame", None)
            entry["gen_frame"] = gen_frame if gen_frame is not None else self.frame_count

        hist = self.position_histogram[label]

        if not is_generated:
            if self.hands_tracker[label] >= self.absent_reset_threshold:
                hist.clear()
                hist.append(entry)
            else:
                # Remove any trailing ghosts before appending real data
                while len(hist) > 0 and hist[-1].get("source") == self.SOURCE_GHOST:
                    hist.pop()
                if len(pts) >= 1:
                    if len(hist) < HISTOGRAM_SIZE:
                        hist.append(entry)
                    else:
                        hist.pop(0)
                        hist.append(entry)
                    self.presence_counter[label] = min(self.presence_threshold_on, self.presence_counter[label] + 1)
                else:
                    self.presence_counter[label] = max(self.presence_threshold_off, self.presence_counter[label] - 1)
            self.hands_tracker[label] = 0
        else:
            if len(hist) < HISTOGRAM_SIZE:
                hist.append(entry)
            else:
                hist.pop(0)
                hist.append(entry)
        
        return pts

    # ------------------------------------------------------------------
    # Ghost pruning by age
    # ------------------------------------------------------------------
    def _prune_ghosts_by_age(self, label):
        hist = self.position_histogram[label]
        changed = False

        for e in hist[:]:
            if e.get("source") == self.SOURCE_GHOST:
                gen_frame = e.get("gen_frame", e.get("frame", None))
                if gen_frame is None:
                    gen_frame = e.get("frame", self.frame_count)
                age = self.frame_count - int(gen_frame)
                if age >= int(self.ghost_age_default):
                    try:
                        hist.remove(e)
                        changed = True
                    except ValueError:
                        pass

        while len(hist) > HISTOGRAM_SIZE:
            hist.pop(0)
            changed = True
        return changed

    # ------------------------------------------------------------------
    # 2D angle helpers (used by optical flow)
    # ------------------------------------------------------------------
    def _angle_between(self, a, b):
        return math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])

    def _normalize_angle(self, ang):
        while ang <= -math.pi:
            ang += 2 * math.pi
        while ang > math.pi:
            ang -= 2 * math.pi
        return ang

    # ------------------------------------------------------------------
    # 3D geometry helpers
    # ------------------------------------------------------------------
    def _normalize(self, v):
        length = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if length < 1e-6:
            return (0.0, 0.0, 0.0)
        return (v[0]/length, v[1]/length, v[2]/length)

    def _cross(self, a, b):
        return (a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0])

    def _dot(self, a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def _rotate_point(self, p, center, axis_angle):
        """
        Rotate point p around center by rotation given as axis * angle.
        axis_angle is a 3-tuple (ax, ay, az) where length = angle (radians).
        Uses Rodrigues' rotation formula.
        """
        angle = math.sqrt(axis_angle[0]**2 + axis_angle[1]**2 + axis_angle[2]**2)
        if angle < 1e-6:
            return p
        ax = axis_angle[0] / angle
        ay = axis_angle[1] / angle
        az = axis_angle[2] / angle

        # Translate point relative to center
        px = p[0] - center[0]
        py = p[1] - center[1]
        pz = p[2] - center[2]

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dot = ax*px + ay*py + az*pz
        cross_x = ay*pz - az*py
        cross_y = az*px - ax*pz
        cross_z = ax*py - ay*px

        rx = px*cos_a + cross_x*sin_a + ax*dot*(1 - cos_a)
        ry = py*cos_a + cross_y*sin_a + ay*dot*(1 - cos_a)
        rz = pz*cos_a + cross_z*sin_a + az*dot*(1 - cos_a)

        return (rx + center[0], ry + center[1], rz + center[2])

    def _compute_hand_orientation(self, pts):
        """
        Compute orientation features from hand landmarks.
        Returns:
          - hand_dir: unit vector from wrist to middle MCP (main hand direction)
          - hand_normal: approximate palm normal (using wrist, index MCP, pinky MCP)
        """
        wrist = pts[WRIST_IDX]
        mcp_mid = pts[MIDDLE_MCP_IDX]
        hand_dir = (mcp_mid[0]-wrist[0], mcp_mid[1]-wrist[1], mcp_mid[2]-wrist[2])
        hand_dir = self._normalize(hand_dir)

        idx_mcp = pts[INDEX_MCP_IDX]
        pinky_mcp = pts[PINKY_MCP_IDX]
        v1 = (idx_mcp[0]-wrist[0], idx_mcp[1]-wrist[1], idx_mcp[2]-wrist[2])
        v2 = (pinky_mcp[0]-wrist[0], pinky_mcp[1]-wrist[1], pinky_mcp[2]-wrist[2])
        normal = self._cross(v1, v2)
        norm_len = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if norm_len > 0:
            normal = (normal[0]/norm_len, normal[1]/norm_len, normal[2]/norm_len)
        else:
            normal = (0.0, 1.0, 0.0)  # fallback
        return hand_dir, normal

    # ------------------------------------------------------------------
    # 3D velocity estimation
    # ------------------------------------------------------------------
    def calculate_velocity(self, label, dir=0, window=4):
        """
        Compute 3D linear velocity (dx, dy, dz) and angular velocity as axis*angle.
        Returns a 6-element list [vx, vy, vz, ax, ay, az] if dir==1,
        otherwise returns linear speed as a float.
        """
        hist = self.position_histogram[label]
        valid = []  # list of (wrist_pt, hand_dir, hand_normal, frame)
        frames = []
        for e in reversed(hist):
            if e.get("source") != self.SOURCE_REAL:
                continue
            pts = e.get("pts", [])
            if isinstance(pts, list) and len(pts) > max(WRIST_IDX, MIDDLE_MCP_IDX, INDEX_MCP_IDX, PINKY_MCP_IDX):
                wrist = pts[WRIST_IDX]
                if wrist:
                    hand_dir, normal = self._compute_hand_orientation(pts)
                    valid.append((wrist, hand_dir, normal))
                    frames.append(int(e.get("frame", self.frame_count)))
                    if len(valid) >= window:
                        break

        if len(valid) < 2:
            return [0.0]*6 if dir else 0.0

        lin_vels = []
        ang_vels = []  # each element is a 3-tuple (axis*angle)

        for i in range(len(valid) - 1):
            (w_new, d_new, n_new) = valid[i]
            (w_old, d_old, n_old) = valid[i + 1]
            f_new = frames[i]; f_old = frames[i + 1]
            df = max(1, f_new - f_old)

            # Linear velocity
            vx = (w_new[0] - w_old[0]) / df
            vy = (w_new[1] - w_old[1]) / df
            vz = (w_new[2] - w_old[2]) / df
            lin_vels.append((vx, vy, vz))

            # Angular velocity from direction change
            axis1 = self._cross(d_old, d_new)
            sin_angle1 = math.sqrt(axis1[0]**2 + axis1[1]**2 + axis1[2]**2)
            cos_angle1 = self._dot(d_old, d_new)
            angle1 = math.atan2(sin_angle1, cos_angle1)
            if sin_angle1 > 0:
                axis1 = (axis1[0]/sin_angle1, axis1[1]/sin_angle1, axis1[2]/sin_angle1)
            else:
                axis1 = (0.0, 1.0, 0.0)  # arbitrary

            # Angular velocity from normal change
            axis2 = self._cross(n_old, n_new)
            sin_angle2 = math.sqrt(axis2[0]**2 + axis2[1]**2 + axis2[2]**2)
            cos_angle2 = self._dot(n_old, n_new)
            angle2 = math.atan2(sin_angle2, cos_angle2)
            if sin_angle2 > 0:
                axis2 = (axis2[0]/sin_angle2, axis2[1]/sin_angle2, axis2[2]/sin_angle2)
            else:
                axis2 = axis1

            # Combine (simple average of axis*angle)
            ang_vel = ((axis1[0]*angle1 + axis2[0]*angle2) * 0.5,
                       (axis1[1]*angle1 + axis2[1]*angle2) * 0.5,
                       (axis1[2]*angle1 + axis2[2]*angle2) * 0.5)
            ang_vels.append(ang_vel)

        avg_vx = sum(v[0] for v in lin_vels) / len(lin_vels)
        avg_vy = sum(v[1] for v in lin_vels) / len(lin_vels)
        avg_vz = sum(v[2] for v in lin_vels) / len(lin_vels)

        avg_ax = sum(a[0] for a in ang_vels) / len(ang_vels)
        avg_ay = sum(a[1] for a in ang_vels) / len(ang_vels)
        avg_az = sum(a[2] for a in ang_vels) / len(ang_vels)

        # Clamp angular speed
        MAX_ANGULAR = 0.5  # radians per frame
        ang_speed = math.sqrt(avg_ax**2 + avg_ay**2 + avg_az**2)
        if ang_speed > MAX_ANGULAR:
            scale = MAX_ANGULAR / ang_speed
            avg_ax *= scale
            avg_ay *= scale
            avg_az *= scale

        # Clamp linear speed (in normalized units per frame)
        MAX_SPEED_NORM = 0.1
        lin_speed = math.sqrt(avg_vx**2 + avg_vy**2 + avg_vz**2)
        if lin_speed > MAX_SPEED_NORM:
            scale = MAX_SPEED_NORM / lin_speed
            avg_vx *= scale
            avg_vy *= scale
            avg_vz *= scale

        if dir:
            return [avg_vx, avg_vy, avg_vz, avg_ax, avg_ay, avg_az]
        return lin_speed

    # ------------------------------------------------------------------
    # Last known real wrist (normalized)
    # ------------------------------------------------------------------
    def _last_real_wrist(self, label):
        hist = self.position_histogram.get(label, [])
        for e in reversed(hist):
            if e.get("source") == self.SOURCE_REAL:
                pts = e.get("pts", [])
                if isinstance(pts, list) and len(pts) > WRIST_IDX:
                    w = pts[WRIST_IDX]
                    if w:
                        return w
        return None

    # ------------------------------------------------------------------
    # Ghost generation helpers
    # ------------------------------------------------------------------
    def _normalize_ghost_genframes(self, label):
        hist = self.position_histogram.get(label, [])
        for e in hist:
            if e.get("source") == self.SOURCE_GHOST:
                if e.get("gen_frame") is None:
                    e["gen_frame"] = e.get("frame", self.frame_count)
                if e["gen_frame"] > self.frame_count:
                    e["gen_frame"] = self.frame_count

    def _swap_hand_state(self, a_label, b_label):
        # Swap all tracking data
        self.position_histogram[a_label], self.position_histogram[b_label] = \
            self.position_histogram[b_label], self.position_histogram[a_label]
        self.hands_tracker[a_label], self.hands_tracker[b_label] = \
            self.hands_tracker[b_label], self.hands_tracker[a_label]
        self.presence_counter[a_label], self.presence_counter[b_label] = \
            self.presence_counter[b_label], self.presence_counter[a_label]
        
        self.ghost_ttl_counter[a_label], self.ghost_ttl_counter[b_label] = \
            self.ghost_ttl_counter[b_label], self.ghost_ttl_counter[a_label]
        self.ghost_velocity[a_label], self.ghost_velocity[b_label] = \
            self.ghost_velocity[b_label], self.ghost_velocity[a_label]
            
        self.lk_tracked_points[a_label], self.lk_tracked_points[b_label] = \
            self.lk_tracked_points[b_label], self.lk_tracked_points[a_label]

        if a_label in self.detector.hands and b_label in self.detector.hands:
            ha = self.detector.hands[a_label]
            hb = self.detector.hands[b_label]
            self.detector.hands[a_label], self.detector.hands[b_label] = hb, ha
        else:
            for L in (a_label, b_label):
                if L not in self.detector.hands:
                    self.detector.hands[L] = HandState(deque(maxlen=HISTOGRAM_SIZE), 0, False)
        
        self._normalize_ghost_genframes(a_label)
        self._normalize_ghost_genframes(b_label)

    def _get_continuous_streak(self, label):
        hist = self.position_histogram.get(label, [])
        if not hist: 
            return 0
        
        streak = 1
        last_f = hist[-1].get("frame", -1)
        for i in range(len(hist)-2, -1, -1):
            f = hist[i].get("frame", -1)
            if last_f - f > 1:
                break
            streak += 1
            last_f = f
        return streak

    # ------------------------------------------------------------------
    # Handedness reconciliation (unchanged, uses pixel coordinates)
    # ------------------------------------------------------------------
    def _reconcile_handedness(self, detections):
        if not detections:
            return

        blobs = [(i, d[0], d[1], d[2]) for i, d in enumerate(detections) if d[1] is not None]
        if not blobs:
            return

        tracked = {
            "LEFT": self._last_real_wrist("LEFT"),
            "RIGHT": self._last_real_wrist("RIGHT")
        }
        # Convert tracked wrists to pixel coordinates for distance calculation
        tracked_px = {}
        for label, pos_norm in tracked.items():
            if pos_norm is not None:
                tracked_px[label] = (pos_norm[0] * self.W, pos_norm[1] * self.H)
            else:
                tracked_px[label] = None

        MAX_JUMP = max(self.W, self.H) * 0.10 
        assigned_blobs = set()
        assigned_labels = set()
        final_assignments = {}

        pairs = []
        for b_idx, orig_label, wrist_px, _ in blobs:
            for label, last_pos in tracked_px.items():
                if last_pos is not None:
                    dist = math.hypot(wrist_px[0] - last_pos[0], wrist_px[1] - last_pos[1])
                    if orig_label != label:
                        track_length = len(self.position_histogram[label])
                        frames_absent = self.hands_tracker[label]
                        if frames_absent > 2 or track_length < 5:
                            dist += (MAX_JUMP * 2.0)
                        else:
                            dist += (MAX_JUMP * 0.6)
                    if dist < MAX_JUMP:
                        pairs.append((dist, b_idx, label))
        
        pairs.sort(key=lambda x: x[0])
        for dist, b_idx, label in pairs:
            if b_idx not in assigned_blobs and label not in assigned_labels:
                final_assignments[b_idx] = label
                assigned_blobs.add(b_idx)
                assigned_labels.add(label)

        unassigned_blobs = [b for b in blobs if b[0] not in assigned_blobs]
        
        # Duplicate hand filter
        filtered_unassigned = []
        for b in unassigned_blobs:
            b_idx, orig_label, wrist_px, _ = b
            is_dup = False
            for a_idx, a_label in final_assignments.items():
                a_wrist = detections[a_idx][1]
                if math.hypot(wrist_px[0] - a_wrist[0], wrist_px[1] - a_wrist[1]) < (MAX_JUMP * 0.5):
                    is_dup = True
                    break
            if not is_dup:
                for a in filtered_unassigned:
                    a_wrist = a[2]
                    if math.hypot(wrist_px[0] - a_wrist[0], wrist_px[1] - a_wrist[1]) < (MAX_JUMP * 0.5):
                        is_dup = True
                        break
            if is_dup:
                self._log('CORE', "DROPPED: MediaPipe hallucinated an overlapping duplicate hand.", True)
            else:
                filtered_unassigned.append(b)
        unassigned_blobs = filtered_unassigned

        # Fallback assignment
        if len(unassigned_blobs) == 2 and len(assigned_labels) == 0:
            label_0 = unassigned_blobs[0][1] 
            label_1 = unassigned_blobs[1][1]
            if label_0 != label_1 and label_0 in ["LEFT", "RIGHT"] and label_1 in ["LEFT", "RIGHT"]:
                final_assignments[unassigned_blobs[0][0]] = label_0
                final_assignments[unassigned_blobs[1][0]] = label_1
                assigned_labels.update([label_0, label_1])
            else:
                self._log('CORE', "MediaPipe duplicate labels detected. Falling back to Spatial Partitioning.", True)
                unassigned_blobs.sort(key=lambda b: b[2][0])
                final_assignments[unassigned_blobs[0][0]] = "LEFT"
                final_assignments[unassigned_blobs[1][0]] = "RIGHT"
                assigned_labels.update(["LEFT", "RIGHT"])
        else:
            for b_idx, orig_label, wrist_px, _ in unassigned_blobs:
                avail_labels = [L for L in ["LEFT", "RIGHT"] if L not in assigned_labels]
                if not avail_labels:
                    break 
                if orig_label in avail_labels:
                    label = orig_label
                elif len(avail_labels) == 1:
                    label = avail_labels[0]
                else:
                    label = "LEFT" if wrist_px[0] < self.W / 2 else "RIGHT"
                final_assignments[b_idx] = label
                assigned_labels.add(label)
                assigned_blobs.add(b_idx)

        valid_detections = []
        for i, (orig_label, wrist_px, lm_set) in enumerate(detections):
            if i in final_assignments:
                new_label = final_assignments[i]
                if new_label != orig_label:
                    self._log('CORE', f"ZERO-TRUST OVERRIDE: Forced MediaPipe's {orig_label} to become {new_label}", True)
                valid_detections.append((new_label, wrist_px, lm_set))
        detections[:] = valid_detections

    # ------------------------------------------------------------------
    # Ghost frame generator (3D)
    # ------------------------------------------------------------------
    def generate_frames(self, velocity, label):
        hist = self.position_histogram[label]
        
        if len(hist) > 0 :
            base_entry = hist[-1]
            base_pts = base_entry.get("pts", []) or []
        else:
            base_pts = []

        class LM:
            def __init__(self, x, y, z):
                self.x = float(x); self.y = float(y); self.z = float(z)
        class HL:
            def __init__(self):
                self.landmark = []
            def add(self, x, y, z):
                self.landmark.append(LM(x, y, z))

        gen = HL()

        vx, vy, vz, ax, ay, az = velocity
        # Limit rotation per frame
        MAX_ANGLE = 0.35
        ang_speed = math.sqrt(ax**2 + ay**2 + az**2)
        if ang_speed > MAX_ANGLE:
            scale = MAX_ANGLE / ang_speed
            ax *= scale; ay *= scale; az *= scale

        base_wrist = None
        if isinstance(base_pts, list) and len(base_pts) > WRIST_IDX:
            w = base_pts[WRIST_IDX]
            if w:
                base_wrist = w

        for p in base_pts:
            px, py, pz = float(p[0]), float(p[1]), float(p[2])
            if base_wrist is not None:
                rx, ry, rz = self._rotate_point((px, py, pz), base_wrist, (ax, ay, az))
                new_x = rx + vx
                new_y = ry + vy
                new_z = rz + vz
            else:
                new_x = px + vx
                new_y = py + vy
                new_z = pz + vz
            gen.add(new_x, new_y, new_z)

        # Adaptive aging based on motion
        lin_speed = math.sqrt(vx**2 + vy**2 + vz**2)
        ang_speed = math.sqrt(ax**2 + ay**2 + az**2)
        speed_factor = lin_speed + ang_speed * 50.0
        
        min_age = 1
        max_age = max(1, int(self.ghost_age_default))
        adaptive_age = int(max(min_age, min(max_age, self.ghost_age_default // (1 + int(speed_factor)))))
        gen._meta_gen_frame = self.frame_count
        gen._meta_age = adaptive_age

        return gen

    # ------------------------------------------------------------------
    # Prune ghosts when real hand appears
    # ------------------------------------------------------------------
    def _prune_ghosts_on_real(self, label):
        hist = self.position_histogram[label]
        if not hist:
            return False
            
        changed = False
        pure_real_hist = [e for e in hist if e.get("source") == self.SOURCE_REAL]
        
        if len(pure_real_hist) != len(hist):
            changed = True
            
        self.position_histogram[label].clear()
        self.position_histogram[label].extend(pure_real_hist)
        
        return changed

    # ------------------------------------------------------------------
    # Pygame helpers (unchanged)
    # ------------------------------------------------------------------
    def cvimage_to_pygame(self, image):
        size = image.shape[1::-1]
        pygame_surface = pygame.image.frombuffer(image.tobytes(), size, "RGB")
        return pygame_surface

    def render_camera_feed(self, surf, pos=(0,0)):
        if self.image is not None:
            surf.blit(pygame.transform.scale(self.image, surf.get_size()), pos)

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------
    def update(self, frame):
        if frame is None:
            self._log('ERROR', f"Frame is a None Object")
            return

        self.frame_count += 1
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ar_data = {
            "POSITION_DATA": {"LEFT": [], "RIGHT": []},
            "SCALE":         {"LEFT": 1,    "RIGHT": 1},
            "FRAME_TYPE":    {"LEFT" : "REAL", "RIGHT" : "REAL"},
            "CLICK_DIST":    {"LEFT": 0,    "RIGHT": 0},
            "CLICK_FLAG":    {"LEFT": False,"RIGHT": False},
            "HAND_PRESENCE" : False
        }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        self.image = self.cvimage_to_pygame(rgb)

        seen = []

        for label in ("LEFT", "RIGHT"):
            self._prune_ghosts_by_age(label)

        detections = []
        if getattr(res, 'multi_hand_landmarks', None):
            for lm_set, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handedness.classification[0].label.upper()
                try:
                    lm_w = lm_set.landmark[WRIST_IDX]
                    wrist_px = (float(lm_w.x) * self.W, float(lm_w.y) * self.H)
                except Exception:
                    wrist_px = None
                detections.append((label, wrist_px, lm_set))

            if len(detections) > 0:
                self._reconcile_handedness(detections)

            for i, (label, wrist_px, lm_set) in enumerate(detections):
                seen.append(label)

                # Get 3D normalized landmarks
                landmarks_norm = [(lm.x, lm.y, lm.z) for lm in lm_set.landmark]
                d = self.detector.update(label, [(lm.x, lm.y) for lm in lm_set.landmark])  # pinch uses 2D

                if self.presence_counter[label] < self.presence_threshold_on:
                    d["is_pinched"] = False

                pts = self.calculate_hand_points(lm_set, label, is_generated=False)
                
                # Update optical flow points (pixel coordinates)
                w_norm = landmarks_norm[WRIST_IDX]
                m_norm = landmarks_norm[MIDDLE_MCP_IDX]
                w_px = (w_norm[0] * self.W, w_norm[1] * self.H)
                m_px = (m_norm[0] * self.W, m_norm[1] * self.H)
                self.lk_tracked_points[label] = np.array([[w_px], [m_px]], dtype=np.float32)

                try:
                    self._prune_ghosts_on_real(label)
                except Exception:
                    pass

                ar_data["POSITION_DATA"][label] = pts
                ar_data["FRAME_TYPE"][label] = "REAL"
                ar_data["SCALE"][label] = d["scale"]
                ar_data["CLICK_DIST"][label] = d["rel_dist"]
                ar_data["CLICK_FLAG"][label] = d["is_pinched"]

                self.hands_tracker[label] = 0

        # Missing hand logic
        for label in ("LEFT", "RIGHT"):
            if label not in seen:
                self.hands_tracker[label] += 1
                self.presence_counter[label] = max(self.presence_threshold_off, self.presence_counter[label] - 1)

                ghost_pts = []
                
                if self.hands_tracker[label] == 1:
                    real_hist = [e for e in self.position_histogram[label] if e.get("source") == self.SOURCE_REAL]
                    if len(real_hist) >= 3:
                        self.ghost_velocity[label] = self.calculate_velocity(label, dir=1)
                        self.ghost_ttl_counter[label] = self.MAX_GHOST_TTL
                    else:
                        self.ghost_velocity[label] = [0.0]*6
                        self.ghost_ttl_counter[label] = 0
                        self.lk_tracked_points[label] = None
                    
                if self.ghost_ttl_counter[label] > 0 and self.hands_tracker[label] < self.absent_reset_threshold:
                    flow_success = False
                    
                    # Attempt optical flow (2D pixel tracking → 3D estimate)
                    if self.lk_tracked_points[label] is not None and self.prev_gray is not None:
                        p1, st, err = cv2.calcOpticalFlowPyrLK(
                            self.prev_gray, current_gray, self.lk_tracked_points[label], None, **self.lk_params
                        )
                        if st is not None and len(st) == 2 and st[0][0] == 1 and st[1][0] == 1:
                            p0 = self.lk_tracked_points[label].reshape(2, 2)
                            p1_flat = p1.reshape(2, 2)
                            
                            dx_px = p1_flat[0][0] - p0[0][0]
                            dy_px = p1_flat[0][1] - p0[0][1]
                            dx_norm = dx_px / self.W
                            dy_norm = dy_px / self.H
                            
                            old_dist_px = math.hypot(p0[1][0]-p0[0][0], p0[1][1]-p0[0][1])
                            new_dist_px = math.hypot(p1_flat[1][0]-p1_flat[0][0], p1_flat[1][1]-p1_flat[0][1])
                            scale_factor = new_dist_px / old_dist_px if old_dist_px > 0 else 1.0
                            dz_norm = -(scale_factor - 1.0) * 0.1   # heuristic depth change
                            
                            v_old = (p0[1][0]-p0[0][0], p0[1][1]-p0[0][1])
                            v_new = (p1_flat[1][0]-p1_flat[0][0], p1_flat[1][1]-p1_flat[0][1])
                            angle2d = 0.0
                            if not ((v_old[0]==0 and v_old[1]==0) or (v_new[0]==0 and v_new[1]==0)):
                                angle2d = math.atan2(v_new[1], v_new[0]) - math.atan2(v_old[1], v_old[0])
                                angle2d = self._normalize_angle(angle2d)
                            
                            vel = [dx_norm, dy_norm, dz_norm, 0.0, 0.0, angle2d]  # rotation around camera forward
                            ghost = self.generate_frames(vel, label)
                            self.ghost_velocity[label] = vel 
                            self.lk_tracked_points[label] = p1
                            flow_success = True
                            self._log('CORE', f'OPTICAL FLOW TRACKED {label}', True)

                    # Fallback to kinematic friction
                    if not flow_success:
                        vx, vy, vz, ax, ay, az = self.ghost_velocity[label]
                        speed = math.sqrt(vx**2 + vy**2 + vz**2)
                        if speed < 0.001:
                            self.ghost_velocity[label] = [0.0]*6
                            self._log('CORE', f'KINEMATIC HALT FOR {label}', True)
                            if self.ghost_ttl_counter[label] > 3:
                                self.ghost_ttl_counter[label] = 3
                        else:
                            self.ghost_velocity[label][0] *= self.KINEMATIC_FRICTION
                            self.ghost_velocity[label][1] *= self.KINEMATIC_FRICTION
                            self.ghost_velocity[label][2] *= self.KINEMATIC_FRICTION
                            self.ghost_velocity[label][3] *= (self.KINEMATIC_FRICTION * 0.7)
                            self.ghost_velocity[label][4] *= (self.KINEMATIC_FRICTION * 0.7)
                            self.ghost_velocity[label][5] *= (self.KINEMATIC_FRICTION * 0.7)
                            self._log('CORE', f'KINEMATIC DECAY FOR {label}', True)

                        vel = self.ghost_velocity[label]
                        ghost = self.generate_frames(vel, label)
                        self.lk_tracked_points[label] = None

                    if ghost is not None:
                        ghost_pts = self.calculate_hand_points(ghost, label, is_generated=True)
                    
                    self.ghost_ttl_counter[label] -= 1
                else:
                    self.ghost_velocity[label] = [0.0]*6
                    if self.hands_tracker[label] > self.MAX_GHOST_TTL:
                        if len(self.position_histogram[label]) > 0:
                            self._log('CORE', f'GHOST EXPIRED: Wiping spatial memory for {label}', True)
                            self.position_histogram[label].clear()
                            self.lk_tracked_points[label] = None
                            try:
                                self.detector.reset(label)
                            except Exception:
                                pass
                
                if ghost_pts and len(ghost_pts) > 0:
                    ar_data["FRAME_TYPE"][label] = "GHOST"
                    ar_data["POSITION_DATA"][label] = ghost_pts
                else:
                    ar_data["FRAME_TYPE"][label] = "REAL" 
                    ar_data["POSITION_DATA"][label] = []
                    
                last_pinch_state = self.detector.hands[label].is_pinched if label in self.detector.hands else False
                ar_data["CLICK_FLAG"][label] = last_pinch_state
                ar_data["CLICK_DIST"][label] = 0
                ar_data["SCALE"][label] = 1

        ar_data["HAND_PRESENCE"] = any(self.presence_counter[label] >= self.presence_threshold_on for label in ("LEFT","RIGHT"))

        self.ar_data = ar_data
        self.prev_gray = current_gray
        
        return ar_data