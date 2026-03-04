# scripts/ar.py
from scripts.config import *
import math
import cv2
import pygame
import mediapipe as mp
from collections import deque, namedtuple

# --- PINCH DETECTOR ---
HandState = namedtuple("HandState", ["pos_hist", "pinch_count", "is_pinched"])

class PinchDetector:
    def __init__(self):
        # one state per hand label
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
        """
        landmarks_norm: list of (x,y) in [0..1]
        Returns dict: { scale, rel_dist, raw_dist, is_pinched }
        """
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

        # hysteresis + debounce
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

        # save state
        self.hands[label] = HandState(state.pos_hist, pc, pinched)

        return {
            "raw_dist":  raw_dist,
            "scale":     hand_scale,
            "rel_dist":  rel_dist,
            "is_pinched": pinched
        }

    def reset(self, label):
        """
        Reset the pinch state for a given hand label.
        Clears pinch_count and is_pinched but preserves the position history deque.
        """
        if label in self.hands:
            state = self.hands[label]
            self.hands[label] = HandState(state.pos_hist, 0, False)


# --- AR CLASS ---
class AR:
    SOURCE_REAL = "real"
    SOURCE_GHOST = "ghost"
    # sentinel for invalid / out-of-bounds points
    INVALID_POINT = (-1, -1)

    def __init__(self, screen_dim=(1280, 720)):
        self.W = screen_dim[0]
        self.H = screen_dim[1]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2
        )

        # pixel-space histogram for smoothing & ghost-frames
        self.position_histogram = {'LEFT': [], 'RIGHT': []}
        # how many frames since last real detection for each hand
        self.hands_tracker     = {'LEFT': 0,      'RIGHT': 0     }

        # presence hysteresis counters (avoid flicker when detection briefly fails)
        self.presence_counter = {'LEFT': 0, 'RIGHT': 0}
        self.presence_threshold_on = 2   
        self.presence_threshold_off = -2 

        # how many consecutive frames a hand must be absent before we consider it "long absent"
        self.absent_reset_threshold = HISTOGRAM_SIZE * 2
        self.pinch_absent_reset = max(3, HISTOGRAM_SIZE // 2)
        
        # --- GHOST KINEMATICS & TTL ---
        # Ghost tracking and Kinematics
        self.ghost_velocity = {'LEFT': [0.0, 0.0, 0.0], 'RIGHT': [0.0, 0.0, 0.0]}
        self.KINEMATIC_FRICTION = 0.85 
        self.MAX_GHOST_TTL = 10 # Maximum frames a ghost is allowed to exist
        self.ghost_ttl_counter = {'LEFT': 0, 'RIGHT': 0}

        self.ghost_age_default = 2
        self.detector = PinchDetector()
        self.frame_count = 0

        self.ar_data = {
            "POSITION_DATA": {"LEFT": [], "RIGHT": []},
            "SCALE":         {"LEFT": 1,    "RIGHT": 1},
            "FRAME_TYPE":    {"LEFT" : "REAL", "RIGHT" : "REAL"},
            "CLICK_DIST":    {"LEFT": 0,    "RIGHT": 0},
            "CLICK_FLAG":    {"LEFT": False,"RIGHT": False},
            "HAND_PRESENCE" : False
        }
        self.image = None
        self.debug = True

    def _log(self, level, msg, force=False):
        if not self.debug and not force:
            return
        get_logger_info(level, msg)

    def _is_normalized(self, x, y):
        return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

    # def _sanitize_point(self, x_px, y_px, W, H, allow_negative_out_of_bounds=True):
    #     # if not math.isfinite(x_px) or not math.isfinite(y_px):
    #     #     return None

    #     # x_i = int(round(x_px))
    #     # y_i = int(round(y_px))
    #     # if allow_negative_out_of_bounds:
    #     #     if x_i < 0 or x_i >= W or y_i < 0 or y_i >= H:
    #     #         return self.INVALID_POINT
    #     #     return (x_i, y_i)
    #     # else:
    #     #     x_i = max(-1, min(W - 1, x_i))
    #     #     y_i = max(-1, min(H - 1, y_i))
    #     #     return (x_i, y_i)
    #     return (x_px, y_px)

    def _valid_landmark(self, lm):
        try:
            x = float(lm.x); y = float(lm.y)
        except Exception:
            return False
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        return True

    def calculate_hand_points(self, landmarks, label, is_generated=False):
        """
        Draws landmarks directly to pixel space. Allows out-of-bounds negative 
        coordinates naturally to prevent skeleton snapping/stretching.
        """
        pts = []
        raw_landmarks = list(landmarks.landmark)

        for lm in raw_landmarks:
            try:
                lx = float(lm.x)
                ly = float(lm.y)
            except Exception:
                pts.append((0.0, 0.0))
                continue

            if not is_generated:
                # Real MediaPipe landmarks are always proportions of the screen
                x_px = lx * self.W
                y_px = ly * self.H
            else:
                # Generated ghosts are already in exact pixel space
                x_px = lx
                y_px = ly

            pts.append((x_px, y_px))

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

    def _angle_between(self, a, b):
        return math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])

    def _normalize_angle(self, ang):
        while ang <= -math.pi:
            ang += 2 * math.pi
        while ang > math.pi:
            ang -= 2 * math.pi
        return ang

    # -------------------------
    # Improved velocity estimator
    # -------------------------
    def calculate_velocity(self, label, dir=0, window=4):
        """
        Stabilized velocity estimator.
        Averages recent real frames and clamps angular velocity to prevent violent flips.
        """
        hist = self.position_histogram[label]
        
        # 1. Collect ONLY REAL valid frames
        valid = []
        frames = []
        for e in reversed(hist):
            if e.get("source") != self.SOURCE_REAL:
                continue
            pts = e.get("pts", [])
            if isinstance(pts, list) and len(pts) > max(WRIST_IDX, MIDDLE_MCP_IDX):
                w = pts[WRIST_IDX]
                m = pts[MIDDLE_MCP_IDX]
                if w and w != self.INVALID_POINT and m and m != self.INVALID_POINT:
                    valid.append((w, m))
                    frames.append(int(e.get("frame", self.frame_count)))
                    if len(valid) >= window:
                        break

        if len(valid) < 2:
            return [0.0, 0.0, 0.0] if dir else 0.0

        lin_vels = []
        ang_vels = []

        # 2. Compute interval velocities
        for i in range(len(valid) - 1):
            (w_new, m_new) = valid[i]
            (w_old, m_old) = valid[i + 1]
            f_new = frames[i]; f_old = frames[i + 1]
            df = max(1, f_new - f_old)

            vx = (w_new[0] - w_old[0]) / df
            vy = (w_new[1] - w_old[1]) / df

            v_new = (m_new[0] - w_new[0], m_new[1] - w_new[1])
            v_old = (m_old[0] - w_old[0], m_old[1] - w_old[1])
            if (v_new[0] == 0 and v_new[1] == 0) or (v_old[0] == 0 and v_old[1] == 0):
                dtheta = 0.0
            else:
                raw_ang = self._angle_between(v_old, v_new)
                raw_ang = self._normalize_angle(raw_ang)
                dtheta = raw_ang / df

            lin_vels.append((vx, vy))
            ang_vels.append(dtheta)

        # 3. Average them out to smooth over MediaPipe jitter
        avg_vx = sum(v[0] for v in lin_vels) / len(lin_vels)
        avg_vy = sum(v[1] for v in lin_vels) / len(lin_vels)
        avg_dtheta = sum(ang_vels) / len(ang_vels)

        # 4. ANTI-FLIP CLAMPING
        # Max rotation of ~5 degrees (0.08 rad) per frame to prevent wild spinning
        avg_dtheta = max(-0.08, min(0.08, avg_dtheta))
        
        # Max linear speed clamp (prevents shooting off-screen from extreme jitter)
        MAX_SPEED = max(self.W, self.H) * 0.1
        speed = math.hypot(avg_vx, avg_vy)
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            avg_vx *= scale
            avg_vy *= scale

        if dir:
            return [avg_vx, avg_vy, avg_dtheta]
        return math.hypot(avg_vx, avg_vy)
    # -------------------------
    # Handedness reconciliation
    # -------------------------
    def _last_real_wrist(self, label):
        hist = self.position_histogram.get(label, [])
        for e in reversed(hist):
            if e.get("source") == self.SOURCE_REAL:
                pts = e.get("pts", [])
                if isinstance(pts, list) and len(pts) > WRIST_IDX:
                    w = pts[WRIST_IDX]
                    if w:
                        return (float(w[0]), float(w[1]))
        return None

    def _normalize_ghost_genframes(self, label):
        hist = self.position_histogram.get(label, [])
        for e in hist:
            if e.get("source") == self.SOURCE_GHOST:
                if e.get("gen_frame") is None:
                    e["gen_frame"] = e.get("frame", self.frame_count)
                if e["gen_frame"] > self.frame_count:
                    e["gen_frame"] = self.frame_count

    def _swap_hand_state(self, a_label, b_label):
        self.position_histogram[a_label], self.position_histogram[b_label] = \
            self.position_histogram[b_label], self.position_histogram[a_label]
        self.hands_tracker[a_label], self.hands_tracker[b_label] = \
            self.hands_tracker[b_label], self.hands_tracker[a_label]
        self.presence_counter[a_label], self.presence_counter[b_label] = \
            self.presence_counter[b_label], self.presence_counter[a_label]
        
        # Swap TTL and Velocities
        self.ghost_ttl_counter[a_label], self.ghost_ttl_counter[b_label] = \
            self.ghost_ttl_counter[b_label], self.ghost_ttl_counter[a_label]
        self.ghost_velocity[a_label], self.ghost_velocity[b_label] = \
            self.ghost_velocity[b_label], self.ghost_velocity[a_label]

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

    def _reconcile_handedness(self, detections):
        """
        TRUE ZERO-TRUST RECONCILIATION:
        Treats MediaPipe's output as anonymous blobs. Assigns "LEFT" or "RIGHT"
        strictly based on spatial history and process of elimination.
        Relabels the incoming detection directly without corrupting history.
        """
        if not detections:
            return

        # 1. Strip labels. Treat as anonymous blobs. 
        # Create list of (original_idx, wrist_px, lm_set)
        blobs = [(i, d[1], d[2]) for i, d in enumerate(detections) if d[1] is not None]
        
        if not blobs:
            return

        # Get the last guaranteed real position for both hands
        tracked = {
            "LEFT": self._last_real_wrist("LEFT"),
            "RIGHT": self._last_real_wrist("RIGHT")
        }

        # 25% of the screen is our max allowance for real-world fast movement
        MAX_JUMP = max(self.W, self.H) * 0.25 

        assigned_blobs = set()
        assigned_labels = set()
        final_assignments = {}

        # 2. CONTINUITY MATCHING: Map blobs to history based on shortest distance
        pairs = []
        for b_idx, wrist_px, _ in blobs:
            for label, last_pos in tracked.items():
                if last_pos is not None:
                    dist = math.hypot(wrist_px[0] - last_pos[0], wrist_px[1] - last_pos[1])
                    if dist < MAX_JUMP:
                        pairs.append((dist, b_idx, label))
        
        # Sort by shortest distance and assign greedily
        pairs.sort(key=lambda x: x[0])
        for dist, b_idx, label in pairs:
            if b_idx not in assigned_blobs and label not in assigned_labels:
                final_assignments[b_idx] = label
                assigned_blobs.add(b_idx)
                assigned_labels.add(label)

        # 3. HANDLE UNASSIGNED BLOBS (Newly appeared hands without valid history)
        unassigned_blobs = [b for b in blobs if b[0] not in assigned_blobs]
        
        # Scenario A: Both hands appeared on screen at the exact same time
        if len(unassigned_blobs) == 2 and len(assigned_labels) == 0:
            # Sort them horizontally. Leftmost is LEFT, Rightmost is RIGHT.
            unassigned_blobs.sort(key=lambda b: b[1][0])
            final_assignments[unassigned_blobs[0][0]] = "LEFT"
            final_assignments[unassigned_blobs[1][0]] = "RIGHT"
            assigned_labels.update(["LEFT", "RIGHT"])
            
        # Scenario B: Assign remaining hands one by one
        else:
            for b_idx, wrist_px, _ in unassigned_blobs:
                avail_labels = [L for L in ["LEFT", "RIGHT"] if L not in assigned_labels]
                if not avail_labels:
                    break # No labels left to assign
                
                # Process of elimination (If LEFT is taken, this must be RIGHT)
                if len(avail_labels) == 1:
                    label = avail_labels[0]
                # Screen splitting fallback (If tracking is lost, use screen halves)
                else:
                    label = "LEFT" if wrist_px[0] < self.W / 2 else "RIGHT"
                
                final_assignments[b_idx] = label
                assigned_labels.add(label)
                assigned_blobs.add(b_idx)

        # 4. OVERRIDE MEDIAPIPE
        for i, (orig_label, wrist_px, lm_set) in enumerate(detections):
            if i in final_assignments:
                new_label = final_assignments[i]
                if new_label != orig_label:
                    self._log('CORE', f"ZERO-TRUST OVERRIDE: Forced MediaPipe's {orig_label} to become {new_label}", True)
                
                # Update the detection tuple in place. History is protected!
                detections[i] = (new_label, wrist_px, lm_set)

    # -------------------------
    # Fixed generate_frames
    # -------------------------
    def generate_frames(self, velocity, label):
        """
        Generate ghost landmarks by applying rotation about wrist + translation.
        Enforces Rigid Body Kinematics so the hand does not deform or squash.
        """
        hist = self.position_histogram[label]
        
        if len(hist) > 0 :
            base_pts = hist[-1].get("pts", []) or []
        else:
            base_pts = []

        class LM:
            def __init__(self,x,y):
                self.x = float(x); self.y = float(y)
        class HL:
            def __init__(self):
                self.landmark = []
            def add(self,x,y):
                self.landmark.append(LM(x,y))

        gen = HL()

        dx, dy, dtheta = (0.0, 0.0, 0.0)
        if isinstance(velocity, (list, tuple)) and len(velocity) >= 3:
            dx, dy, dtheta = float(velocity[0]), float(velocity[1]), float(velocity[2])

        # 1. Clamp Angular Velocity (prevent "helicopter" spinning)
        MAX_DTHETA = 0.35 # ~20 degrees max rotation per frame
        dtheta = max(-MAX_DTHETA, min(MAX_DTHETA, dtheta))

        # 2. Clamp Translation globally (prevents the skeleton from deforming)
        max_jump = max(self.W, self.H) * 0.15 # 15% of screen max per frame
        speed = math.hypot(dx, dy)
        if speed > max_jump:
            scale = max_jump / speed
            dx *= scale
            dy *= scale

        # compute base wrist if available
        base_wrist = None
        if isinstance(base_pts, list) and len(base_pts) > WRIST_IDX:
            w = base_pts[WRIST_IDX]
            if w and w != self.INVALID_POINT:
                base_wrist = (float(w[0]), float(w[1]))

        # rotation helper
        def rotate_point(px, py, cx, cy, ang):
            # rotate (px,py) around center (cx,cy) by ang radians CCW
            s = math.sin(ang); c = math.cos(ang)
            x = px - cx; y = py - cy
            rx = x * c - y * s
            ry = x * s + y * c
            return (rx + cx, ry + cy)
        
        for p in base_pts:
            px, py = float(p[0]), float(p[1])
            
            if base_wrist is not None:
                rx, ry = rotate_point(px, py, base_wrist[0], base_wrist[1], dtheta)
                new_x = rx + dx
                new_y = ry + dy
            else:
                new_x = px + dx
                new_y = py + dy

            # Just append directly. Allow off-screen negatives!
            gen.add(new_x, new_y)

        # adaptive age suggestion 
        lin_speed = math.hypot(dx, dy)
        ang_speed = abs(dtheta)
        speed_factor = lin_speed + (ang_speed * 50.0)
        
        min_age = 1
        max_age = max(1, int(self.ghost_age_default))
        adaptive_age = int(max(min_age, min(max_age, self.ghost_age_default // (1 + int(speed_factor)))))
        gen._meta_gen_frame = self.frame_count
        gen._meta_age = adaptive_age

        self._log('DEBUG', f"[AR] GENERATED GHOST {label} lin={lin_speed:.2f} ang={ang_speed:.3f} age={adaptive_age}", True)
        return gen

    def _prune_ghosts_on_real(self, label):
        """
        Ruthlessly purges all ghost frames from the history the moment a real hand is detected.
        This prevents predictive tracking errors from compounding.
        """
        hist = self.position_histogram[label]
        if not hist:
            return False
            
        changed = False
        # Filter the history to ONLY keep real frames
        pure_real_hist = [e for e in hist if e.get("source") == self.SOURCE_REAL]
        
        if len(pure_real_hist) != len(hist):
            changed = True
            
        # Replace the polluted histogram with the purified one
        self.position_histogram[label].clear()
        self.position_histogram[label].extend(pure_real_hist)
        
        return changed

    def cvimage_to_pygame(self, image):
        size = image.shape[1::-1]
        pygame_surface = pygame.image.frombuffer(image.tobytes(), size, "RGB")
        return pygame_surface

    def render_camera_feed(self, surf, pos=(0,0)):
        if self.image is not None:
            surf.blit(pygame.transform.scale(self.image, surf.get_size()), pos)

    def update(self, frame):
        if frame is None:
            self._log('ERROR', f"Frame is a None Object")
            return

        self.frame_count += 1

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

        # 1. Prune old ghosts purely by age
        for label in ("LEFT", "RIGHT"):
            self._prune_ghosts_by_age(label)

        # 2. Extract RAW detections
        detections = []
        if getattr(res, 'multi_hand_landmarks', None):
            for lm_set, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handedness.classification[0].label.upper()
                try:
                    lm_w = lm_set.landmark[WRIST_IDX]
                    lx = float(lm_w.x); ly = float(lm_w.y)
                    if self._is_normalized(lx, ly):
                        wrist_px = (lx * self.W, ly * self.H)
                    else:
                        wrist_px = (lx, ly)
                except Exception:
                    wrist_px = None
                
                detections.append((label, wrist_px, lm_set))

            # 3. Reconcile Handedness (Zero-Trust Policy)
            if len(detections) > 0:
                self._reconcile_handedness(detections)

            # 4. Process valid, reconciled hands
            for i, (label, wrist_px, lm_set) in enumerate(detections):
                seen.append(label)

                landmarks_norm = [(lm.x, lm.y) for lm in lm_set.landmark]
                d = self.detector.update(label, landmarks_norm)

                # Prevent clicks if the hand is flickering
                if self.presence_counter[label] < self.presence_threshold_on:
                    d["is_pinched"] = False

                pts = self.calculate_hand_points(lm_set, label, is_generated=False)

                try:
                    # Purge ghosts entirely upon finding reality
                    self._prune_ghosts_on_real(label)
                except Exception:
                    pass

                ar_data["POSITION_DATA"][label] = pts
                ar_data["FRAME_TYPE"][label] = "REAL"
                ar_data["SCALE"][label] = d["scale"]
                ar_data["CLICK_DIST"][label] = d["rel_dist"]
                ar_data["CLICK_FLAG"][label] = d["is_pinched"]

                self.hands_tracker[label] = 0

        # 5. UNIFIED missing hand logic (handles both 1 missing hand or no hands detected)
        for label in ("LEFT", "RIGHT"):
            if label not in seen:
                self.hands_tracker[label] += 1
                self.presence_counter[label] = max(self.presence_threshold_off, self.presence_counter[label] - 1)

                ghost_pts = []
                
                # A. Capture velocity exactly on the frame it gets lost
                if self.hands_tracker[label] == 1:
                    real_hist = [e for e in self.position_histogram[label] if e.get("source") == self.SOURCE_REAL]
                    
                    # ANTI-PHANTOM CHECK: 
                    # Only spawn ghosts if this hand was solid reality for at least 3 frames.
                    if len(real_hist) >= 3:
                        self.ghost_velocity[label] = self.calculate_velocity(label, dir=1)
                        self.ghost_ttl_counter[label] = self.MAX_GHOST_TTL
                    else:
                        self.ghost_velocity[label] = [0.0, 0.0, 0.0]
                        self.ghost_ttl_counter[label] = 0
                    
                # B. Generate ghosts using Kinematic Friction & TTL
                if self.ghost_ttl_counter[label] > 0 and self.hands_tracker[label] < self.absent_reset_threshold:
                    # Apply friction deceleration (graceful braking)
                    self.ghost_velocity[label][0] *= self.KINEMATIC_FRICTION
                    self.ghost_velocity[label][1] *= self.KINEMATIC_FRICTION
                    self.ghost_velocity[label][2] *= self.KINEMATIC_FRICTION
                    
                    ghost = self.generate_frames(self.ghost_velocity[label], label)
                    if ghost is not None:
                        ghost_pts = self.calculate_hand_points(ghost, label, is_generated=True)
                        self._log('CORE', f'GENERATED GHOST FOR {label} (TTL: {self.ghost_ttl_counter[label]})', True)
                    
                    self.ghost_ttl_counter[label] -= 1
                else:
                    # Ensure velocity is zeroed out if TTL expires or hand is long absent
                    self.ghost_velocity[label] = [0.0, 0.0, 0.0]
                
                # C. Final state sync 
                if ghost_pts and len(ghost_pts) > 0:
                    ar_data["FRAME_TYPE"][label] = "GHOST"
                    ar_data["POSITION_DATA"][label] = ghost_pts
                else:
                    ar_data["FRAME_TYPE"][label] = "REAL" 
                    ar_data["POSITION_DATA"][label] = []

                ar_data["CLICK_FLAG"][label] = False
                ar_data["CLICK_DIST"][label] = 0
                ar_data["SCALE"][label] = 1

                # D. Deep cleanup for hands gone too long
                if self.hands_tracker[label] >= self.absent_reset_threshold:
                    if len(self.position_histogram[label]) > 0:
                        self._log('CORE', f'CLEARING HISTOGRAM FOR {label} DUE TO LONG ABSENCE', True)
                        self.position_histogram[label].clear()
                    try:
                        self.detector.reset(label)
                    except Exception:
                        pass

        # 6. Global Hand Presence Check
        ar_data["HAND_PRESENCE"] = any(self.presence_counter[label] >= self.presence_threshold_on for label in ("LEFT","RIGHT"))

        self.ar_data = ar_data
        return ar_data