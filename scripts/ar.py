# scripts/ar.py
from scripts.config import *
import math
import cv2
import pygame
import numpy as np # Added for Optical Flow
import mediapipe as mp
from collections import deque, namedtuple

# --- PINCH DETECTOR ---
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


# --- AR CLASS ---
class AR:
    SOURCE_REAL = "real"
    SOURCE_GHOST = "ghost"

    def __init__(self, screen_dim=(1280, 720)):
        self.W = screen_dim[0]
        self.H = screen_dim[1]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6
        )

        self.position_histogram = {'LEFT': [], 'RIGHT': []}
        self.hands_tracker     = {'LEFT': 0,      'RIGHT': 0     }
        self.presence_counter = {'LEFT': 0, 'RIGHT': 0}
        self.presence_threshold_on = 2   
        self.presence_threshold_off = -2 
        self.absent_reset_threshold = HISTOGRAM_SIZE
        self.pinch_absent_reset = max(3, HISTOGRAM_SIZE // 2)
        
        # --- GHOST KINEMATICS & TTL ---
        self.ghost_velocity = {'LEFT': [0.0, 0.0, 0.0], 'RIGHT': [0.0, 0.0, 0.0]}
        self.KINEMATIC_FRICTION = 0.85 
        self.MAX_GHOST_TTL = HISTOGRAM_SIZE - 1 
        self.ghost_ttl_counter = {'LEFT': 0, 'RIGHT': 0}
        self.ghost_age_default = 2
        
        # --- OPTICAL FLOW HYBRID TRACKER ---
        self.prev_gray = None
        self.lk_params = dict(
            winSize=(25, 25), 
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # Holds the exact pixel coordinates of the Wrist and Middle MCP for Optical Flow
        self.lk_tracked_points = {'LEFT': None, 'RIGHT': None}

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

    def _valid_landmark(self, lm):
        try:
            x = float(lm.x); y = float(lm.y)
        except Exception:
            return False
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        return True

    def calculate_hand_points(self, landmarks, label, is_generated=False):
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
                x_px = lx * self.W
                y_px = ly * self.H
            else:
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

    def calculate_velocity(self, label, dir=0, window=4):
        hist = self.position_histogram[label]
        valid = []
        frames = []
        for e in reversed(hist):
            if e.get("source") != self.SOURCE_REAL:
                continue
            pts = e.get("pts", [])
            if isinstance(pts, list) and len(pts) > max(WRIST_IDX, MIDDLE_MCP_IDX):
                w = pts[WRIST_IDX]
                m = pts[MIDDLE_MCP_IDX]
                if w and m:
                    valid.append((w, m))
                    frames.append(int(e.get("frame", self.frame_count)))
                    if len(valid) >= window:
                        break

        if len(valid) < 2:
            return [0.0, 0.0, 0.0] if dir else 0.0

        lin_vels = []
        ang_vels = []

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

        avg_vx = sum(v[0] for v in lin_vels) / len(lin_vels)
        avg_vy = sum(v[1] for v in lin_vels) / len(lin_vels)
        avg_dtheta = sum(ang_vels) / len(ang_vels)

        avg_dtheta = max(-0.08, min(0.08, avg_dtheta))
        
        MAX_SPEED = max(self.W, self.H) * 0.1
        speed = math.hypot(avg_vx, avg_vy)
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            avg_vx *= scale
            avg_vy *= scale

        if dir:
            return [avg_vx, avg_vy, avg_dtheta]
        return math.hypot(avg_vx, avg_vy)

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
        
        # Walk backward through history to check for frame gaps
        for i in range(len(hist)-2, -1, -1):
            f = hist[i].get("frame", -1)
            # If the gap between recorded frames is > 1, a blackout occurred. The streak is broken.
            if last_f - f > 1:
                break
            streak += 1
            last_f = f
            
        return streak

    def _reconcile_handedness(self, detections):
        if not detections:
            return

        # 1. BLOB CREATION: Structure raw MediaPipe data into a trackable blob.
        # We explicitly preserve d[0] (MediaPipe's original label guess) so the Bouncer 
        # can debate with the AI rather than just blindly trusting physical distance.
        blobs = [(i, d[0], d[1], d[2]) for i, d in enumerate(detections) if d[1] is not None]
        if not blobs:
            return

        # Fetch the last known *real* physical coordinates of both hands (ignoring ghosts)
        tracked = {
            "LEFT": self._last_real_wrist("LEFT"),
            "RIGHT": self._last_real_wrist("RIGHT")
        }

        # THE BLAST RADIUS: A dead hand only "owns" a tight 10% radius around its last known pixel.
        # This prevents a hand from stealing an identity from across the screen.
        MAX_JUMP = max(self.W, self.H) * 0.10 
        assigned_blobs = set()
        assigned_labels = set()
        final_assignments = {}

        # 2. THE BOUNCER (Temporal Freshness & Label Penalties)
        pairs = []
        for b_idx, orig_label, wrist_px, _ in blobs:
            for label, last_pos in tracked.items():
                if last_pos is not None:
                    # Calculate pure physical distance between the new hand and the historical track
                    dist = math.hypot(wrist_px[0] - last_pos[0], wrist_px[1] - last_pos[1])
                    
                    # Apply penalty if the AI's label disagrees with our Kinematic History
                    if orig_label != label:
                        track_length = len(self.position_histogram[label])
                        frames_absent = self.hands_tracker[label]
                        
                        # WEAK OR STALE MEMORY:
                        # If the hand has been missing for >2 frames, or tracking history is <5 frames,
                        # MediaPipe is likely correcting a first-frame hallucination.
                        # We apply a massive 200% penalty to force the Bouncer to back off and let MediaPipe win.
                        if frames_absent > 2 or track_length < 5:
                            dist += (MAX_JUMP * 2.0)
                        
                        # STRONG, FRESH MEMORY:
                        # The hand is actively on screen. MediaPipe is likely flickering mid-air.
                        # Apply a standard 60% penalty. The Bouncer will protect the state and override MediaPipe.
                        else:
                            dist += (MAX_JUMP * 0.6)

                    # Only consider it a match if the final penalized distance is within the blast radius
                    if dist < MAX_JUMP:
                        pairs.append((dist, b_idx, label))
        
        # Sort by lowest penalized distance to assign the most confident matches first
        pairs.sort(key=lambda x: x[0])
        for dist, b_idx, label in pairs:
            if b_idx not in assigned_blobs and label not in assigned_labels:
                final_assignments[b_idx] = label
                assigned_blobs.add(b_idx)
                assigned_labels.add(label)

        unassigned_blobs = [b for b in blobs if b[0] not in assigned_blobs]
        
        # 3. THE DOUBLE-BOX HALLUCINATION FIX
        # MediaPipe occasionally draws two overlapping bounding boxes on one single physical hand.
        # We must filter out any unassigned blob that is physically touching an already verified hand
        # so Optical Flow doesn't lock onto the duplicate and spawn a permanent ghost.
        filtered_unassigned = []
        for b in unassigned_blobs:
            b_idx, orig_label, wrist_px, _ = b
            is_dup = False
            
            # Check proximity against already finalized hand tracks
            for a_idx, a_label in final_assignments.items():
                a_wrist = detections[a_idx][1]
                # If it's within 5% of the screen to an existing hand, it's anatomically impossible. Drop it.
                if math.hypot(wrist_px[0] - a_wrist[0], wrist_px[1] - a_wrist[1]) < (MAX_JUMP * 0.5):
                    is_dup = True
                    break
            
            # Check proximity against other unassigned blobs (in case both duplicates were unassigned)
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

        # 4. HIERARCHY OF TRUST (Fallback Logic)
        # Scenario A: Both hands appear simultaneously with zero history.
        if len(unassigned_blobs) == 2 and len(assigned_labels) == 0:
            label_0 = unassigned_blobs[0][1] 
            label_1 = unassigned_blobs[1][1]
            
            # Trust MediaPipe natively if it correctly output one LEFT and one RIGHT
            if label_0 != label_1 and label_0 in ["LEFT", "RIGHT"] and label_1 in ["LEFT", "RIGHT"]:
                final_assignments[unassigned_blobs[0][0]] = label_0
                final_assignments[unassigned_blobs[1][0]] = label_1
                assigned_labels.update([label_0, label_1])
            else:
                # Emergency Parachute: MediaPipe failed (e.g., output two RIGHTs). 
                # Fallback to Spatial Partitioning (Left side of screen = LEFT hand)
                self._log('CORE', "MediaPipe duplicate labels detected. Falling back to Spatial Partitioning.", True)
                unassigned_blobs.sort(key=lambda b: b[2][0]) # Sort by X coordinate
                final_assignments[unassigned_blobs[0][0]] = "LEFT"
                final_assignments[unassigned_blobs[1][0]] = "RIGHT"
                assigned_labels.update(["LEFT", "RIGHT"])
                
        # Scenario B: Only one unassigned hand remains
        else:
            for b_idx, orig_label, wrist_px, _ in unassigned_blobs:
                avail_labels = [L for L in ["LEFT", "RIGHT"] if L not in assigned_labels]
                if not avail_labels:
                    break 
                
                # Trust MediaPipe's original label if that slot is currently empty
                if orig_label in avail_labels:
                    label = orig_label
                # If MediaPipe's label is taken, but there's only one slot left, take the remaining slot
                elif len(avail_labels) == 1:
                    label = avail_labels[0]
                # Absolute last resort fallback
                else:
                    label = "LEFT" if wrist_px[0] < self.W / 2 else "RIGHT"
                
                final_assignments[b_idx] = label
                assigned_labels.add(label)
                assigned_blobs.add(b_idx)

        # 5. EXECUTE THE ASSIGNMENTS
        # Mutate detections in-place to apply the overridden labels, 
        # which completely erases the dropped hallucinated duplicates.
        valid_detections = []
        for i, (orig_label, wrist_px, lm_set) in enumerate(detections):
            if i in final_assignments:
                new_label = final_assignments[i]
                if new_label != orig_label:
                    self._log('CORE', f"ZERO-TRUST OVERRIDE: Forced MediaPipe's {orig_label} to become {new_label}", True)
                valid_detections.append((new_label, wrist_px, lm_set))
                
        detections[:] = valid_detections

    def generate_frames(self, velocity, label):
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

        MAX_DTHETA = 0.35 
        dtheta = max(-MAX_DTHETA, min(MAX_DTHETA, dtheta))

        max_jump = max(self.W, self.H) * 0.15 
        speed = math.hypot(dx, dy)
        if speed > max_jump:
            scale = max_jump / speed
            dx *= scale
            dy *= scale

        base_wrist = None
        if isinstance(base_pts, list) and len(base_pts) > WRIST_IDX:
            w = base_pts[WRIST_IDX]
            if w:
                base_wrist = (float(w[0]), float(w[1]))

        def rotate_point(px, py, cx, cy, ang):
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

            gen.add(new_x, new_y)

        lin_speed = math.hypot(dx, dy)
        ang_speed = abs(dtheta)
        speed_factor = lin_speed + (ang_speed * 50.0)
        
        min_age = 1
        max_age = max(1, int(self.ghost_age_default))
        adaptive_age = int(max(min_age, min(max_age, self.ghost_age_default // (1 + int(speed_factor)))))
        gen._meta_gen_frame = self.frame_count
        gen._meta_age = adaptive_age

        return gen

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
        
        # Convert to Grayscale for Optical Flow Tracker
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
                    # FIX: Blindly multiply proportional coordinates
                    wrist_px = (float(lm_w.x) * self.W, float(lm_w.y) * self.H)
                except Exception:
                    wrist_px = None
                
                detections.append((label, wrist_px, lm_set))

            if len(detections) > 0:
                self._reconcile_handedness(detections)

            for i, (label, wrist_px, lm_set) in enumerate(detections):
                seen.append(label)

                landmarks_norm = [(lm.x, lm.y) for lm in lm_set.landmark]
                d = self.detector.update(label, landmarks_norm)

                if self.presence_counter[label] < self.presence_threshold_on:
                    d["is_pinched"] = False

                pts = self.calculate_hand_points(lm_set, label, is_generated=False)
                
                # OPTICAL FLOW SETUP: When real hand is detected, save Wrist & Middle MCP to track
                w = pts[WRIST_IDX]
                m = pts[MIDDLE_MCP_IDX]
                if w and m:
                    self.lk_tracked_points[label] = np.array([[w], [m]], dtype=np.float32)

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

        # UNIFIED missing hand logic + OPTICAL FLOW TRACKING
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
                        self.ghost_velocity[label] = [0.0, 0.0, 0.0]
                        self.ghost_ttl_counter[label] = 0
                        self.lk_tracked_points[label] = None # Invalidate Optical Flow
                    
                if self.ghost_ttl_counter[label] > 0 and self.hands_tracker[label] < self.absent_reset_threshold:
                    
                    flow_success = False
                    
                    # 1. ATTEMPT OPTICAL FLOW (Track physical pixels)
                    if self.lk_tracked_points[label] is not None and self.prev_gray is not None:
                        p1, st, err = cv2.calcOpticalFlowPyrLK(
                            self.prev_gray, current_gray, self.lk_tracked_points[label], None, **self.lk_params
                        )
                        
                        # If both wrist and middle MCP were found
                        if st is not None and len(st) == 2 and st[0][0] == 1 and st[1][0] == 1:
                            p0 = self.lk_tracked_points[label].reshape(2, 2)
                            p1_flat = p1.reshape(2, 2)
                            
                            # Measure how much the pixels actually moved
                            dx = p1_flat[0][0] - p0[0][0]
                            dy = p1_flat[0][1] - p0[0][1]
                            
                            v_old = (p0[1][0] - p0[0][0], p0[1][1] - p0[0][1])
                            v_new = (p1_flat[1][0] - p1_flat[0][0], p1_flat[1][1] - p1_flat[0][1])
                            
                            dtheta = 0.0
                            if not ((v_old[0] == 0 and v_old[1] == 0) or (v_new[0] == 0 and v_new[1] == 0)):
                                dtheta = self._normalize_angle(self._angle_between(v_old, v_new))
                                
                            # Apply precise pixel movement
                            vel = [dx, dy, dtheta]
                            ghost = self.generate_frames(vel, label)
                            
                            # Sync the kinetic engine to this physical movement
                            self.ghost_velocity[label] = vel 
                            self.lk_tracked_points[label] = p1
                            flow_success = True
                            self._log('CORE', f'OPTICAL FLOW TRACKED {label}', True)

                    # 2. FALLBACK TO KINEMATIC FRICTION (If hand occluded or left screen)
                    if not flow_success:
                        # Calculate current speed
                        vx, vy, vtheta = self.ghost_velocity[label]
                        speed = math.hypot(vx, vy)
                        
                        # STATIC FRICTION CUTOFF: If moving too slow, snap to a halt to prevent infinite drifting
                        if speed < 2.0:
                            self.ghost_velocity[label] = [0.0, 0.0, 0.0]
                            self._log('CORE', f'KINEMATIC HALT FOR {label}', True)
                        else:
                            # Apply dynamic friction
                            self.ghost_velocity[label][0] *= self.KINEMATIC_FRICTION
                            self.ghost_velocity[label][1] *= self.KINEMATIC_FRICTION
                            
                            # Angular momentum (rotation) should decay much faster than linear momentum
                            self.ghost_velocity[label][2] *= (self.KINEMATIC_FRICTION * 0.7) 
                            
                            self._log('CORE', f'KINEMATIC DECAY FOR {label}', True)

                        vel = self.ghost_velocity[label]
                        ghost = self.generate_frames(vel, label)
                        
                        # Invalidate tracking points so it doesn't try tracking background pixels
                        self.lk_tracked_points[label] = None

                    if ghost is not None:
                        ghost_pts = self.calculate_hand_points(ghost, label, is_generated=True)
                    
                    self.ghost_ttl_counter[label] -= 1
                else:
                    self.ghost_velocity[label] = [0.0, 0.0, 0.0]
                    # --- NEW AGGRESSIVE RESET LOGIC ---
                    if self.hands_tracker[label] > self.MAX_GHOST_TTL:
                        if len(self.position_histogram[label]) > 0:
                            self._log('CORE', f'GHOST EXPIRED: Wiping spatial memory for {label}', True)
                            self.position_histogram[label].clear()
                            # -> THIS IS THE KILL SWITCH <-
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

                ar_data["CLICK_FLAG"][label] = False
                ar_data["CLICK_DIST"][label] = 0
                ar_data["SCALE"][label] = 1

        ar_data["HAND_PRESENCE"] = any(self.presence_counter[label] >= self.presence_threshold_on for label in ("LEFT","RIGHT"))

        self.ar_data = ar_data
        
        # Save current frame for the next Optical Flow pass
        self.prev_gray = current_gray 
        
        return ar_data