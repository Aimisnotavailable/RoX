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
        # store entries as dicts: {"pts": [(x,y),...], "source": "real"|"ghost", "ttl": int}
        self.position_histogram = {'LEFT': [], 'RIGHT': []}
        # how many frames since last real detection for each hand
        self.hands_tracker     = {'LEFT': 0,      'RIGHT': 0     }

        # presence hysteresis counters (avoid flicker when detection briefly fails)
        self.presence_counter = {'LEFT': 0, 'RIGHT': 0}
        self.presence_threshold_on = 2   # require 2 consecutive frames to assert presence
        self.presence_threshold_off = -2 # require 2 consecutive misses to clear presence

        # how many consecutive frames a hand must be absent before we consider it "long absent"
        self.absent_reset_threshold = HISTOGRAM_SIZE * 2
        # shorter threshold specifically for resetting pinch state to avoid sticky pinches
        self.pinch_absent_reset = max(3, HISTOGRAM_SIZE // 2)

        # ghost TTL default (frames). Each generated ghost entry gets this TTL and decrements each render.
        self.ghost_ttl_default = max(1, HISTOGRAM_SIZE // 2)

        # our new pinch detector
        self.detector = PinchDetector()

        # diagnostic counters
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

        # debug flag to gate verbose logs
        self.debug = True

    def _log(self, level, msg, force=False):
        if not self.debug and not force:
            return
        get_logger_info(level, msg)

    def _is_normalized(self, x, y):
        """Return True if coordinates look like normalized MediaPipe coords."""
        return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

    def _sanitize_point(self, x_px, y_px, W, H, allow_negative_out_of_bounds=True):
        """
        Clamp and convert to integer pixel coords.
        - If coordinates are non-finite -> return None
        - If allow_negative_out_of_bounds is True:
            * If point is outside [0..W-1] or [0..H-1], return INVALID_POINT sentinel (-1,-1)
          Else:
            * Clamp to [0..W-1] and [0..H-1] (legacy behavior)
        """
        if not math.isfinite(x_px) or not math.isfinite(y_px):
            return None

        x_i = int(round(x_px))
        y_i = int(round(y_px))
        if allow_negative_out_of_bounds:
            if x_i < 0 or x_i >= W or y_i < 0 or y_i >= H:
                # return sentinel so renderers can filter it out explicitly
                return self.INVALID_POINT
            return (x_i, y_i)
        else:
            # legacy clamp behavior
            x_i = max(-1, min(W - 1, x_i))
            y_i = max(-1, min(H - 1, y_i))
            return (x_i, y_i)

    def _valid_landmark(self, lm):
        """
        Relaxed validity check:
        - Accept landmarks unless they are non-finite.
        - Treat exact (0,0) sentinel only if *all* landmarks are (0,0) (handled in render_hands).
        - Use visibility only as a weak hint (very small threshold).
        """
        try:
            x = float(lm.x); y = float(lm.y)
        except Exception:
            return False
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        # visibility is a weak hint; don't reject solely on low visibility here
        return True

    def calculate_hand_points(self, landmarks, label, is_generated=False):
        """
        Draws landmarks→pixel & updates pixel histogram.
        Handles both normalized (0..1) MediaPipe landmarks and
        pixel-space landmarks generated by generate_frames.
        `is_generated` marks whether this call is drawing ghost frames.

        Important: we preserve per-index sentinel entries (INVALID_POINT) so that
        downstream renderers can detect missing joints by index and avoid stretching.
        """
        pts = []
        raw_landmarks = list(landmarks.landmark)

        # First pass: count how many landmarks pass _valid_landmark
        valid_flags = [self._valid_landmark(lm) for lm in raw_landmarks]
        valid_count = sum(1 for v in valid_flags if v)

        # If no landmarks passed validity (rare), treat all as valid to avoid dropping frames
        if valid_count == 0 and len(raw_landmarks) > 0:
            self._log('DEBUG', f"[AR] WARNING: no landmarks passed _valid_landmark for {label}; accepting all to avoid drop", True)
            valid_flags = [True] * len(raw_landmarks)
            valid_count = len(raw_landmarks)

        for idx, lm in enumerate(raw_landmarks):
            if not valid_flags[idx]:
                # preserve index with sentinel
                pts.append(self.INVALID_POINT)
                continue

            # ensure numeric attributes
            try:
                lx = float(lm.x)
                ly = float(lm.y)
            except Exception:
                pts.append(self.INVALID_POINT)
                continue

            # If values look like normalized coords (0..1), convert to pixels.
            # If they are already >1, treat them as pixel coords.
            if self._is_normalized(lx, ly):
                x_px = lx * self.W
                y_px = ly * self.H
            else:
                x_px = lx
                y_px = ly

            sanitized = self._sanitize_point(x_px, y_px, self.W, self.H, allow_negative_out_of_bounds=True)
            if sanitized is None:
                pts.append(self.INVALID_POINT)
                continue
            pts.append(sanitized)

        # diagnostic print for render_hands
        self._log('DEBUG', f"[AR] render_hands {label} pts_count={len(pts)} is_generated={is_generated}")
        entry = {
            "pts": pts,
            "source": self.SOURCE_GHOST if is_generated else self.SOURCE_REAL,
            "ttl": None,
            "frame": self.frame_count
        }
        # If the landmarks object provided a meta ttl (from generate_frames), use it
        if is_generated and hasattr(landmarks, "_meta_ttl"):
            entry["ttl"] = int(landmarks._meta_ttl)
        else:
            entry["ttl"] = self.ghost_ttl_default if is_generated else None

        hist = self.position_histogram[label]

        # If this is a real detection and the hand was absent for a long time, reset histogram
        if not is_generated:
            if self.hands_tracker[label] >= self.absent_reset_threshold:
                # Reset histogram to only the current real entry for clean behavior
                self._log('CORE', f'RESETTING HISTOGRAM FOR {label} AFTER LONG ABSENCE {self.hands_tracker[label]}', True)
                hist.clear()
                hist.append(entry)
            else:
                # prune trailing ghosts so ghosts stop immediately
                while len(hist) > 0 and hist[-1].get("source") == self.SOURCE_GHOST:
                    popped = hist.pop()
                    self._log('DEBUG', f'PRUNED TRAILING GHOST FOR {label} TTL={popped.get("ttl")}', True)
                # append the real entry if we have at least one point (avoid appending empty pts lists)
                if len(pts) >= 1:
                    if len(hist) < HISTOGRAM_SIZE:
                        hist.append(entry)
                    else:
                        hist.pop(0)
                        hist.append(entry)
                    # presence counter increment (we saw a valid frame)
                    self.presence_counter[label] = min(self.presence_threshold_on, self.presence_counter[label] + 1)
                else:
                    # treat as a weak/missing frame: decrement presence counter
                    self.presence_counter[label] = max(self.presence_threshold_off, self.presence_counter[label] - 1)

            # reset hands_tracker because we saw a real hand (even if weak)
            self.hands_tracker[label] = 0
        else:
            # generated ghost entry: append and ensure we don't exceed HISTOGRAM_SIZE
            if len(hist) < HISTOGRAM_SIZE:
                hist.append(entry)
            else:
                hist.pop(0)
                hist.append(entry)
            self._log('CORE', f'APPENDED GHOST FOR {label} TTL={entry["ttl"]}', True)

    def _decrement_and_prune_ghosts(self, label):
        """
        Decrement TTL for ghost entries and remove any with ttl <= 0.
        Also ensure histogram length does not exceed HISTOGRAM_SIZE.
        """
        hist = self.position_histogram[label]
        changed = False
        # iterate and decrement TTL for ghosts
        for e in hist[:]:
            if e.get("source") == self.SOURCE_GHOST:
                if e.get("ttl") is None:
                    e["ttl"] = self.ghost_ttl_default
                e["ttl"] -= 1
                if e["ttl"] <= 0:
                    hist.remove(e)
                    changed = True
                    self._log('DEBUG', f'GHOST TTL EXPIRED FOR {label}', True)
        # enforce max length
        while len(hist) > HISTOGRAM_SIZE:
            hist.pop(0)
            changed = True
        return changed

    def _angle_between(self, a, b):
        """Return signed angle from vector a to b (radians). a,b are (x,y)."""
        return math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])

    def _normalize_angle(self, ang):
        """Normalize angle to [-pi, pi]."""
        while ang <= -math.pi:
            ang += 2 * math.pi
        while ang > math.pi:
            ang -= 2 * math.pi
        return ang

    def calculate_velocity(self, label, dir=0, window=2, max_disp=80, max_ang=0.9):
        """
        Returns either scalar speed (dir=0) or [dx,dy, dtheta] per-frame (dir=1).
        - dx,dy: per-frame wrist translation (pixels/frame)
        - dtheta: per-frame angular rotation (radians/frame), positive = CCW
        """
        hist = self.position_histogram[label]
        valid = []
        frames = []
        # collect last `window` entries with valid wrist and middle_mcp
        for e in reversed(hist):
            pts = e.get("pts", [])
            if isinstance(pts, list) and len(pts) > max(WRIST_IDX, MIDDLE_MCP_IDX):
                w = pts[WRIST_IDX]
                m = pts[MIDDLE_MCP_IDX]
                if w and w != self.INVALID_POINT and m and m != self.INVALID_POINT:
                    valid.append((w, m))
                    frames.append(int(e.get("frame", self.frame_count)))
                    if len(valid) >= window:
                        break

        if len(valid) >= 2:
            # newest = valid[0], previous = valid[1]
            (w_new, m_new) = valid[0]
            (w_old, m_old) = valid[1]
            f_new = frames[0]; f_old = frames[1]
            df = max(1, f_new - f_old)

            # linear per-frame translation (wrist)
            dx = (w_new[0] - w_old[0]) / df
            dy = (w_new[1] - w_old[1]) / df

            # angular per-frame rotation around wrist using middle mcp vector
            v_new = (m_new[0] - w_new[0], m_new[1] - w_new[1])
            v_old = (m_old[0] - w_old[0], m_old[1] - w_old[1])
            # if either vector is degenerate, fallback to zero rotation
            if (v_new[0] == 0 and v_new[1] == 0) or (v_old[0] == 0 and v_old[1] == 0):
                dtheta = 0.0
            else:
                raw_ang = self._angle_between(v_old, v_new)
                raw_ang = self._normalize_angle(raw_ang)
                dtheta = raw_ang / df  # radians per frame

            # clamp translation and angular velocity
            mag = math.hypot(dx, dy)
            if mag > max_disp:
                scale = max_disp / mag
                dx *= scale; dy *= scale

            if abs(dtheta) > max_ang:
                dtheta = math.copysign(max_ang, dtheta)

            self._log('DEBUG', f"[AR] calc_vel {label} dx={dx:.2f} dy={dy:.2f} dtheta={dtheta:.3f}", True)
            return [dx, dy, dtheta] if dir else math.hypot(dx, dy)

        return [0.0, 0.0, 0.0] if dir else 0.0



    def generate_frames(self, velocity, label, max_jump=120, lerp_alpha=0.35):
        """
        Generate ghost landmarks by applying rotation about wrist + translation.
        velocity: [dx,dy,dtheta] per-frame deltas (pixels/frame, radians/frame).
        """
        hist = self.position_histogram[label]
        base_entry = None
        # prefer newest REAL entry
        for e in reversed(hist):
            if e.get("source") == self.SOURCE_REAL and e.get("pts"):
                if any((p and p != self.INVALID_POINT) for p in e["pts"]):
                    base_entry = e
                    break
        # fallback to last entry only if recent
        if base_entry is None and hist:
            last = hist[-1]
            if (self.frame_count - int(last.get("frame", self.frame_count))) <= 2 and any((p and p != self.INVALID_POINT) for p in last.get("pts", [])):
                base_entry = last

        if base_entry is None:
            self._log('DEBUG', f'NO VALID BASE FOR GHOST GENERATION FOR {label}', True)
            return None

        base_pts = base_entry["pts"]
        base_frame = int(base_entry.get("frame", self.frame_count))

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

        # compute base wrist if available
        base_wrist = None
        if len(base_pts) > WRIST_IDX:
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

        # apply transform to each base point
        for p in base_pts:
            if not p or p == self.INVALID_POINT:
                gen.add(float(self.INVALID_POINT[0]), float(self.INVALID_POINT[1]))
                continue
            px, py = float(p[0]), float(p[1])
            if base_wrist is not None:
                # rotate around wrist by dtheta, then translate by dx,dy
                rx, ry = rotate_point(px, py, base_wrist[0], base_wrist[1], dtheta)
                new_x = rx + dx
                new_y = ry + dy

                # check wrist jump magnitude and lerp if too large
                proposed_wrist_x, proposed_wrist_y = rotate_point(base_wrist[0], base_wrist[1], base_wrist[0], base_wrist[1], dtheta)
                proposed_wrist_x += dx; proposed_wrist_y += dy
                jump = math.hypot(proposed_wrist_x - base_wrist[0], proposed_wrist_y - base_wrist[1])
                if jump > max_jump:
                    scale = max_jump / jump
                    new_x = px + (new_x - px) * scale
                    new_y = py + (new_y - py) * scale

                # lerp to smooth sudden changes
                new_x = px * (1.0 - lerp_alpha) + new_x * lerp_alpha
                new_y = py * (1.0 - lerp_alpha) + new_y * lerp_alpha
            else:
                # no wrist: fallback to simple translate
                new_x = px + dx
                new_y = py + dy

            gen.add(new_x, new_y)

        # adaptive TTL based on angular+linear speed
        lin_speed = math.hypot(dx, dy)
        ang_speed = abs(dtheta)
        # combine heuristics: faster motion -> shorter TTL
        speed_factor = lin_speed + (ang_speed * 50.0)  # scale angular to pixel-like magnitude
        adaptive_ttl = int(max(1, min(self.ghost_ttl_default * 3, self.ghost_ttl_default * (1.0 / (0.01 + speed_factor)))))

        gen._meta_ttl = adaptive_ttl
        gen._meta_base_frame = base_frame

        self._log('DEBUG', f"[AR] GENERATED GHOST {label} lin={lin_speed:.2f} ang={ang_speed:.3f} ttl={adaptive_ttl}", True)
        return gen

    def cvimage_to_pygame(self, image):
        """Convert cv2 image into a pygame surface"""
        # Get the image dimensions
        size = image.shape[1::-1]
        # Create a Pygame surface from the numpy array
        pygame_surface = pygame.image.frombuffer(image.tobytes(), size, "RGB")
        return pygame_surface

    def render_camera_feed(self, surf, pos=(0,0)):
        """Render the last camera image onto the provided pygame surface."""
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

        self._log('DEBUG', f"[AR] frame.shape: {getattr(frame, 'shape', None)}")
        # prep for Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        self.image = self.cvimage_to_pygame(rgb)

        # diagnostic prints
        self._log('DEBUG', f"[AR] FRAME {self.frame_count} Mediapipe hands: {bool(getattr(res, 'multi_hand_landmarks', None))}")
        if getattr(res, 'multi_hand_landmarks', None):
            for i, lm_set in enumerate(res.multi_hand_landmarks):
                valid_count = sum(1 for lm in lm_set.landmark if self._valid_landmark(lm))
                self._log('DEBUG', f"[AR]  hand {i} valid landmarks: {valid_count}")
        self._log('DEBUG', f"[AR] presence_counter: {self.presence_counter} hands_tracker: {self.hands_tracker}")

        # keep track of which hands appear
        seen = []

        # decrement ghost TTLs each frame and prune expired ghosts
        for label in ("LEFT", "RIGHT"):
            self._decrement_and_prune_ghosts(label)

        if getattr(res, 'multi_hand_landmarks', None):
            for lm_set, handedness in zip(res.multi_hand_landmarks,
                                          res.multi_handedness):
                label = handedness.classification[0].label.upper()
                seen.append(label)

                # 1) pinch detection on normalized coords
                landmarks_norm = [(lm.x, lm.y) for lm in lm_set.landmark]
                d = self.detector.update(label, landmarks_norm)

                # If presence counter is low, force pinch off to avoid sticky clicks
                if self.presence_counter[label] < self.presence_threshold_on:
                    d["is_pinched"] = False

                # 2) draw & update pixel histogram
                # pass is_generated=False for real detections
                self.calculate_hand_points(lm_set, label, is_generated=False)

                # 3) fill AR output
                # position data should be pixel-space pts from the last real entry
                hist = self.position_histogram[label]
                if hist:
                    # prefer most recent real pts
                    last_real = None
                    for e in reversed(hist):
                        if e.get("source") == self.SOURCE_REAL and e.get("pts"):
                            last_real = e["pts"]
                            break
                    # fallback to last entry even if ghost or partial
                    if last_real is None:
                        last_real = hist[-1]["pts"]
                    else:
                        ar_data["FRAME_TYPE"][label] = "REAL"
                    ar_data["POSITION_DATA"][label] = last_real
                else:
                    # keep previous ar_data value (avoid overwriting with empty)
                    pass

                ar_data["SCALE"][label]         = d["scale"]
                ar_data["CLICK_DIST"][label]    = d["rel_dist"]
                ar_data["CLICK_FLAG"][label]    = d["is_pinched"]

                # reset hands_tracker for this label (we saw a real hand)
                self.hands_tracker[label] = 0

            # ghost frames for missing hands
            for label in ("LEFT","RIGHT"):
                if label not in seen:
                    self.hands_tracker[label] += 1
                    # decrement presence counter on missing frames
                    self.presence_counter[label] = max(self.presence_threshold_off, self.presence_counter[label] - 1)

                    # If the hand has been missing for a while, reset pinch detector state and clear history
                    if self.hands_tracker[label] >= self.absent_reset_threshold:
                        if len(self.position_histogram[label]) > 0:
                            self._log('CORE', f'CLEARING HISTOGRAM FOR {label} DUE TO LONG ABSENCE {self.hands_tracker[label]}', True)
                            self.position_histogram[label].clear()
                        # reset pinch detector state for this hand to avoid sticky pinches
                        try:
                            self.detector.reset(label)
                        except Exception:
                            if label in self.detector.hands:
                                hs = self.detector.hands[label]
                                self.detector.hands[label] = HandState(hs.pos_hist, 0, False)
                            else:
                                self._log('ERROR', f"PinchDetector missing label {label}", True)
                        # ensure AR output flags are cleared for this hand
                        ar_data["CLICK_FLAG"][label] = False
                        ar_data["CLICK_DIST"][label] = 0
                        ar_data["SCALE"][label] = 1

                    # If the hand has been missing for pinch_absent_reset frames, reset pinch to avoid sticky pinches
                    if self.hands_tracker[label] >= self.pinch_absent_reset:
                        try:
                            self.detector.reset(label)
                        except Exception:
                            if label in self.detector.hands:
                                hs = self.detector.hands[label]
                                self.detector.hands[label] = HandState(hs.pos_hist, 0, False)
                            else:
                                self._log('ERROR', f"PinchDetector missing label {label}", True)
                        ar_data["CLICK_FLAG"][label] = False

                    # only generate ghosts if we have at least one real base and haven't exceeded lifetime
                    if (len(self.position_histogram[label]) >= 1):
                        vel = self.calculate_velocity(label, dir=1)
                        ghost = self.generate_frames(vel, label)
                        # render ghost frames and mark as generated (only if ghost is valid)
                        if ghost is not None:
                            self.calculate_hand_points(ghost, label, is_generated=True)
                            self._log('CORE',
                                f'GENERATED HAND FRAMES FOR {label} HAND_TRACKER={self.hands_tracker[label]}', True)
                        ar_data["FRAME_TYPE"][label] = "GHOST"
                    else:
                        # if we've been absent for a long time, ensure histogram is small
                        if self.hands_tracker[label] >= self.absent_reset_threshold:
                            if len(self.position_histogram[label]) > 0:
                                self._log('CORE', f'CLEARING HISTOGRAM FOR {label} DUE TO LONG ABSENCE {self.hands_tracker[label]}', True)
                                self.position_histogram[label].clear()
        else:
            self._log('ERROR', 'NO HANDS DETECTED', True)
            for label in ("LEFT","RIGHT"):
                self.hands_tracker[label] += 1
                # decrement presence counter on missing frames
                self.presence_counter[label] = max(self.presence_threshold_off, self.presence_counter[label] - 1)

                # If the hand has been missing for a while, reset pinch detector state and clear history
                if self.hands_tracker[label] >= self.absent_reset_threshold:
                    if len(self.position_histogram[label]) > 0:
                        self._log('CORE', f'CLEARING HISTOGRAM FOR {label} DUE TO LONG ABSENCE {self.hands_tracker[label]}', True)
                        self.position_histogram[label].clear()
                    try:
                        self.detector.reset(label)
                    except Exception:
                        if label in self.detector.hands:
                            hs = self.detector.hands[label]
                            self.detector.hands[label] = HandState(hs.pos_hist, 0, False)
                        else:
                            self._log('ERROR', f"PinchDetector missing label {label}", True)
                    ar_data["CLICK_FLAG"][label] = False
                    ar_data["CLICK_DIST"][label] = 0
                    ar_data["SCALE"][label] = 1

                # if we have recent history and haven't exceeded ghost lifetime, generate ghosts
                if (len(self.position_histogram[label]) >= 1):
                    vel = self.calculate_velocity(label, dir=1)
                    ghost = self.generate_frames(vel, label)
                    if ghost is not None:
                        self.calculate_hand_points(ghost, label, is_generated=True)
                    ar_data["FRAME_TYPE"][label] = "GHOST"
                else:
                    # no hands detected at all
                    pass

        # determine HAND_PRESENCE using presence_counter hysteresis (per-frame)
        ar_data["HAND_PRESENCE"] = any(self.presence_counter[label] >= self.presence_threshold_on for label in ("LEFT","RIGHT"))

        self.ar_data = ar_data
        return ar_data
