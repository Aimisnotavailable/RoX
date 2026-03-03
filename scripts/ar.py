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
        # store entries as dicts: {"pts": [(x,y),...], "source": "real"|"ghost", "frame": int, "gen_frame": int (for ghosts)}
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

        # ghost TTL default (frames). Set to 3 as requested.
        # This value now controls "max age since generation" for ghost entries.
        self.ghost_ttl_default = 3

        # our pinch detector
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
            # 'frame' is the frame when this entry was created (real or ghost)
            "frame": self.frame_count
        }
        # For generated ghosts, also store generation frame explicitly (used for age-based pruning)
        if is_generated:
            entry["gen_frame"] = self.frame_count

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
                    self._log('DEBUG', f'PRUNED TRAILING GHOST FOR {label} gen_frame={popped.get("gen_frame")} frame={popped.get("frame")}', True)
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
            self._log('CORE', f'APPENDED GHOST FOR {label} gen_frame={entry.get("gen_frame")} frame={entry.get("frame")}', True)
        
        return pts

    def _decrement_and_prune_ghosts(self, label):
        """
        Age-based pruning for ghost entries.

        Instead of per-entry TTL counters, remove any ghost entries that are older
        than `self.ghost_ttl_default` frames since their generation. This ensures
        ghosts only persist for a small number of frames (e.g., 2-3) after creation.
        """
        hist = self.position_histogram[label]
        changed = False

        # Remove ghosts that are older than allowed age (age measured from gen_frame)
        for e in hist[:]:
            if e.get("source") == self.SOURCE_GHOST:
                gen_frame = e.get("gen_frame", e.get("frame", None))
                if gen_frame is None:
                    # if no gen_frame, fall back to entry frame
                    gen_frame = e.get("frame", self.frame_count)
                age = self.frame_count - int(gen_frame)
                if age >= int(self.ghost_ttl_default):
                    try:
                        hist.remove(e)
                        changed = True
                        self._log('DEBUG', f'PRUNED GHOST DUE TO AGE FOR {label} gen_frame={gen_frame} age={age}', True)
                    except ValueError:
                        pass

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

    # -------------------------
    # Improved velocity estimator
    # -------------------------
    def calculate_velocity(self, label, dir=0, window=3, blend=0.8):
        """
        Improved velocity estimator with acceleration-based prediction.

        - window: number of valid frames to use (default 3).
        - blend: how strongly to trust the predicted velocity vs. the last measured velocity (0..1).
        Returns:
          - if dir==1: [pred_dx, pred_dy, pred_dtheta] (per-frame deltas)
          - else: predicted scalar speed (pixels/frame)
        """
        hist = self.position_histogram[label]
        # collect up to `window` valid entries with wrist and middle_mcp
        valid = []
        frames = []
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

        # need at least two frames to compute velocity
        if len(valid) < 2:
            return [0.0, 0.0, 0.0] if dir else 0.0

        # compute per-interval velocities and angular velocities
        lin_vels = []   # list of (vx, vy) per-frame
        ang_vels = []   # list of dtheta per-frame
        time_deltas = []

        for i in range(len(valid) - 1):
            # newest = valid[i], previous = valid[i+1] because we iterated reversed
            (w_new, m_new) = valid[i]
            (w_old, m_old) = valid[i + 1]
            f_new = frames[i]; f_old = frames[i + 1]
            df = max(1, f_new - f_old)  # frames between samples

            vx = (w_new[0] - w_old[0]) / df
            vy = (w_new[1] - w_old[1]) / df

            # angular velocity: angle between middle_mcp vectors
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
            time_deltas.append(df)

        # if only one interval, fallback to that velocity
        if len(lin_vels) == 1:
            last_vx, last_vy = lin_vels[0]
            last_dtheta = ang_vels[0]
            if dir:
                return [last_vx, last_vy, last_dtheta]
            return math.hypot(last_vx, last_vy)

        # compute average velocity and acceleration (simple finite differences)
        # velocities are ordered newest->older in lin_vels because valid was newest first
        # reverse to chronological order oldest->newest for acceleration calc
        lin_vels_chrono = list(reversed(lin_vels))
        ang_vels_chrono = list(reversed(ang_vels))

        # last measured velocity (most recent)
        last_vx, last_vy = lin_vels_chrono[-1]
        last_dtheta = ang_vels_chrono[-1]

        # compute acceleration as difference between last two measured velocities
        prev_vx, prev_vy = lin_vels_chrono[-2]
        prev_dtheta = ang_vels_chrono[-2]

        ax = last_vx - prev_vx
        ay = last_vy - prev_vy
        adtheta = last_dtheta - prev_dtheta

        # predicted next velocity (per-frame)
        pred_vx = last_vx + ax
        pred_vy = last_vy + ay
        pred_dtheta = last_dtheta + adtheta

        # optional smoothing/blending to avoid overshoot
        pred_vx = last_vx * (1.0 - blend) + pred_vx * blend
        pred_vy = last_vy * (1.0 - blend) + pred_vy * blend
        pred_dtheta = last_dtheta * (1.0 - blend) + pred_dtheta * blend

        # log for debugging
        self._log('DEBUG', f"[AR] predict_vel {label} last=({last_vx:.2f},{last_vy:.2f},{last_dtheta:.3f}) "
                           f"acc=({ax:.2f},{ay:.2f},{adtheta:.3f}) pred=({pred_vx:.2f},{pred_vy:.2f},{pred_dtheta:.3f})", True)

        if dir:
            return [pred_vx, pred_vy, pred_dtheta]
        return math.hypot(pred_vx, pred_vy)

    # -------------------------
    # Handedness reconciliation
    # -------------------------
    def _last_real_wrist(self, label):
        """Return last real wrist point (x,y) or None."""
        hist = self.position_histogram.get(label, [])
        # iterate from newest to oldest for first REAL source
        for e in reversed(hist):
            if e.get("source") == self.SOURCE_REAL:
                pts = e.get("pts", [])
                if isinstance(pts, list) and len(pts) > WRIST_IDX:
                    w = pts[WRIST_IDX]
                    if w and w != self.INVALID_POINT:
                        return (float(w[0]), float(w[1]))
        return None

    def _normalize_ghost_genframes(self, label):
        """
        Ensure ghost entries have a gen_frame and that it's not absurdly old.
        This helps after swaps so ghosts don't persist unexpectedly.
        """
        hist = self.position_histogram.get(label, [])
        for e in hist:
            if e.get("source") == self.SOURCE_GHOST:
                if e.get("gen_frame") is None:
                    e["gen_frame"] = e.get("frame", self.frame_count)
                # clamp gen_frame to not be older than current frame
                if e["gen_frame"] > self.frame_count:
                    e["gen_frame"] = self.frame_count

    def _swap_hand_state(self, a_label, b_label):
        """Swap all per-hand state between two labels."""
        # swap histograms
        self.position_histogram[a_label], self.position_histogram[b_label] = \
            self.position_histogram[b_label], self.position_histogram[a_label]
        # swap trackers and counters
        self.hands_tracker[a_label], self.hands_tracker[b_label] = \
            self.hands_tracker[b_label], self.hands_tracker[a_label]
        self.presence_counter[a_label], self.presence_counter[b_label] = \
            self.presence_counter[b_label], self.presence_counter[a_label]
        # swap detector state if present
        if a_label in self.detector.hands and b_label in self.detector.hands:
            ha = self.detector.hands[a_label]
            hb = self.detector.hands[b_label]
            self.detector.hands[a_label], self.detector.hands[b_label] = hb, ha
        else:
            # safe fallback: ensure both exist
            for L in (a_label, b_label):
                if L not in self.detector.hands:
                    self.detector.hands[L] = HandState(deque(maxlen=HISTOGRAM_SIZE), 0, False)
        # normalize ghost gen frames after swap
        self._normalize_ghost_genframes(a_label)
        self._normalize_ghost_genframes(b_label)

    def _reconcile_handedness(self, detections):
        """
        detections: list of tuples (label_str, wrist_px_or_None, landmark_obj)
        wrist_px_or_None = (x_px, y_px) or None if not available yet.
        """
        # build last-known wrists
        tracked = {lbl: self._last_real_wrist(lbl) for lbl in ("LEFT","RIGHT")}
        # build distance matrix
        pairs = []  # (det_idx, tracked_label, dist)
        for i, (det_label, wrist_pt, lm_obj) in enumerate(detections):
            if wrist_pt is None:
                continue
            for tlabel, tpt in tracked.items():
                if tpt is None:
                    continue
                dx = wrist_pt[0] - tpt[0]; dy = wrist_pt[1] - tpt[1]
                pairs.append((i, tlabel, math.hypot(dx, dy)))

        # greedy match: for small number of hands this is fine
        assigned = {}  # det_idx -> tracked_label
        used_tracked = set()
        pairs.sort(key=lambda x: x[2])  # smallest distance first
        MAX_MATCH_DIST = max(self.W, self.H) * 0.15  # tuneable threshold (15% of larger dim)
        for det_idx, tlabel, dist in pairs:
            if det_idx in assigned or tlabel in used_tracked:
                continue
            if dist <= MAX_MATCH_DIST:
                assigned[det_idx] = tlabel
                used_tracked.add(tlabel)

        # Now check for swaps: if a detection's MediaPipe label != assigned tracked label, swap states
        for i, (det_label, wrist_pt, lm_obj) in enumerate(detections):
            if i not in assigned:
                continue
            matched_label = assigned[i]
            if det_label != matched_label:
                # swap the internal state so that the detection's label aligns with the tracked history
                self._log('CORE', f"HANDEDNESS SWAP: detection {i} labeled {det_label} matches {matched_label}", True)
                self._swap_hand_state(det_label, matched_label)

    # -------------------------
    # Fixed generate_frames
    # -------------------------
    def generate_frames(self, velocity, label):
        """
        Generate ghost landmarks by applying rotation about wrist + translation.
        velocity: [dx,dy,dtheta] per-frame deltas (pixels/frame, radians/frame).
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

        # smoothing factor for lerp (0..1). 0 => keep original, 1 => full transform
        smooth = 0.5
        max_jump = max(self.W, self.H) * 0.5

        for p in base_pts:
            if not p or p == self.INVALID_POINT:
                gen.add(float(self.INVALID_POINT[0]), float(self.INVALID_POINT[1]))
                continue
            px, py = float(p[0]), float(p[1])
            if base_wrist is not None:
                # rotate around wrist by dtheta, then translate by dx,dy
                rx, ry = rotate_point(px, py, base_wrist[0], base_wrist[1], dtheta)
                transformed_x = rx + dx
                transformed_y = ry + dy

                # limit huge jumps by clamping the translation component
                jump = math.hypot(transformed_x - px, transformed_y - py)
                if jump > max_jump:
                    scale = max_jump / jump
                    transformed_x = px + (transformed_x - px) * scale
                    transformed_y = py + (transformed_y - py) * scale

                # proper lerp between original and transformed point
                new_x = px * (1.0 - smooth) + transformed_x * smooth
                new_y = py * (1.0 - smooth) + transformed_y * smooth
            else:
                # no wrist: fallback to simple translate with smoothing
                transformed_x = px + dx
                transformed_y = py + dy
                new_x = px * (1.0 - smooth) + transformed_x * smooth
                new_y = py * (1.0 - smooth) + transformed_y * smooth

            # final defensive clamp to finite numbers
            if not (math.isfinite(new_x) and math.isfinite(new_y)):
                gen.add(float(self.INVALID_POINT[0]), float(self.INVALID_POINT[1]))
            else:
                gen.add(new_x, new_y)

        # adaptive TTL is now interpreted as "max ghost age in frames"
        lin_speed = math.hypot(dx, dy)
        ang_speed = abs(dtheta)
        speed_factor = lin_speed + (ang_speed * 50.0)
        # we bias toward short-lived ghosts; keep adaptive logic but ensure small range
        min_ttl = 1
        max_ttl = max(1, int(self.ghost_ttl_default))
        # faster motion -> shorter age; slower motion -> allow up to ghost_ttl_default
        adaptive_age = int(max(min_ttl, min(max_ttl, self.ghost_ttl_default // (1 + int(speed_factor)))))
        # store generation frame on the generated object (calculate_hand_points will copy it)
        gen._meta_gen_frame = self.frame_count
        gen._meta_age = adaptive_age

        self._log('DEBUG', f"[AR] GENERATED GHOST {label} lin={lin_speed:.2f} ang={ang_speed:.3f} age={adaptive_age}", True)
        return gen

    def _prune_ghosts_on_real(self, label, keep_recent_frames=1):
        """
        When a real detection arrives for `label`, aggressively remove trailing ghosts
        and any ghosts older than `keep_recent_frames` frames relative to the current frame.
        """
        hist = self.position_histogram[label]
        if not hist:
            return False
        changed = False
        # Remove trailing ghosts immediately
        while len(hist) > 0 and hist[-1].get("source") == self.SOURCE_GHOST:
            hist.pop()
            changed = True
            self._log('DEBUG', f'PRUNED TRAILING GHOST ON REAL FOR {label}', True)
        # Remove ghosts that are older than keep_recent_frames
        cutoff_frame = self.frame_count - keep_recent_frames
        for e in hist[:]:
            if e.get("source") == self.SOURCE_GHOST:
                gen_frame = e.get("gen_frame", e.get("frame", 0))
                if gen_frame < cutoff_frame:
                    hist.remove(e)
                    changed = True
                    self._log('DEBUG', f'PRUNED OLD GHOST ON REAL FOR {label} gen_frame={gen_frame}', True)
        return changed

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

        # age-prune ghost entries each frame (remove ghosts older than ghost_ttl_default)
        for label in ("LEFT", "RIGHT"):
            self._decrement_and_prune_ghosts(label)

        # If we have detections, build a small detection list (label, wrist_px, lm_set)
        detections = []
        if getattr(res, 'multi_hand_landmarks', None):
            for lm_set, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = handedness.classification[0].label.upper()
                # attempt to extract wrist pixel quickly for matching
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

            # reconcile handedness before updating per-detection state
            if len(detections) > 0:
                self._reconcile_handedness(detections)

            # now process detections (after possible swaps)
            for i, (label, wrist_px, lm_set) in enumerate(detections):
                seen.append(label)

                # 1) pinch detection on normalized coords
                landmarks_norm = [(lm.x, lm.y) for lm in lm_set.landmark]
                d = self.detector.update(label, landmarks_norm)

                # If presence counter is low, force pinch off to avoid sticky clicks
                if self.presence_counter[label] < self.presence_threshold_on:
                    d["is_pinched"] = False

                # 2) draw & update pixel histogram
                # pass is_generated=False for real detections
                pts = self.calculate_hand_points(lm_set, label, is_generated=False)

                # Aggressively prune ghosts now that a real frame arrived
                try:
                    self._prune_ghosts_on_real(label)
                except Exception:
                    pass

                # 3) fill AR output - REAL frames
                ar_data["POSITION_DATA"][label] = pts
                ar_data["FRAME_TYPE"][label] = "REAL"
                ar_data["SCALE"][label] = d["scale"]
                ar_data["CLICK_DIST"][label] = d["rel_dist"]
                ar_data["CLICK_FLAG"][label] = d["is_pinched"]

                # reset hands_tracker for this label (we saw a real hand)
                self.hands_tracker[label] = 0

            # handle missing hands (generate ghost frames as predictions)
            for label in ("LEFT","RIGHT"):
                if label not in seen:
                    self.hands_tracker[label] += 1
                    # decrement presence counter on missing frames
                    self.presence_counter[label] = max(self.presence_threshold_off, self.presence_counter[label] - 1)
                    # Generate ghost frames as predictions
                    ghost_pts = []
                    if (len(self.position_histogram[label]) >= 1 and 
                        self.hands_tracker[label] < self.absent_reset_threshold):
                        vel = self.calculate_velocity(label, dir=1)
                        ghost = self.generate_frames(vel, label)
                        if ghost is not None:
                            # ensure generated object carries gen_frame info into histogram entry
                            # calculate_hand_points will set gen_frame based on current frame
                            ghost_pts = self.calculate_hand_points(ghost, label, is_generated=True)
                            self._log('CORE',
                                f'GENERATED GHOST FOR {label} HAND_TRACKER={self.hands_tracker[label]}', True)
                    
                    # For ghost frames: set FRAME_TYPE to GHOST but STILL provide position data
                    ar_data["FRAME_TYPE"][label] = "GHOST"
                    ar_data["POSITION_DATA"][label] = ghost_pts
                    
                    # Clear click flags for ghost frames (can't click with predicted hands)
                    ar_data["CLICK_FLAG"][label] = False
                    ar_data["CLICK_DIST"][label] = 0
                    ar_data["SCALE"][label] = 1

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
        else:
            self._log('ERROR', 'NO HANDS DETECTED', True)
            for label in ("LEFT","RIGHT"):
                self.hands_tracker[label] += 1
                # decrement presence counter on missing frames
                self.presence_counter[label] = max(self.presence_threshold_off, self.presence_counter[label] - 1)

                # Generate ghost frames as predictions when no hands detected
                ghost_pts = []
                if (len(self.position_histogram[label]) >= 1 and 
                    self.hands_tracker[label] < self.absent_reset_threshold):
                    vel = self.calculate_velocity(label, dir=1)
                    ghost = self.generate_frames(vel, label)
                    if ghost is not None:
                        ghost_pts = self.calculate_hand_points(ghost, label, is_generated=True)
                        self._log('CORE', f'GENERATED GHOST FOR {label} (no hands detected)', True)
                
                # All frames are ghost frames when no detection, but still provide position data
                ar_data["FRAME_TYPE"][label] = "GHOST"
                ar_data["POSITION_DATA"][label] = ghost_pts
                ar_data["CLICK_FLAG"][label] = False
                ar_data["CLICK_DIST"][label] = 0
                ar_data["SCALE"][label] = 1

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

        # determine HAND_PRESENCE using presence_counter hysteresis (per-frame)
        ar_data["HAND_PRESENCE"] = any(self.presence_counter[label] >= self.presence_threshold_on for label in ("LEFT","RIGHT"))

        # debug: log ghost counts
        self._log('DEBUG', f"[AR] GHOST_COUNTS L={sum(1 for e in self.position_histogram['LEFT'] if e['source']=='ghost')} R={sum(1 for e in self.position_histogram['RIGHT'] if e['source']=='ghost')}", True)

        self.ar_data = ar_data
        return ar_data
