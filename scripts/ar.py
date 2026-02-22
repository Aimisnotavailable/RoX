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
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
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

    def _is_normalized(self, x, y):
        """Return True if coordinates look like normalized MediaPipe coords."""
        return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0

    def _sanitize_point(self, x_px, y_px, W, H):
        """Clamp and convert to integer pixel coords; return None if invalid."""
        if not math.isfinite(x_px) or not math.isfinite(y_px):
            return None
        # optional clamp to surface bounds to avoid drawing outside surface
        x_i = int(round(x_px))
        y_i = int(round(y_px))
        # keep within a small margin to avoid negative indices
        x_i = max(0, min(W - 1, x_i))
        y_i = max(0, min(H - 1, y_i))
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

    def render_hands(self, surf, landmarks, label, is_generated=False):
        """
        Draws landmarks→pixel & updates pixel histogram.
        Handles both normalized (0..1) MediaPipe landmarks and
        pixel-space landmarks generated by generate_frames.
        `is_generated` marks whether this call is drawing ghost frames.
        """
        W, H = surf.get_width(), surf.get_height()
        pts = []
        raw_landmarks = list(landmarks.landmark)

        # First pass: count how many landmarks pass _valid_landmark
        valid_flags = [self._valid_landmark(lm) for lm in raw_landmarks]
        valid_count = sum(1 for v in valid_flags if v)

        # If no landmarks passed validity (rare), treat all as valid to avoid dropping frames
        if valid_count == 0 and len(raw_landmarks) > 0:
            # log once per frame for diagnostics
            get_logger_info('DEBUG',f"[AR] WARNING: no landmarks passed _valid_landmark for {label}; accepting all to avoid drop")
            valid_flags = [True] * len(raw_landmarks)
            valid_count = len(raw_landmarks)

        for idx, lm in enumerate(raw_landmarks):
            if not valid_flags[idx]:
                continue

            # ensure numeric attributes
            try:
                lx = float(lm.x)
                ly = float(lm.y)
            except Exception:
                continue

            # If values look like normalized coords (0..1), convert to pixels.
            # If they are already >1, treat them as pixel coords.
            if self._is_normalized(lx, ly):
                # standard mapping: left-to-right is lx*W; keep mirror if you prefer by flipping elsewhere
                x_px = lx * W
                y_px = ly * H
            else:
                # already pixel coordinates (or ghost frames)
                x_px = lx
                y_px = ly

            sanitized = self._sanitize_point(x_px, y_px, W, H)
            if sanitized is None:
                continue

            pts.append(sanitized)
            # draw small landmark dot
            try:
                pygame.draw.circle(surf, (255,255,255), center=sanitized, radius=2)
            except TypeError:
                pygame.draw.circle(surf, (255,255,255), center=(int(sanitized[0]), int(sanitized[1])), radius=2)

        # diagnostic print for render_hands
        get_logger_info('DEBUG', f"[AR] render_hands {label} pts_count={len(pts)} is_generated={is_generated}")

        # store as dict with source tag (store pixel coords only)
        entry = {
            "pts": pts,
            "source": "ghost" if is_generated else "real",
            # TTL only used for ghosts; real entries have None
            "ttl": self.ghost_ttl_default if is_generated else None
        }
        hist = self.position_histogram[label]

        # If this is a real detection and the hand was absent for a long time, reset histogram
        if not is_generated:
            if self.hands_tracker[label] >= self.absent_reset_threshold:
                # Reset histogram to only the current real entry for clean behavior
                get_logger_info('CORE', f'RESETTING HISTOGRAM FOR {label} AFTER LONG ABSENCE {self.hands_tracker[label]}', True)
                hist.clear()
                hist.append(entry)
            else:
                # prune trailing ghosts so ghosts stop immediately
                while len(hist) > 0 and hist[-1].get("source") == "ghost":
                    popped = hist.pop()
                    get_logger_info('DEBUG', f'PRUNED TRAILING GHOST FOR {label} TTL={popped.get("ttl")}', True)
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
            get_logger_info('CORE', f'APPENDED GHOST FOR {label} TTL={entry["ttl"]}', True)

        # draw connections (only if we have enough points)
        if pts:
            try:
                max_idx = max(max(c) for c in self.mp_hands.HAND_CONNECTIONS)
            except Exception:
                max_idx = 0
            if len(pts) > max_idx:
                for c in self.mp_hands.HAND_CONNECTIONS:
                    a_idx, b_idx = c[0], c[1]
                    try:
                        pygame.draw.line(surf, (0,0,255),
                                         pts[a_idx], pts[b_idx], 1)
                    except Exception:
                        continue

    def _decrement_and_prune_ghosts(self, label):
        """
        Decrement TTL for ghost entries and remove any with ttl <= 0.
        Also ensure histogram length does not exceed HISTOGRAM_SIZE.
        """
        hist = self.position_histogram[label]
        changed = False
        # iterate and decrement TTL for ghosts
        for e in hist[:]:
            if e.get("source") == "ghost":
                if e.get("ttl") is None:
                    e["ttl"] = self.ghost_ttl_default
                e["ttl"] -= 1
                if e["ttl"] <= 0:
                    hist.remove(e)
                    changed = True
                    get_logger_info('DEBUG', f'GHOST TTL EXPIRED FOR {label}', True)
        # enforce max length
        while len(hist) > HISTOGRAM_SIZE:
            hist.pop(0)
            changed = True
        return changed

    def calculate_velocity(self, label, dir=0):
        """
        Pixel-space velocity between oldest & newest in histogram.
        dir=0 returns scalar dist, dir=1 returns [dx,dy] vector
        Uses the MIDDLE_MCP_IDX point from stored pixel pts.
        """
        hist = self.position_histogram[label]
        # need at least 2 entries to compute velocity
        if len(hist) >= 2:
            # use the oldest and newest entries
            start_entry = hist[0]
            end_entry   = hist[-1]
            # ensure both entries have enough points
            try:
                start = start_entry["pts"][MIDDLE_MCP_IDX]
                end   = end_entry["pts"][MIDDLE_MCP_IDX]
            except Exception:
                return [0,0] if dir else 0

            # normalize by number of frames between them to get per-frame velocity
            frames = max(1, len(hist) - 1)
            dx = (end[0] - start[0]) / frames
            dy = (end[1] - start[1]) / frames
            return [dx, dy] if dir else math.hypot(dx, dy)
        return [0,0] if dir else 0

    def generate_frames(self, velocity, label):
        """
        Create a ghost HandLandmarks from last pixel positions + vel.
        Uses the last real entry as the base to avoid ghost-feedback loops.
        Returns an object with .landmark list where each landmark has .x and .y
        (keeps the same interface as MediaPipe landmarks for render_hands).
        """
        # find last real entry
        hist = self.position_histogram[label]
        base_pts = None
        for e in reversed(hist):
            if e.get("source") == "real" and e.get("pts"):
                base_pts = e["pts"]
                break
        if base_pts is None:
            # fallback to last entry if no real exists
            base_pts = hist[-1]["pts"] if hist and hist[-1].get("pts") else []

        class LM:
            def __init__(self,x,y):
                # store as pixel coords (not normalized)
                self.x = float(x)
                self.y = float(y)
        class HL:
            def __init__(self):
                self.landmark = []
            def add(self,x,y):
                self.landmark.append(LM(x,y))

        gen = HL()
        for x,y in base_pts:
            gen.add(x + velocity[0], y + velocity[1])
        return gen

    def cvimage_to_pygame(self, image):
        """Convert cv2 image into a pygame surface"""
        # Get the image dimensions
        size = image.shape[1::-1]
        # Create a Pygame surface from the numpy array
        pygame_surface = pygame.image.frombuffer(image.tobytes(), size, "RGB")
        return pygame_surface

    def render(self, surf):
        self.frame_count += 1
        ar_data = {
            "POSITION_DATA": {"LEFT": [], "RIGHT": []},
            "SCALE":         {"LEFT": 1,    "RIGHT": 1},
            "CLICK_DIST":    {"LEFT": 0,    "RIGHT": 0},
            "CLICK_FLAG":    {"LEFT": False,"RIGHT": False},
            "HAND_PRESENCE" : False
        }

        # quick camera sanity prints
        get_logger_info('DEBUG', f"[AR] cap.isOpened: {self.cap.isOpened()}")

        ret, frame = self.cap.read()
        if not ret:
            get_logger_info('DEBUG', "[AR] cap.read failed")
            return ar_data
        get_logger_info('DEBUG', f"[AR] frame.shape: {getattr(frame, 'shape', None)}")

        # prep for Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        image = self.cvimage_to_pygame(rgb)
        surf.blit(image, (surf.get_width() - image.get_width(), 0))
        # diagnostic prints
        get_logger_info('DEBUG', f"[AR] FRAME {self.frame_count} Mediapipe hands: {bool(res.multi_hand_landmarks)}")
        if res.multi_hand_landmarks:
            for i, lm_set in enumerate(res.multi_hand_landmarks):
                valid_count = sum(1 for lm in lm_set.landmark if self._valid_landmark(lm))
                get_logger_info('DEBUG', f"[AR]  hand {i} valid landmarks: {valid_count}")
        get_logger_info('DEBUG', f"[AR] presence_counter: {self.presence_counter} hands_tracker: {self.hands_tracker}")

        # keep track of which hands appear
        seen = []

        # decrement ghost TTLs each frame and prune expired ghosts
        for label in ("LEFT", "RIGHT"):
            self._decrement_and_prune_ghosts(label)

        if res.multi_hand_landmarks:
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
                self.render_hands(surf, lm_set, label, is_generated=False)

                # 3) fill AR output
                # position data should be pixel-space pts from the last real entry
                hist = self.position_histogram[label]
                if hist:
                    # prefer most recent real pts
                    last_real = None
                    for e in reversed(hist):
                        if e.get("source") == "real" and e.get("pts"):
                            last_real = e["pts"]
                            break
                    # fallback to last entry even if ghost or partial
                    if last_real is None:
                        last_real = hist[-1]["pts"]
                    ar_data["POSITION_DATA"][label] = last_real
                else:
                    # keep previous ar_data value (avoid overwriting with empty)
                    pass

                ar_data["SCALE"][label]         = d["scale"]
                ar_data["CLICK_DIST"][label]    = d["rel_dist"]
                ar_data["CLICK_FLAG"][label]    = d["is_pinched"]

                # determine HAND_PRESENCE using presence_counter hysteresis
                if self.presence_counter[label] >= self.presence_threshold_on:
                    ar_data["HAND_PRESENCE"] = True

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
                            get_logger_info('CORE', f'CLEARING HISTOGRAM FOR {label} DUE TO LONG ABSENCE {self.hands_tracker[label]}', True)
                            self.position_histogram[label].clear()
                        # reset pinch detector state for this hand to avoid sticky pinches
                        try:
                            self.detector.reset(label)
                        except Exception:
                            hs = self.detector.hands[label]
                            self.detector.hands[label] = HandState(hs.pos_hist, 0, False)
                        # ensure AR output flags are cleared
                        ar_data["CLICK_FLAG"][label] = False
                        ar_data["CLICK_DIST"][label] = 0
                        ar_data["SCALE"][label] = 1

                    # If the hand has been missing for pinch_absent_reset frames, reset pinch to avoid sticky pinches
                    if self.hands_tracker[label] >= self.pinch_absent_reset:
                        try:
                            self.detector.reset(label)
                        except Exception:
                            hs = self.detector.hands[label]
                            self.detector.hands[label] = HandState(hs.pos_hist, 0, False)
                        ar_data["CLICK_FLAG"][label] = False

                    # only generate ghosts if we have at least one real base and haven't exceeded lifetime
                    if (len(self.position_histogram[label]) >= 1
                       and self.hands_tracker[label] < HISTOGRAM_SIZE):
                        vel = self.calculate_velocity(label, dir=1)
                        ghost = self.generate_frames(vel, label)
                        # render ghost frames and mark as generated
                        self.render_hands(surf, ghost, label, is_generated=True)
                        get_logger_info('CORE',
                            f'GENERATED HAND FRAMES FOR {label} HAND_TRACKER={self.hands_tracker[label]}', True)
                    else:
                        # if we've been absent for a long time, ensure histogram is small
                        if self.hands_tracker[label] >= self.absent_reset_threshold:
                            if len(self.position_histogram[label]) > 0:
                                get_logger_info('CORE', f'CLEARING HISTOGRAM FOR {label} DUE TO LONG ABSENCE {self.hands_tracker[label]}', True)
                                self.position_histogram[label].clear()

        else:
            get_logger_info('ERROR', 'NO HANDS DETECTED', True)
            for label in ("LEFT","RIGHT"):
                self.hands_tracker[label] += 1
                # decrement presence counter on missing frames
                self.presence_counter[label] = max(self.presence_threshold_off, self.presence_counter[label] - 1)

                # If the hand has been missing for a while, reset pinch detector state and clear history
                if self.hands_tracker[label] >= self.absent_reset_threshold:
                    if len(self.position_histogram[label]) > 0:
                        get_logger_info('CORE', f'CLEARING HISTOGRAM FOR {label} DUE TO LONG ABSENCE {self.hands_tracker[label]}', True)
                        self.position_histogram[label].clear()
                    try:
                        self.detector.reset(label)
                    except Exception:
                        hs = self.detector.hands[label]
                        self.detector.hands[label] = HandState(hs.pos_hist, 0, False)
                    ar_data["CLICK_FLAG"][label] = False
                    ar_data["CLICK_DIST"][label] = 0
                    ar_data["SCALE"][label] = 1

                # if we have recent history and haven't exceeded ghost lifetime, generate ghosts
                if (len(self.position_histogram[label]) >= 1
                       and self.hands_tracker[label] < HISTOGRAM_SIZE):
                    vel = self.calculate_velocity(label, dir=1)
                    ghost = self.generate_frames(vel, label)
                    self.render_hands(surf, ghost, label, is_generated=True)
                else:
                    ar_data["HAND_PRESENCE"] = False
                    # no hands detected at all

        # Fallback visual smoke test
        # If nothing is being drawn by render_hands, draw a visible fallback marker for each hand
        for label in ("LEFT", "RIGHT"):
            hist = self.position_histogram[label]
            if hist:
                last = None
                for e in reversed(hist):
                    if e.get("pts"):
                        last = e["pts"]
                        break
                if last:
                    try:
                        color = (200, 80, 80) if label == "LEFT" else (80, 80, 200)
                        p = last[WRIST_IDX] if len(last) > WRIST_IDX else last[0]
                        pygame.draw.circle(surf, color, p, 10, 2)
                        font = pygame.font.SysFont("Arial", 14)
                        txt = font.render(f"{label} hist:{len(hist)}", True, color)
                        surf.blit(txt, (max(0, p[0]-20), max(0, p[1]-30)))
                    except Exception as e:
                        get_logger_info('CORE', f"[AR] fallback draw error {e}")

        return ar_data
