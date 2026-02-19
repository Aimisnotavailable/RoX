from scripts.config import *

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

        # pixel‐space histogram for smoothing & ghost‐frames
        self.position_histogram = {'LEFT': [], 'RIGHT': []}
        self.hands_tracker     = {'LEFT': 0,      'RIGHT': 0     }

        # our new pinch detector
        self.detector = PinchDetector()

    def render_hands(self, surf, landmarks, label):
        """
        Draws landmarks→pixel & updates pixel histogram.
        """
        W, H = surf.get_width(), surf.get_height()
        pts = []
        for lm in landmarks.landmark:
            x_px = -lm.x * W + W
            y_px =  lm.y * H
            pts.append((x_px, y_px))
            pygame.draw.circle(surf, (255,255,255), (int(x_px),int(y_px)), 2)

        # maintain sliding window
        hist = self.position_histogram[label]
        if len(hist) < HISTOGRAM_SIZE:
            hist.append(pts)
        else:
            hist.pop(0)
            hist.append(pts)

        # draw connections
        for c in self.mp_hands.HAND_CONNECTIONS:
            pygame.draw.line(surf, (0,0,255),
                             pts[c[0]], pts[c[1]], 1)

    def calculate_velocity(self, label, dir=0):
        """
        Pixel‐space velocity between oldest & newest in histogram.
        dir=0 returns scalar dist, dir=1 returns [dx,dy] vector
        """
        hist = self.position_histogram[label]
        if len(hist) == HISTOGRAM_SIZE:
            start = hist[0][MIDDLE_MCP_IDX]
            end   = hist[-1][MIDDLE_MCP_IDX]
            dx = (end[0] - start[0]) / HISTOGRAM_SIZE
            dy = (end[1] - start[1]) / HISTOGRAM_SIZE
            return [dx, dy] if dir else math.hypot(dx, dy)
        return [0,0] if dir else 0

    def generate_frames(self, velocity, label):
        """
        Create a ghost HandLandmarks from last pixel positions + vel.
        """
        class LM: 
            def __init__(self,x,y): 
                self.x, self.y = x, y
        class HL: 
            def __init__(self): 
                self.landmark = []
            def add(self,x,y): 
                self.landmark.append(LM(x,y))

        gen = HL()
        for x,y in self.position_histogram[label][-1]:
            gen.add(x + velocity[0], y + velocity[1])
        return gen

    def render(self, surf):
        ar_data = {
            "POSITION_DATA": {"LEFT": [], "RIGHT": []},
            "SCALE":         {"LEFT": 1,    "RIGHT": 1},
            "CLICK_DIST":    {"LEFT": 0,    "RIGHT": 0},
            "CLICK_FLAG":    {"LEFT": False,"RIGHT": False},
            "HAND_PRESENCE" : False
        }

        ret, frame = self.cap.read()
        if not ret:
            return ar_data

        # prep for Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        # keep track of which hands appear
        seen = []

        if res.multi_hand_landmarks:
            for lm_set, handedness in zip(res.multi_hand_landmarks,
                                          res.multi_handedness):
                label = handedness.classification[0].label.upper()
                seen.append(label)

                # 1) pinch detection on normalized coords
                landmarks_norm = [(lm.x, lm.y) for lm in lm_set.landmark]
                d = self.detector.update(label, landmarks_norm)

                # 2) draw & update pixel histogram
                self.render_hands(surf, lm_set, label)

                # 3) fill AR output
                ar_data["POSITION_DATA"][label] = self.position_histogram[label][-1]
                ar_data["SCALE"][label]         = d["scale"]
                ar_data["CLICK_DIST"][label]    = d["rel_dist"]
                ar_data["CLICK_FLAG"][label]    = d["is_pinched"]
                ar_data["HAND_PRESENCE"]        = True

                self.hands_tracker[label] = 0

            # ghost frames for missing hands
            for label in ("LEFT","RIGHT"):
                if label not in seen:
                    self.hands_tracker[label] += 1
                    if (len(self.position_histogram[label]) >= 2
                       and self.hands_tracker[label] < HISTOGRAM_SIZE):
                        vel = self.calculate_velocity(label, dir=1)
                        ghost = self.generate_frames(vel, label)
                        self.render_hands(surf, ghost, label)
                        get_logger_info('CORE',
                            f'GENERATED HAND FRAMES FOR {label}', True)

        else:
            get_logger_info('ERROR', 'NO HANDS DETECTED', True)
            for label in ("LEFT","RIGHT"):
                if (len(self.position_histogram[label]) >= 2
                       and self.hands_tracker[label] < HISTOGRAM_SIZE):
                    vel = self.calculate_velocity(label, dir=1)
                    ghost = self.generate_frames(vel, label)
                    self.render_hands(surf, ghost, label)
                else:
                    ar_data["HAND_PRESENCE"] = False
                    # no hands detected at all

        return ar_data
