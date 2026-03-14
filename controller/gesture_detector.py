# controller/gesture_detector.py
from abc import ABC, abstractmethod
import math
from configs.arconfig import (
    THUMB_TIP_IDX, INDEX_TIP_IDX, INDEX_MCP_IDX, MIDDLE_MCP_IDX,
    PINKY_MCP_IDX, WRIST_IDX, PINCH_ON_THRESH, PINCH_OFF_THRESH,
    PINCH_FRAMES_REQ
)

# Additional indices not in arconfig
MIDDLE_TIP_IDX = 12
RING_TIP_IDX = 16
RING_MCP_IDX = 13
PINKY_TIP_IDX = 20
PINKY_MCP_IDX = 17
THUMB_MCP_IDX = 2

class GestureEvent:
    def __init__(self, hand, gesture_name, event_type, value=None):
        self.hand = hand
        self.gesture_name = gesture_name
        self.event_type = event_type
        self.value = value

    def __repr__(self):
        return f"GestureEvent({self.hand}, {self.gesture_name}, {self.event_type}, {self.value})"


class BaseGestureDetector(ABC):
    def __init__(self, hand_label, gesture_name, hold_frames=None):
        self.hand = hand_label
        self.gesture_name = gesture_name
        self.hold_frames = hold_frames
        self.active = False
        self._raw_active = False
        self._count = 0
        self._active_frames = 0
        self._hold_emitted = False
        self._start_time = None
        self._value = None

    @abstractmethod
    def _compute_raw_state(self, landmarks):
        pass

    def process(self, landmarks, current_time):
        events = []
        raw_active, value = self._compute_raw_state(landmarks)

        if raw_active:
            self._count += 1
        else:
            self._count -= 1

        self._count = max(-PINCH_FRAMES_REQ, min(PINCH_FRAMES_REQ, self._count))
        new_active = self.active

        if not self.active and self._count >= PINCH_FRAMES_REQ:
            new_active = True
        elif self.active and self._count <= -PINCH_FRAMES_REQ:
            new_active = False

        if new_active != self.active:
            if new_active:
                self._active_frames = 0
                self._hold_emitted = False
                self._start_time = current_time
                events.append(GestureEvent(self.hand, self.gesture_name, 'START', value))
            else:
                events.append(GestureEvent(self.hand, self.gesture_name, 'END', self._value))
            self.active = new_active

        if self.active:
            self._active_frames += 1
            self._value = value
            events.append(GestureEvent(self.hand, self.gesture_name, 'UPDATE', value))

            if self.hold_frames and self._active_frames >= self.hold_frames and not self._hold_emitted:
                events.append(GestureEvent(self.hand, self.gesture_name, 'HOLD', value))
                self._hold_emitted = True

        return events


class PinchDetector(BaseGestureDetector):
    def __init__(self, hand_label, hold_frames=None):
        super().__init__(hand_label, 'pinch', hold_frames)
        self.on_thresh = PINCH_ON_THRESH
        self.off_thresh = PINCH_OFF_THRESH

    def _compute_raw_state(self, landmarks):
        if len(landmarks) < 21:
            return False, None

        p_thumb = landmarks[THUMB_TIP_IDX]
        p_index = landmarks[INDEX_TIP_IDX]
        p_wrist = landmarks[WRIST_IDX]
        p_mcp   = landmarks[MIDDLE_MCP_IDX]

        dx = p_index[0] - p_thumb[0]
        dy = p_index[1] - p_thumb[1]
        dz = p_index[2] - p_thumb[2]
        raw_dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        sx = p_mcp[0] - p_wrist[0]
        sy = p_mcp[1] - p_wrist[1]
        sz = p_mcp[2] - p_wrist[2]
        hand_scale = math.sqrt(sx*sx + sy*sy + sz*sz) or 1.0

        rel_dist = raw_dist / hand_scale

        raw_active = rel_dist < self.on_thresh if not self.active else rel_dist < self.off_thresh
        return raw_active, rel_dist


class TwoFingerUpDetector(BaseGestureDetector):
    def __init__(self, hand_label, hold_frames=None, require_others_down=True):
        super().__init__(hand_label, 'two_finger_up', hold_frames)
        self.require_others_down = require_others_down

    def _compute_raw_state(self, landmarks):
        if len(landmarks) < 21:
            return False, None

        index_tip = landmarks[INDEX_TIP_IDX]
        index_mcp = landmarks[INDEX_MCP_IDX]
        middle_tip = landmarks[MIDDLE_TIP_IDX]
        middle_mcp = landmarks[MIDDLE_MCP_IDX]
        wrist = landmarks[WRIST_IDX]
        thumb_tip = landmarks[THUMB_TIP_IDX]

        # Condition 1: index and middle tips above their MCPs and above wrist
        index_up = index_tip[1] < index_mcp[1] and index_tip[1] < wrist[1]
        middle_up = middle_tip[1] < middle_mcp[1] and middle_tip[1] < wrist[1]

        if not (index_up and middle_up):
            return False, None

        # Condition 2: ring and pinky down (optional)
        if self.require_others_down:
            ring_tip = landmarks[RING_TIP_IDX]
            ring_mcp = landmarks[RING_MCP_IDX]
            pinky_tip = landmarks[PINKY_TIP_IDX]
            pinky_mcp = landmarks[PINKY_MCP_IDX]

            ring_down = ring_tip[1] >= ring_mcp[1]
            pinky_down = pinky_tip[1] >= pinky_mcp[1]
            if not (ring_down and pinky_down):
                return False, None

        # Condition 3: exclude pinch (index and thumb not too close)
        dx = index_tip[0] - thumb_tip[0]
        dy = index_tip[1] - thumb_tip[1]
        dz = index_tip[2] - thumb_tip[2]
        pinch_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        sx = middle_mcp[0] - wrist[0]
        sy = middle_mcp[1] - wrist[1]
        sz = middle_mcp[2] - wrist[2]
        hand_scale = math.sqrt(sx*sx + sy*sy + sz*sz) or 1.0
        rel_pinch_dist = pinch_dist / hand_scale
        if rel_pinch_dist < 0.4:
            return False, None

        avg_x = (index_tip[0] + middle_tip[0]) / 2.0
        avg_y = (index_tip[1] + middle_tip[1]) / 2.0
        return True, (avg_x, avg_y)


class OpenPalmDetector(BaseGestureDetector):
    def __init__(self, hand_label, hold_frames=None):
        super().__init__(hand_label, 'open_palm', hold_frames)

    def _compute_raw_state(self, landmarks):
        if len(landmarks) < 21:
            return False, None

        wrist = landmarks[WRIST_IDX]
        # All five finger tips above their respective MCPs and above wrist
        tips = [THUMB_TIP_IDX, INDEX_TIP_IDX, MIDDLE_TIP_IDX, RING_TIP_IDX, PINKY_TIP_IDX]
        mcps = [THUMB_MCP_IDX, INDEX_MCP_IDX, MIDDLE_MCP_IDX, RING_MCP_IDX, PINKY_MCP_IDX]

        for tip_idx, mcp_idx in zip(tips, mcps):
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            if tip[1] >= mcp[1] or tip[1] >= wrist[1]:
                return False, None

        # Optionally check thumb not too close to index to exclude pinch
        thumb_tip = landmarks[THUMB_TIP_IDX]
        index_tip = landmarks[INDEX_TIP_IDX]
        dx = thumb_tip[0] - index_tip[0]
        dy = thumb_tip[1] - index_tip[1]
        dz = thumb_tip[2] - index_tip[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        sx = landmarks[MIDDLE_MCP_IDX][0] - wrist[0]
        sy = landmarks[MIDDLE_MCP_IDX][1] - wrist[1]
        sz = landmarks[MIDDLE_MCP_IDX][2] - wrist[2]
        hand_scale = math.sqrt(sx*sx + sy*sy + sz*sz) or 1.0
        rel_dist = dist / hand_scale
        if rel_dist < 0.4:
            return False, None

        return True, None


class PointDetector(BaseGestureDetector):
    def __init__(self, hand_label, hold_frames=None):
        super().__init__(hand_label, 'point', hold_frames)

    def _compute_raw_state(self, landmarks):
        if len(landmarks) < 21:
            return False, None

        index_tip = landmarks[INDEX_TIP_IDX]
        index_mcp = landmarks[INDEX_MCP_IDX]
        wrist = landmarks[WRIST_IDX]

        # Index extended (tip above MCP and above wrist)
        index_up = index_tip[1] < index_mcp[1] and index_tip[1] < wrist[1]

        # All other fingers not extended (optional, but helps reduce false positives)
        other_tips = [MIDDLE_TIP_IDX, RING_TIP_IDX, PINKY_TIP_IDX]
        others_down = True
        for tip_idx in other_tips:
            tip = landmarks[tip_idx]
            mcp = landmarks[tip_idx - 3]  # approximate MCP index offset
            if tip[1] < mcp[1]:
                others_down = False
                break

        return index_up and others_down, index_tip


class FistDetector(BaseGestureDetector):
    def __init__(self, hand_label, hold_frames=None):
        super().__init__(hand_label, 'fist', hold_frames)

    def _compute_raw_state(self, landmarks):
        if len(landmarks) < 21:
            return False, None

        wrist = landmarks[WRIST_IDX]
        finger_indices = [
            (THUMB_TIP_IDX, THUMB_MCP_IDX),
            (INDEX_TIP_IDX, INDEX_MCP_IDX),
            (MIDDLE_TIP_IDX, MIDDLE_MCP_IDX),
            (RING_TIP_IDX, RING_MCP_IDX),
            (PINKY_TIP_IDX, PINKY_MCP_IDX)
        ]

        # All fingertips below (greater y) their MCP and below wrist
        for tip_idx, mcp_idx in finger_indices:
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            if tip[1] <= mcp[1] or tip[1] <= wrist[1]:   # not curled enough
                return False, None

        return True, None


class GestureManager:
    def __init__(self):
        self.detectors = {'LEFT': [], 'RIGHT': []}

    def add_detector(self, detector):
        self.detectors[detector.hand].append(detector)

    def process_hand(self, hand_label, landmarks, current_time):
        events = []
        for det in self.detectors.get(hand_label, []):
            events.extend(det.process(landmarks, current_time))
        return events

    def process_both(self, left_landmarks, right_landmarks, current_time):
        events = []
        if left_landmarks:
            events.extend(self.process_hand('LEFT', left_landmarks, current_time))
        if right_landmarks:
            events.extend(self.process_hand('RIGHT', right_landmarks, current_time))
        return events