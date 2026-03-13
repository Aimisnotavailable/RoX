# gesture_detector.py
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

class GestureEvent:
    """Simple event container emitted by detectors."""
    def __init__(self, hand, gesture_name, event_type, value=None):
        self.hand = hand          # 'LEFT' or 'RIGHT'
        self.gesture_name = gesture_name  # e.g., 'pinch', 'two_finger_up'
        self.event_type = event_type      # 'START', 'UPDATE', 'END', 'HOLD'
        self.value = value                # optional data

    def __repr__(self):
        return f"GestureEvent({self.hand}, {self.gesture_name}, {self.event_type}, {self.value})"


class BaseGestureDetector(ABC):
    """Abstract base for all gesture detectors."""
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
        """
        Return (raw_active, value) based on current landmarks.
        raw_active: boolean indicating if gesture condition is met.
        value: any numeric data (e.g., pinch ratio) for UPDATE events.
        """
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
                events.append(GestureEvent(self.hand, self.gesture_name,
                                           'START', value))
            else:
                events.append(GestureEvent(self.hand, self.gesture_name,
                                           'END', self._value))
            self.active = new_active

        if self.active:
            self._active_frames += 1
            self._value = value
            events.append(GestureEvent(self.hand, self.gesture_name,
                                       'UPDATE', value))

            if self.hold_frames and self._active_frames >= self.hold_frames and not self._hold_emitted:
                events.append(GestureEvent(self.hand, self.gesture_name,
                                           'HOLD', value))
                self._hold_emitted = True

        return events


class PinchDetector(BaseGestureDetector):
    """Detects pinch gesture using thumb and index tips, normalized by hand scale."""
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
    """
    Detect when index and middle fingers are extended upward.
    Condition: both index and middle tips have y < their respective MCPs,
    and optionally ring and pinky tips are not extended (y > their MCPs).
    """
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

        index_up = index_tip[1] < index_mcp[1] and index_tip[1] < wrist[1]
        middle_up = middle_tip[1] < middle_mcp[1] and middle_tip[1] < wrist[1]

        if not (index_up and middle_up):
            return False, None

        if self.require_others_down:
            ring_tip = landmarks[RING_TIP_IDX]
            ring_mcp = landmarks[RING_MCP_IDX]
            pinky_tip = landmarks[PINKY_TIP_IDX]
            pinky_mcp = landmarks[PINKY_MCP_IDX]

            ring_down = ring_tip[1] >= ring_mcp[1]
            pinky_down = pinky_tip[1] >= pinky_mcp[1]
            if not (ring_down and pinky_down):
                return False, None

        avg_x = (index_tip[0] + middle_tip[0]) / 2.0
        avg_y = (index_tip[1] + middle_tip[1]) / 2.0
        return True, (avg_x, avg_y)


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