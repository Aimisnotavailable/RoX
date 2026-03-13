# gesture_detector.py
from abc import ABC, abstractmethod
import math
from controller.arconfig import (
    THUMB_TIP_IDX, INDEX_TIP_IDX, INDEX_MCP_IDX, MIDDLE_MCP_IDX,
    PINKY_MCP_IDX, WRIST_IDX, PINCH_ON_THRESH, PINCH_OFF_THRESH,
    PINCH_FRAMES_REQ
)

class GestureEvent:
    """Simple event container emitted by detectors."""
    def __init__(self, hand, gesture_name, event_type, value=None):
        self.hand = hand          # 'LEFT' or 'RIGHT'
        self.gesture_name = gesture_name  # e.g., 'pinch', 'two_finger_up'
        self.event_type = event_type      # 'START', 'UPDATE', 'END', 'HOLD'
        self.value = value                # optional data (distance, confidence)

    def __repr__(self):
        return f"GestureEvent({self.hand}, {self.gesture_name}, {self.event_type}, {self.value})"


class BaseGestureDetector(ABC):
    """Abstract base for all gesture detectors."""
    def __init__(self, hand_label, hold_frames=None):
        self.hand = hand_label
        self.hold_frames = hold_frames   # frames after which HOLD is emitted
        self.active = False               # current debounced state
        self._raw_active = False          # raw state before debounce
        self._count = 0                    # debounce counter
        self._active_frames = 0            # frames since activation (for hold)
        self._hold_emitted = False         # whether HOLD already sent
        self._start_time = None            # for time-based hold (optional)
        self._value = None                  # current gesture value (e.g., distance)

    @abstractmethod
    def _compute_raw_state(self, landmarks):
        """
        Return (raw_active, value) based on current landmarks.
        raw_active: boolean indicating if gesture condition is met.
        value: any numeric data (e.g., pinch ratio) for UPDATE events.
        """
        pass

    def process(self, landmarks, current_time):
        """
        Called each frame with landmarks (list of (x,y,z) tuples).
        Returns a list of GestureEvent(s) for this detector.
        """
        events = []
        raw_active, value = self._compute_raw_state(landmarks)

        # Debounce logic similar to original PinchDetector
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

        # Handle state transition
        if new_active != self.active:
            if new_active:
                # START
                self._active_frames = 0
                self._hold_emitted = False
                self._start_time = current_time
                events.append(GestureEvent(self.hand, self.__class__.__name__.lower(),
                                           'START', value))
            else:
                # END
                events.append(GestureEvent(self.hand, self.__class__.__name__.lower(),
                                           'END', self._value))
            self.active = new_active

        if self.active:
            self._active_frames += 1
            self._value = value
            # UPDATE (optional, every frame)
            events.append(GestureEvent(self.hand, self.__class__.__name__.lower(),
                                       'UPDATE', value))

            # HOLD if hold_frames defined and reached
            if self.hold_frames and self._active_frames >= self.hold_frames and not self._hold_emitted:
                events.append(GestureEvent(self.hand, self.__class__.__name__.lower(),
                                           'HOLD', value))
                self._hold_emitted = True

        return events


class PinchDetector(BaseGestureDetector):
    """Detects pinch gesture using thumb and index tips, normalized by hand scale."""
    def __init__(self, hand_label, hold_frames=None):
        super().__init__(hand_label, hold_frames)
        self.on_thresh = PINCH_ON_THRESH
        self.off_thresh = PINCH_OFF_THRESH

    def _compute_raw_state(self, landmarks):
        if len(landmarks) < 21:
            return False, None

        p_thumb = landmarks[THUMB_TIP_IDX]
        p_index = landmarks[INDEX_TIP_IDX]
        p_wrist = landmarks[WRIST_IDX]
        p_mcp   = landmarks[MIDDLE_MCP_IDX]

        # Euclidean distance in 3D
        dx = p_index[0] - p_thumb[0]
        dy = p_index[1] - p_thumb[1]
        dz = p_index[2] - p_thumb[2]
        raw_dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Hand scale (wrist to middle MCP)
        sx = p_mcp[0] - p_wrist[0]
        sy = p_mcp[1] - p_wrist[1]
        sz = p_mcp[2] - p_wrist[2]
        hand_scale = math.sqrt(sx*sx + sy*sy + sz*sz) or 1.0

        rel_dist = raw_dist / hand_scale

        # Raw active state uses hysteresis
        raw_active = rel_dist < self.on_thresh if not self.active else rel_dist < self.off_thresh
        return raw_active, rel_dist


class TwoFingerUpDetector(BaseGestureDetector):
    """
    Detect when index and middle fingers are extended upward.
    Condition: both tips have y coordinate higher (smaller) than their MCPs,
    and both are above wrist y.
    """
    def __init__(self, hand_label, hold_frames=None):
        super().__init__(hand_label, hold_frames)
        # Indices
        self.INDEX_TIP = INDEX_TIP_IDX
        self.INDEX_MCP = INDEX_MCP_IDX
        self.MIDDLE_TIP = 12          # MediaPipe index for middle finger tip
        self.MIDDLE_MCP = MIDDLE_MCP_IDX
        self.WRIST = WRIST_IDX

    def _compute_raw_state(self, landmarks):
        if len(landmarks) < 21:
            return False, None

        index_tip = landmarks[self.INDEX_TIP]
        index_mcp = landmarks[self.INDEX_MCP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        middle_mcp = landmarks[self.MIDDLE_MCP]
        wrist = landmarks[self.WRIST]

        # Note: MediaPipe y increases downward, so "up" means smaller y.
        index_up = index_tip[1] < index_mcp[1] and index_tip[1] < wrist[1]
        middle_up = middle_tip[1] < middle_mcp[1] and middle_tip[1] < wrist[1]
        raw_active = index_up and middle_up
        # Value: average height of the two tips
        value = (index_tip[1] + middle_tip[1]) / 2.0
        return raw_active, value


class GestureManager:
    """Manages multiple detectors per hand and collects events."""
    def __init__(self):
        self.detectors = {'LEFT': [], 'RIGHT': []}

    def add_detector(self, detector):
        """Add a detector instance (must have hand attribute)."""
        self.detectors[detector.hand].append(detector)

    def process_hand(self, hand_label, landmarks, current_time):
        """Process all detectors for one hand and return list of events."""
        events = []
        for det in self.detectors.get(hand_label, []):
            events.extend(det.process(landmarks, current_time))
        return events

    def process_both(self, left_landmarks, right_landmarks, current_time):
        """Process both hands and return combined events."""
        events = []
        if left_landmarks:
            events.extend(self.process_hand('LEFT', left_landmarks, current_time))
        if right_landmarks:
            events.extend(self.process_hand('RIGHT', right_landmarks, current_time))
        return events