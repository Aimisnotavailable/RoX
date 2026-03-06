# src/arinput.py
import time
from scripts.arconfig import *
from src.configs.arinputconfig import *
from src.handstateaction import HandActionState

class ARInputHandler:
    def __init__(self, app):
        self.app = app
        
        # State machines for each hand
        self.left_state = HandActionState()
        self.right_state = HandActionState()

        # Smoothed Screen Positions (Exponential Moving Average)
        self.left_finger_ema = None
        self.right_finger_ema = None

    def _update_ema(self, current_ema, new_pos):
        """
        Applies exponential smoothing to reduce jitter.
        """
        if new_pos is None: 
            return None
        if current_ema is None: 
            return new_pos
            
        # FINGER_EMA_ALPHA should be defined in scripts.config (e.g., 0.5)
        alpha = FINGER_EMA_ALPHA
        x = alpha * new_pos[0] + (1 - alpha) * current_ema[0]
        y = alpha * new_pos[1] + (1 - alpha) * current_ema[1]
        return (x, y)

    def update(self, data):
        """
        Called every frame with raw AR data.
        Updates smoothing and hand states.
        """
        now = time.time()

        # 1. Extract Raw Data
        # Safely get positions and flags (default to None/False if missing)
        l_pts = data.get("POSITION_DATA", {}).get("LEFT", [])
        r_pts = data.get("POSITION_DATA", {}).get("RIGHT", [])
        
        l_pinched = data.get("CLICK_FLAG", {}).get("LEFT", False)
        r_pinched = data.get("CLICK_FLAG", {}).get("RIGHT", False)

        # 2. Extract Index Finger Tip (Index 8 in MediaPipe)
        l_raw = l_pts[INDEX_TIP_IDX] if (l_pts and len(l_pts) > INDEX_TIP_IDX) else None
        r_raw = r_pts[INDEX_TIP_IDX] if (r_pts and len(r_pts) > INDEX_TIP_IDX) else None

        # 3. Apply Smoothing (EMA)
        self.left_finger_ema = self._update_ema(self.left_finger_ema, l_raw)
        self.right_finger_ema = self._update_ema(self.right_finger_ema, r_raw)

        # 4. Update State Machines
        self.left_state.update(l_pinched, self.left_finger_ema, now)
        self.right_state.update(r_pinched, self.right_finger_ema, now)

        # Return self so engine can easily access .left_state / .right_state
        return self

    def get_pos(self, is_right_hand=True):
        """
        Helper: Returns smoothed (x, y) as integers for pygame drawing/logic.
        """
        pos = self.right_finger_ema if is_right_hand else self.left_finger_ema
        if pos:
            return (int(pos[0]), int(pos[1]))
        return None