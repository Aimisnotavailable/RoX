# src/handstateaction.py
from scripts.config import PINCH_STABLE_TIME

class HandActionState:
    """
    Tracks the state of a single hand (Left or Right).
    Handles debouncing (stability checks) and movement deltas.
    """
    def __init__(self):
        # Is the pinch currently active and stable?
        self.active = False
        
        # Screen coordinates where the current stable pinch started
        self.start_pos = None
        
        # Current screen coordinates (smoothed)
        self.current_pos = None

        # Internal state for debouncing
        self._raw_pinch_start_time = None
        self._is_raw_pinched = False

    def update(self, is_pinched, current_pos, current_time):
        """
        Updates the state based on raw pinch data and smoothed position.
        """
        self.current_pos = current_pos

        if is_pinched:
            # If this is a new pinch, mark the start time
            if not self._is_raw_pinched:
                self._is_raw_pinched = True
                self._raw_pinch_start_time = current_time

            # If we are not yet 'active', check if we have held it long enough
            if not self.active:
                if (current_time - self._raw_pinch_start_time) >= PINCH_STABLE_TIME:
                    self.active = True
                    self.start_pos = current_pos
        else:
            # Pinch released
            self._is_raw_pinched = False
            self._raw_pinch_start_time = None
            self.active = False
            self.start_pos = None

    @property
    def drag_delta(self):
        """
        Returns (dx, dy) tuple of movement since the pinch started.
        Returns (0, 0) if not active.
        """
        if self.active and self.current_pos and self.start_pos:
            return (self.current_pos[0] - self.start_pos[0], 
                    self.current_pos[1] - self.start_pos[1])
        return (0, 0)