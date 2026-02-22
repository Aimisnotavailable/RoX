# Helper classes
from scripts.config import *

class HandActionState:
    def __init__(self):
        self.prev_pinched = False
        self.pinched_since = None
        self.start_pos = None
        self.start_screen = None
        self.start_cam = None
        self.start_zoom = None
        self.stable = False
        self._tentative_start = None

    def update(self, is_pinched, pos_world, screen_px, cam_pos, cam_zoom, t_now):
        rising = False
        falling = False

        if is_pinched:
            if not self.prev_pinched:
                self.pinched_since = t_now
                self.prev_pinched = True
                self.stable = False
                self._tentative_start = (pos_world, screen_px, (cam_pos[0], cam_pos[1]), cam_zoom)
            else:
                if not self.stable and self.pinched_since is not None and (t_now - self.pinched_since) >= PINCH_STABLE_TIME:
                    rising = True
                    self.stable = True
                    if self._tentative_start is not None:
                        self.start_pos, self.start_screen, self.start_cam, self.start_zoom = self._tentative_start
                        self._tentative_start = None
                    else:
                        self.start_pos = pos_world
                        self.start_screen = screen_px
                        self.start_cam = (cam_pos[0], cam_pos[1])
                        self.start_zoom = cam_zoom
        else:
            if self.prev_pinched:
                if self.stable:
                    falling = True
                self.prev_pinched = False
                self.pinched_since = None
                self.start_pos = None
                self.start_screen = None
                self.start_cam = None
                self.start_zoom = None
                self.stable = False
                self._tentative_start = None

        return rising, falling, self.stable