# game_main.py
from scripts.config import *
from scripts.ar import AR
import pygame
import sys
import math
import time

# TUNABLES
PLACEMENT_COOLDOWN = 0.25   # seconds between automatic placements while holding left pinch
PLACEMENT_MIN_MOVE = 4.0    # pixels: require this much fingertip movement (screen space) to consider a new placement
PINCH_STABLE_TIME = 0.06    # seconds: require pinch to be stable for this long to count as active
MIN_SCREEN_BLOCK_SIZE = 2   # pixels: don't draw blocks smaller than this (avoids visual clutter/overlap at extreme zoom)
ZOOM_MIN = 0.25
ZOOM_MAX = 4.0

# New: block visual scale multiplier (makes blocks appear larger)
BLOCK_VISUAL_SCALE = 1.4

# Small EMA smoothing for fingertip to reduce single-frame misses/stutter
FINGER_EMA_ALPHA = 0.45

# Helper small classes for actions and camera

class HandActionState:
    """
    Tracks pinch rising/falling edges and stores start positions for actions.
    start_pos is in world coordinates (x,y).
    Adds a small stable-time debounce for pinch recognition.
    """
    def __init__(self):
        self.prev_pinched = False
        self.pinched_since = None
        self.start_pos = None
        self.stable = False

    def update(self, is_pinched, pos_world, t_now):
        """
        Returns (rising, falling, stable_state)
        rising/falling are edges after debounce; stable_state is True while pinch is considered stable.
        """
        rising = False
        falling = False

        if is_pinched:
            if not self.prev_pinched:
                # just observed pinch; start timer
                self.pinched_since = t_now
                self.prev_pinched = True
                self.stable = False
            else:
                # still pinched; check stable time
                if not self.stable and self.pinched_since is not None and (t_now - self.pinched_since) >= PINCH_STABLE_TIME:
                    # now stable: treat as rising (debounced)
                    rising = True
                    self.stable = True
                    self.start_pos = pos_world
        else:
            # not pinched now
            if self.prev_pinched:
                # was pinched previously; falling edge
                if self.stable:
                    falling = True
                # reset
                self.prev_pinched = False
                self.pinched_since = None
                self.start_pos = None
                self.stable = False

        return rising, falling, self.stable

class Camera:
    """
    Simple 2D camera with pan, zoom, rotation.
    screen_to_world and world_to_screen handle transforms.
    """
    def __init__(self, pos=(0.0,0.0), zoom=1.0, angle=0.0):
        self.pos = [float(pos[0]), float(pos[1])]
        self.zoom = float(zoom)
        self.angle = float(angle)

    def screen_to_world(self, sx, sy, screen_w, screen_h):
        cx, cy = screen_w/2.0, screen_h/2.0
        dx = (sx - cx) / self.zoom
        dy = (sy - cy) / self.zoom
        # inverse rotation
        cos_a = math.cos(-self.angle)
        sin_a = math.sin(-self.angle)
        wx = cos_a*dx - sin_a*dy + self.pos[0]
        wy = sin_a*dx + cos_a*dy + self.pos[1]
        return (wx, wy)

    def world_to_screen(self, wx, wy, screen_w, screen_h):
        cx, cy = screen_w/2.0, screen_h/2.0
        dx = wx - self.pos[0]
        dy = wy - self.pos[1]
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        sx = cos_a*dx - sin_a*dy
        sy = sin_a*dx + cos_a*dy
        return (int(sx*self.zoom + cx), int(sy*self.zoom + cy))

    def clamp_zoom(self, min_z=ZOOM_MIN, max_z=ZOOM_MAX):
        self.zoom = max(min_z, min(max_z, self.zoom))

# Simple block world
class BlockWorld:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        self.blocks = set()  # store (gx, gy) integer grid coords

    def world_to_grid(self, wx, wy):
        gx = int(round(wx / self.grid_size))
        gy = int(round(wy / self.grid_size))
        return (gx, gy)

    def add_block_at_world(self, wx, wy):
        """
        Add block snapped to grid. Returns True if a new block was added, False if already present.
        """
        gx, gy = self.world_to_grid(wx, wy)
        if (gx, gy) in self.blocks:
            return False
        self.blocks.add((gx, gy))
        return True

    def draw(self, surf, camera, screen_size):
        sw, sh = screen_size
        for (gx, gy) in self.blocks:
            wx = gx * self.grid_size
            wy = gy * self.grid_size
            sx, sy = camera.world_to_screen(wx, wy, sw, sh)
            size = int(round(self.grid_size * camera.zoom * BLOCK_VISUAL_SCALE))
            if size < MIN_SCREEN_BLOCK_SIZE:
                # skip drawing extremely small blocks to avoid visual clutter/overlap
                continue
            rect = pygame.Rect(sx - size//2, sy - size//2, size, size)
            pygame.draw.rect(surf, (50,200,50), rect)
            pygame.draw.rect(surf, (0,0,0), rect, 1)

# Main graphics engine + game logic
class GraphicsEngine2D:

    def __init__(self, win_size=(1280, 720)):
        pygame.init()
        self.ar = AR()
        self.win_size = win_size
        self.screen = pygame.display.set_mode(win_size)
        pygame.display.set_caption("AR Block Builder")
        self.clock = pygame.time.Clock()
        self.delta_time = 0.0

        self.block_world = BlockWorld(grid_size=BLOCK_SIZE)
        self.camera = Camera(pos=(0.0, 0.0), zoom=1.0, angle=0.0)

        # per-hand action states
        self.left_state = HandActionState()
        self.right_state = HandActionState()

        # both-hand gesture state
        self.both_pinched = False
        self.both_initial = None  # (dist, zoom, angle)

        # placement control
        self.left_last_place_time = 0.0
        self.left_last_place_cell = None
        self.left_last_place_px = None

        # fingertip EMA state to reduce single-frame misses
        self.left_finger_ema = None
        self.right_finger_ema = None

        # debug
        self.debug = True

    def euclid(self, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.hypot(dx, dy)

    def _ema_update(self, prev, value, alpha=FINGER_EMA_ALPHA):
        if prev is None:
            return value
        return (alpha * value[0] + (1 - alpha) * prev[0],
                alpha * value[1] + (1 - alpha) * prev[1])

    def run(self):
        while True:
            self.screen.fill((30, 30, 40))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        self.debug = not self.debug

            t_now = time.perf_counter()
            data = self.ar.render(self.screen)

            # extract positions and pinch flags
            left_pts = data["POSITION_DATA"].get("LEFT", [])
            right_pts = data["POSITION_DATA"].get("RIGHT", [])
            left_pinched = data["CLICK_FLAG"].get("LEFT", False)
            right_pinched = data["CLICK_FLAG"].get("RIGHT", False)

            # choose fingertip indices for actions (use INDEX_TIP for left placement)
            left_tip_px = None
            right_tip_px = None
            if left_pts and len(left_pts) > INDEX_TIP_IDX:
                left_tip_px = left_pts[INDEX_TIP_IDX]
            if right_pts and len(right_pts) > INDEX_TIP_IDX:
                right_tip_px = right_pts[INDEX_TIP_IDX]

            # apply small EMA smoothing to fingertip pixel coords to reduce single-frame misses/stutter
            if left_tip_px:
                self.left_finger_ema = self._ema_update(self.left_finger_ema, left_tip_px)
                left_tip_px_smoothed = (int(round(self.left_finger_ema[0])), int(round(self.left_finger_ema[1])))
            else:
                left_tip_px_smoothed = None
                self.left_finger_ema = None

            if right_tip_px:
                self.right_finger_ema = self._ema_update(self.right_finger_ema, right_tip_px)
                right_tip_px_smoothed = (int(round(self.right_finger_ema[0])), int(round(self.right_finger_ema[1])))
            else:
                right_tip_px_smoothed = None
                self.right_finger_ema = None

            # convert to world coords if available (use smoothed px)
            left_world = None
            right_world = None
            if left_tip_px_smoothed:
                left_world = self.camera.screen_to_world(left_tip_px_smoothed[0], left_tip_px_smoothed[1], *self.win_size)
            if right_tip_px_smoothed:
                right_world = self.camera.screen_to_world(right_tip_px_smoothed[0], right_tip_px_smoothed[1], *self.win_size)

            # update per-hand action states (with debounce)
            l_rising, l_falling, l_stable = self.left_state.update(left_pinched, left_world, t_now)
            r_rising, r_falling, r_stable = self.right_state.update(right_pinched, right_world, t_now)

            # BOTH pinch handling (zoom/rotate) - when both hands are stable pinched
            if l_stable and r_stable:
                # entering both-pinched
                if not self.both_pinched:
                    self.both_pinched = True
                    # store initial distance and camera state
                    if left_tip_px_smoothed and right_tip_px_smoothed:
                        dist = self.euclid(left_tip_px_smoothed, right_tip_px_smoothed)
                        self.both_initial = {
                            "dist": dist,
                            "zoom": self.camera.zoom,
                            "angle": self.camera.angle,
                            "left_px": left_tip_px_smoothed,
                            "right_px": right_tip_px_smoothed
                        }
                else:
                    # update zoom and rotation based on current distance and angle
                    if self.both_initial and left_tip_px_smoothed and right_tip_px_smoothed:
                        cur_dist = self.euclid(left_tip_px_smoothed, right_tip_px_smoothed)
                        if self.both_initial["dist"] > 1e-6:
                            scale = cur_dist / self.both_initial["dist"]
                            self.camera.zoom = self.both_initial["zoom"] * scale
                            self.camera.clamp_zoom()
                        # compute angle between hands and rotate camera relative to initial
                        lx, ly = left_tip_px_smoothed
                        rx, ry = right_tip_px_smoothed
                        cur_angle = math.atan2(ry - ly, rx - lx)
                        init_lx, init_ly = self.both_initial["left_px"]
                        init_rx, init_ry = self.both_initial["right_px"]
                        init_angle = math.atan2(init_ry - init_ly, init_rx - init_lx)
                        delta_angle = cur_angle - init_angle
                        self.camera.angle = self.both_initial["angle"] + delta_angle
            else:
                # leaving both-pinched
                if self.both_pinched:
                    self.both_pinched = False
                    self.both_initial = None

            # LEFT hand placement logic
            # - allow continuous placement while left is stable pinched
            # - do NOT place while both hands are pinched (zoom/rotate mode)
            if l_stable and not (l_stable and r_stable):
                # only attempt placement if we have a valid fingertip and world pos
                if left_world:
                    # compute grid cell
                    gx, gy = self.block_world.world_to_grid(left_world[0], left_world[1])
                    current_cell = (gx, gy)
                    # compute screen movement since last placement to avoid placing many blocks in same cell due to jitter
                    place_allowed = False
                    # time-based cooldown
                    if (t_now - self.left_last_place_time) >= PLACEMENT_COOLDOWN:
                        # movement-based check: if last placed cell is different, allow
                        if self.left_last_place_cell != current_cell:
                            place_allowed = True
                        else:
                            # if same cell but finger moved enough in screen space, allow (rare)
                            if self.left_last_place_px and left_tip_px_smoothed:
                                move_px = self.euclid(self.left_last_place_px, left_tip_px_smoothed)
                                if move_px >= PLACEMENT_MIN_MOVE:
                                    place_allowed = True
                    if place_allowed:
                        added = self.block_world.add_block_at_world(left_world[0], left_world[1])
                        if added:
                            get_logger_info('GAME', f'Placed block at grid {current_cell} world {left_world}', True)
                        else:
                            get_logger_info('GAME', f'Block already present at grid {current_cell}', False)
                        self.left_last_place_time = t_now
                        self.left_last_place_cell = current_cell
                        self.left_last_place_px = left_tip_px_smoothed
            else:
                # reset last place px when not pinched to avoid stale movement checks
                self.left_last_place_px = None

            # RIGHT hand pinch -> pan camera while held (only when not both-pinched)
            if r_stable and not (l_stable and r_stable) and right_world and self.right_state.start_pos:
                # compute delta in world coords between start_pos and current right_world
                sx, sy = self.right_state.start_pos
                cx, cy = right_world
                dx = sx - cx
                dy = sy - cy
                # apply pan (drag world)
                self.camera.pos[0] += dx
                self.camera.pos[1] += dy
                # update start_pos so panning is continuous and smooth
                self.right_state.start_pos = right_world

            # RIGHT falling -> log
            if r_falling:
                get_logger_info('GAME', f'Right pinch released; camera pos {self.camera.pos}', True)

            # draw world and UI
            self.block_world.draw(self.screen, self.camera, self.win_size)

            # debug overlay: draw hand cursors, pinch states, camera info
            if self.debug:
                # draw left cursor (use smoothed px)
                if left_tip_px_smoothed:
                    pygame.draw.circle(self.screen, (255, 100, 100), left_tip_px_smoothed, 8, 2)
                    if left_pinched:
                        pygame.draw.circle(self.screen, (255, 50, 50), left_tip_px_smoothed, 6)
                # draw right cursor
                if right_tip_px_smoothed:
                    pygame.draw.circle(self.screen, (100, 100, 255), right_tip_px_smoothed, 8, 2)
                    if right_pinched:
                        pygame.draw.circle(self.screen, (50, 50, 255), right_tip_px_smoothed, 6)

                # HUD text
                font = pygame.font.SysFont("Arial", 16)
                lines = [
                    f"FPS: {int(self.clock.get_fps())}",
                    f"Camera pos: ({self.camera.pos[0]:.1f}, {self.camera.pos[1]:.1f})",
                    f"Zoom: {self.camera.zoom:.2f} Angle: {math.degrees(self.camera.angle):.1f}",
                    f"Left pinched (stable): {l_stable} Right pinched (stable): {r_stable}",
                    f"Blocks: {len(self.block_world.blocks)}",
                    f"Left last cell: {self.left_last_place_cell}",
                    "Press D to toggle debug"
                ]
                for i, line in enumerate(lines):
                    surf_text = font.render(line, True, (220,220,220))
                    self.screen.blit(surf_text, (10, 10 + i*18))

            pygame.display.update()
            # cap to 60 FPS
            self.delta_time = self.clock.tick(60) / 1000.0  # seconds

if __name__ == "__main__":
    GraphicsEngine2D().run()
