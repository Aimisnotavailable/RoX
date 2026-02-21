# game_main.py
from scripts.config import *
from scripts.ar import AR
import pygame
import sys
import math
import time

# TUNABLES
PLACEMENT_COOLDOWN = 0.25
PLACEMENT_MIN_MOVE = 4.0
PINCH_STABLE_TIME = 0.06
MIN_SCREEN_BLOCK_SIZE = 2
ZOOM_MIN = 0.25
ZOOM_MAX = 4.0

# Block visuals
BLOCK_VISUAL_SCALE = 1.0
BLOCK_VISUAL_SCALE_ALLOW_OVERLAP = False

# Finger smoothing
FINGER_EMA_ALPHA = 0.45

# Camera pan interpolation
DEADZONE_PX = 6
SMOOTH_TAU = 0.06
INSTANT_STOP = False
MAX_STEP_WORLD = 10000.0

# Brightside lighting defaults
BRIGHTSIDE_DEFAULT = True
LIGHT_DIR = (-0.6, -0.4)        # screen-space light direction (x,y)
HIGHLIGHT_INTENSITY = 0.28
SHADOW_INTENSITY = 0.12

# Helper classes
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

class Camera:
    def __init__(self, pos=(0.0,0.0), zoom=1.0, angle=0.0):
        self.pos = [float(pos[0]), float(pos[1])]
        self.zoom = float(zoom)
        self.angle = float(angle)

    def screen_to_world(self, sx, sy, screen_w, screen_h):
        cx, cy = screen_w/2.0, screen_h/2.0
        dx = (sx - cx) / self.zoom
        dy = (sy - cy) / self.zoom
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

class BlockWorld:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        self.blocks = set()

    def world_to_grid(self, wx, wy):
        gx = int(round(wx / self.grid_size))
        gy = int(round(wy / self.grid_size))
        return (gx, gy)

    def add_block_at_world(self, wx, wy):
        gx, gy = self.world_to_grid(wx, wy)
        if (gx, gy) in self.blocks:
            return False
        self.blocks.add((gx, gy))
        return True

    @staticmethod
    def _shade_color(color, amount):
        r = max(0, min(255, int(round(color[0] * (1 + amount)))))
        g = max(0, min(255, int(round(color[1] * (1 + amount)))))
        b = max(0, min(255, int(round(color[2] * (1 + amount)))))
        return (r, g, b)

    def draw(self, surf, camera, screen_size, visual_scale=1.0, brightside=True, light_dir=(-0.6, -0.4)):
        sw, sh = screen_size
        lx, ly = light_dir
        mag = math.hypot(lx, ly)
        if mag == 0:
            lx, ly = -0.6, -0.4
            mag = math.hypot(lx, ly)
        lx /= mag; ly /= mag

        for (gx, gy) in self.blocks:
            wx = gx * self.grid_size
            wy = gy * self.grid_size
            sx, sy = camera.world_to_screen(wx, wy, sw, sh)
            size = int(round(self.grid_size * camera.zoom * visual_scale))
            if size < MIN_SCREEN_BLOCK_SIZE:
                continue
            rect = pygame.Rect(sx - size//2, sy - size//2, size, size)

            base_color = (50, 200, 50)
            if brightside:
                highlight = self._shade_color(base_color, HIGHLIGHT_INTENSITY)
                shadow = self._shade_color(base_color, -SHADOW_INTENSITY)
                pygame.draw.rect(surf, base_color, rect)
                t = max(1, int(round(size * 0.18)))
                if lx < 0:
                    left_rect = pygame.Rect(rect.left, rect.top, t, rect.height)
                    right_rect = pygame.Rect(rect.right - t, rect.top, t, rect.height)
                    pygame.draw.rect(surf, highlight, left_rect)
                    pygame.draw.rect(surf, shadow, right_rect)
                else:
                    left_rect = pygame.Rect(rect.left, rect.top, t, rect.height)
                    right_rect = pygame.Rect(rect.right - t, rect.top, t, rect.height)
                    pygame.draw.rect(surf, shadow, left_rect)
                    pygame.draw.rect(surf, highlight, right_rect)
                if ly < 0:
                    top_rect = pygame.Rect(rect.left, rect.top, rect.width, t)
                    bottom_rect = pygame.Rect(rect.left, rect.bottom - t, rect.width, t)
                    pygame.draw.rect(surf, highlight, top_rect)
                    pygame.draw.rect(surf, shadow, bottom_rect)
                else:
                    top_rect = pygame.Rect(rect.left, rect.top, rect.width, t)
                    bottom_rect = pygame.Rect(rect.left, rect.bottom - t, rect.width, t)
                    pygame.draw.rect(surf, shadow, top_rect)
                    pygame.draw.rect(surf, highlight, bottom_rect)
                pygame.draw.rect(surf, (0,0,0), rect, 1)
            else:
                pygame.draw.rect(surf, base_color, rect)
                pygame.draw.rect(surf, (0,0,0), rect, 1)

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

        self.left_state = HandActionState()
        self.right_state = HandActionState()

        self.both_pinched = False
        self.both_initial = None

        self.left_last_place_time = 0.0
        self.left_last_place_cell = None
        self.left_last_place_px = None

        self.left_finger_ema = None
        self.right_finger_ema = None

        self.camera_target = [self.camera.pos[0], self.camera.pos[1]]

        self.debug = True
        self.visual_scale = BLOCK_VISUAL_SCALE
        self.brightside = BRIGHTSIDE_DEFAULT
        self.light_dir = LIGHT_DIR

    def euclid(self, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return math.hypot(dx, dy)

    def _ema_update(self, prev, value, alpha=FINGER_EMA_ALPHA):
        if prev is None:
            return value
        return (alpha * value[0] + (1 - alpha) * prev[0],
                alpha * value[1] + (1 - alpha) * prev[1])

    def _exp_alpha(self, dt, tau):
        if tau <= 0 or dt <= 0:
            return 1.0
        return 1.0 - math.exp(-dt / tau)

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
                    if event.key == pygame.K_b:
                        self.brightside = not self.brightside
                        get_logger_info('GAME', f'Brightside toggled: {self.brightside}', True)
                    if event.key == pygame.K_v and BLOCK_VISUAL_SCALE_ALLOW_OVERLAP:
                        self.visual_scale = 1.4 if self.visual_scale == 1.0 else 1.0

            t_now = time.perf_counter()
            data = self.ar.render(self.screen)

            left_pts = data["POSITION_DATA"].get("LEFT", [])
            right_pts = data["POSITION_DATA"].get("RIGHT", [])
            left_pinched = data["CLICK_FLAG"].get("LEFT", False)
            right_pinched = data["CLICK_FLAG"].get("RIGHT", False)

            left_tip_px = None
            right_tip_px = None
            if left_pts and len(left_pts) > INDEX_TIP_IDX:
                left_tip_px = left_pts[INDEX_TIP_IDX]
            if right_pts and len(right_pts) > INDEX_TIP_IDX:
                right_tip_px = right_pts[INDEX_TIP_IDX]

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

            left_world = None
            right_world = None
            if left_tip_px_smoothed:
                left_world = self.camera.screen_to_world(left_tip_px_smoothed[0], left_tip_px_smoothed[1], *self.win_size)
            if right_tip_px_smoothed:
                right_world = self.camera.screen_to_world(right_tip_px_smoothed[0], right_tip_px_smoothed[1], *self.win_size)

            l_rising, l_falling, l_stable = self.left_state.update(left_pinched, left_world, left_tip_px_smoothed, self.camera.pos, self.camera.zoom, t_now)
            r_rising, r_falling, r_stable = self.right_state.update(right_pinched, right_world, right_tip_px_smoothed, self.camera.pos, self.camera.zoom, t_now)

            if l_stable and r_stable:
                if not self.both_pinched:
                    self.both_pinched = True
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
                    if self.both_initial and left_tip_px_smoothed and right_tip_px_smoothed:
                        cur_dist = self.euclid(left_tip_px_smoothed, right_tip_px_smoothed)
                        if self.both_initial["dist"] > 1e-6:
                            scale = cur_dist / self.both_initial["dist"]
                            self.camera.zoom = self.both_initial["zoom"] * scale
                            self.camera.clamp_zoom()
                        lx, ly = left_tip_px_smoothed
                        rx, ry = right_tip_px_smoothed
                        cur_angle = math.atan2(ry - ly, rx - lx)
                        init_lx, init_ly = self.both_initial["left_px"]
                        init_rx, init_ry = self.both_initial["right_px"]
                        init_angle = math.atan2(init_ry - init_ly, init_rx - init_lx)
                        delta_angle = cur_angle - init_angle
                        self.camera.angle = self.both_initial["angle"] + delta_angle
            else:
                if self.both_pinched:
                    self.both_pinched = False
                    self.both_initial = None

            if l_stable and not (l_stable and r_stable):
                if left_world:
                    gx, gy = self.block_world.world_to_grid(left_world[0], left_world[1])
                    current_cell = (gx, gy)
                    place_allowed = False
                    if (t_now - self.left_last_place_time) >= PLACEMENT_COOLDOWN:
                        if self.left_last_place_cell != current_cell:
                            place_allowed = True
                        else:
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
                self.left_last_place_px = None

            if r_stable and not (l_stable and r_stable) and right_world and self.right_state.start_screen and self.right_state.start_cam is not None:
                start_sx, start_sy = self.right_state.start_screen
                cur_sx, cur_sy = right_tip_px_smoothed
                dx_px = cur_sx - start_sx
                dy_px = cur_sy - start_sy

                dz = max(1.0, DEADZONE_PX * self.camera.zoom)
                if abs(dx_px) < dz and abs(dy_px) < dz:
                    dx_px = 0
                    dy_px = 0

                start_zoom = self.right_state.start_zoom if self.right_state.start_zoom else self.camera.zoom
                dx_world = -dx_px / start_zoom
                dy_world = -dy_px / start_zoom

                start_cam_x, start_cam_y = self.right_state.start_cam
                target_x = start_cam_x + dx_world
                target_y = start_cam_y + dy_world

                self.camera_target[0] = target_x
                self.camera_target[1] = target_y

                dt = self.delta_time or (1.0/60.0)
                alpha = self._exp_alpha(dt, SMOOTH_TAU)
                step_x = (self.camera_target[0] - self.camera.pos[0]) * alpha
                step_y = (self.camera_target[1] - self.camera.pos[1]) * alpha
                step_x = max(-MAX_STEP_WORLD, min(MAX_STEP_WORLD, step_x))
                step_y = max(-MAX_STEP_WORLD, min(MAX_STEP_WORLD, step_y))
                self.camera.pos[0] += step_x
                self.camera.pos[1] += step_y
            else:
                dt = self.delta_time or (1.0/60.0)
                if INSTANT_STOP:
                    self.camera.pos[0] = self.camera_target[0]
                    self.camera.pos[1] = self.camera_target[1]
                else:
                    alpha = self._exp_alpha(dt, SMOOTH_TAU)
                    step_x = (self.camera_target[0] - self.camera.pos[0]) * alpha
                    step_y = (self.camera_target[1] - self.camera.pos[1]) * alpha
                    step_x = max(-MAX_STEP_WORLD, min(MAX_STEP_WORLD, step_x))
                    step_y = max(-MAX_STEP_WORLD, min(MAX_STEP_WORLD, step_y))
                    self.camera.pos[0] += step_x
                    self.camera.pos[1] += step_y

            if r_falling:
                get_logger_info('GAME', f'Right pinch released; camera pos {self.camera.pos}', True)

            visual_scale = self.visual_scale if BLOCK_VISUAL_SCALE_ALLOW_OVERLAP else 1.0
            self.block_world.draw(self.screen, self.camera, self.win_size, visual_scale, brightside=self.brightside, light_dir=self.light_dir)

            if self.debug:
                if left_tip_px_smoothed:
                    pygame.draw.circle(self.screen, (255, 100, 100), left_tip_px_smoothed, 8, 2)
                    if left_pinched:
                        pygame.draw.circle(self.screen, (255, 50, 50), left_tip_px_smoothed, 6)
                if right_tip_px_smoothed:
                    pygame.draw.circle(self.screen, (100, 100, 255), right_tip_px_smoothed, 8, 2)
                    if right_pinched:
                        pygame.draw.circle(self.screen, (50, 50, 255), right_tip_px_smoothed, 6)

                font = pygame.font.SysFont("Arial", 16)
                lines = [
                    f"FPS: {int(self.clock.get_fps())}",
                    f"Camera pos: ({self.camera.pos[0]:.1f}, {self.camera.pos[1]:.1f})",
                    f"Target pos: ({self.camera_target[0]:.1f}, {self.camera_target[1]:.1f})",
                    f"Zoom: {self.camera.zoom:.2f}",
                    f"Left pinched: {l_stable} Right pinched: {r_stable}",
                    f"Blocks: {len(self.block_world.blocks)}",
                    f"Brightside: {self.brightside} (press B)",
                    f"Light dir: ({self.light_dir[0]:.2f},{self.light_dir[1]:.2f})",
                    f"Visual scale: {visual_scale}",
                    f"Right start_screen: {self.right_state.start_screen}",
                    f"Right start_cam: {self.right_state.start_cam}",
                    "Press D to toggle debug"
                ]
                for i, line in enumerate(lines):
                    surf_text = font.render(line, True, (220,220,220))
                    self.screen.blit(surf_text, (10, 10 + i*18))

            pygame.display.update()
            self.delta_time = self.clock.tick(60) / 1000.0

if __name__ == "__main__":
    GraphicsEngine2D().run()
