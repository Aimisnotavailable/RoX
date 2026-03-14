# ui/hud.py
import pygame as pg
import moderngl as mgl
import array
import math
import time
from settings import WIN_RES

# Radial menu configuration – same techy look
RADIAL_MENU_RADIUS = 160
RADIAL_MENU_OPTIONS = [
    {"name": "SAND",   "color": (230, 210, 180), "voxel_id": 1},
    {"name": "GRASS",  "color": (100, 200, 100), "voxel_id": 2},
    {"name": "DIRT",   "color": (140, 100, 70),  "voxel_id": 3},
    {"name": "STONE",  "color": (160, 160, 170), "voxel_id": 4},
    {"name": "SNOW",   "color": (240, 240, 255), "voxel_id": 5},
    {"name": "LEAVES", "color": (80, 160, 80),   "voxel_id": 6},
    {"name": "WOOD",   "color": (180, 140, 100), "voxel_id": 7},
]

class RadialMenu:
    def __init__(self):
        self.active = False
        self.center = (0, 0)
        self.selected_index = -1
        self.options = RADIAL_MENU_OPTIONS
        self.font = pg.font.SysFont('Consolas', 16, bold=True)  # bigger
        self.pulse = 0.0

    def activate(self, center):
        self.active = True
        self.center = center
        self.selected_index = -1
        self.pulse = 0.0

    def deactivate(self):
        self.active = False

    def update_selection(self, hand_pos):
        if not self.active:
            return
        dx = hand_pos[0] - self.center[0]
        dy = hand_pos[1] - self.center[1]
        dist = math.hypot(dx, dy)
        if dist < 40:
            self.selected_index = -1
            return
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi
        n = len(self.options)
        sector = int(angle / (2 * math.pi / n))
        self.selected_index = sector % n

    def draw(self, surface, pulse_factor):
        if not self.active:
            return
        cx, cy = self.center
        n = len(self.options)

        # Outer glow ring (pulsing)
        glow_alpha = int(30 + 20 * math.sin(pulse_factor * 8))
        pg.draw.circle(surface, (100, 150, 255, glow_alpha), (cx, cy), RADIAL_MENU_RADIUS + 4, 2)

        # Dark glass base
        bg_surf = pg.Surface((RADIAL_MENU_RADIUS*2, RADIAL_MENU_RADIUS*2), pg.SRCALPHA)
        pg.draw.circle(bg_surf, (10, 15, 25, 220), (RADIAL_MENU_RADIUS, RADIAL_MENU_RADIUS), RADIAL_MENU_RADIUS)
        surface.blit(bg_surf, (cx - RADIAL_MENU_RADIUS, cy - RADIAL_MENU_RADIUS))

        # Draw sectors
        for i, option in enumerate(self.options):
            start_angle = i * (2 * math.pi / n)
            end_angle = (i + 1) * (2 * math.pi / n)
            base_color = option["color"]
            if i == self.selected_index:
                base_color = (255, 255, 100)

            points = [(cx, cy)]
            for t in range(0, 11):
                angle = start_angle + (end_angle - start_angle) * (t / 10)
                x = cx + math.cos(angle) * RADIAL_MENU_RADIUS
                y = cy + math.sin(angle) * RADIAL_MENU_RADIUS
                points.append((x, y))
            pg.draw.polygon(surface, (*base_color, 180), points)

            # Dividing lines
            line_x = cx + math.cos(start_angle) * RADIAL_MENU_RADIUS
            line_y = cy + math.sin(start_angle) * RADIAL_MENU_RADIUS
            pg.draw.line(surface, (80, 90, 120, 150), (cx, cy), (line_x, line_y), 1)

        # Last dividing line
        last_angle = 2 * math.pi
        line_x = cx + math.cos(last_angle) * RADIAL_MENU_RADIUS
        line_y = cy + math.sin(last_angle) * RADIAL_MENU_RADIUS
        pg.draw.line(surface, (80, 90, 120, 150), (cx, cy), (line_x, line_y), 1)

        # Labels
        for i, option in enumerate(self.options):
            mid_angle = (i + 0.5) * (2 * math.pi / n)
            label_x = cx + math.cos(mid_angle) * (RADIAL_MENU_RADIUS * 0.7)
            label_y = cy + math.sin(mid_angle) * (RADIAL_MENU_RADIUS * 0.7)
            text_surf = self.font.render(option["name"], True, (220, 230, 255))
            text_rect = text_surf.get_rect(center=(label_x, label_y))
            bg_rect = text_rect.inflate(8, 6)
            pg.draw.rect(surface, (10, 10, 20, 200), bg_rect, border_radius=4)
            surface.blit(text_surf, text_rect)

        # Inner circle
        pg.draw.circle(surface, (80, 100, 140, 100), (cx, cy), 12, 2)
        pg.draw.circle(surface, (150, 180, 255, 60), (cx, cy), 8, 1)


class HUD:
    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.ctx
        self.res = (int(WIN_RES[0]), int(WIN_RES[1]))
        self.visible = True
        
        self.surface = pg.Surface(self.res, pg.SRCALPHA)
        pg.font.init()
        # Larger fonts
        self.font = pg.font.SysFont('Consolas', 16)
        self.title_font = pg.font.SysFont('Consolas', 20, bold=True)
        self.big_font = pg.font.SysFont('Consolas', 26, bold=True)
        self.small_font = pg.font.SysFont('Consolas', 13)
        
        self.texture = self.ctx.texture(self.res, 4)
        self.texture.filter = (mgl.LINEAR, mgl.LINEAR)
        self.program = engine.shader_program.hud
        self.program['u_texture_0'] = 0
        
        quad_data = [
            -1.0,  1.0,  0.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
             1.0, -1.0,  1.0, 1.0,
        ]
        self.vbo = self.ctx.buffer(data=array.array('f', quad_data))
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f 2f', 'in_position', 'in_uv')])
        
        self.pulse_timer = 0.0
        self.radial_menu = RadialMenu()

    def draw_tech_panel(self, rect, color=(15, 20, 30, 200), accent=(80, 140, 200, 150)):
        pg.draw.rect(self.surface, color, rect, border_radius=8)
        inner_rect = rect.inflate(-4, -4)
        pg.draw.rect(self.surface, (*color[:3], 60), inner_rect, border_radius=6, width=1)
        pg.draw.rect(self.surface, accent, rect, width=1, border_radius=8)

    def draw_crosshair(self, x, y, color, is_pinched, label, sub_label=None):
        radius = 22 if is_pinched else 28
        thickness = 3 if is_pinched else 2
        pg.draw.circle(self.surface, (*color, 180), (x, y), radius, thickness)
        if is_pinched:
            pg.draw.circle(self.surface, color, (x, y), 6)

        length = 16
        pg.draw.line(self.surface, color, (x - radius - length, y), (x - radius, y), 2)
        pg.draw.line(self.surface, color, (x + radius, y), (x + radius + length, y), 2)
        pg.draw.line(self.surface, color, (x, y - radius - length), (x, y - radius), 2)
        pg.draw.line(self.surface, color, (x, y + radius), (x, y + radius + length), 2)

        # Label background (centered on text)
        label_surf = self.small_font.render(label, True, color)
        label_rect = label_surf.get_rect(midleft=(x + radius + 18, y - 10))
        bg_rect = label_rect.inflate(10, 4)
        pg.draw.rect(self.surface, (5, 5, 10, 200), bg_rect, border_radius=4)
        self.surface.blit(label_surf, label_rect)

        if sub_label:
            sub_surf = self.small_font.render(sub_label, True, (200, 200, 200))
            sub_rect = sub_surf.get_rect(midleft=(x + radius + 18, y + 8))
            sub_bg = sub_rect.inflate(10, 4)
            pg.draw.rect(self.surface, (5, 5, 10, 200), sub_bg, border_radius=4)
            self.surface.blit(sub_surf, sub_rect)

    def draw_text_centered(self, text, rect, color, font=None, bg=None):
        f = font if font else self.font
        text_surf = f.render(text, True, color)
        text_rect = text_surf.get_rect(center=rect.center)
        if bg:
            bg_rect = text_rect.inflate(10, 6)
            pg.draw.rect(self.surface, bg, bg_rect, border_radius=4)
        self.surface.blit(text_surf, text_rect)

    def draw_text(self, text, pos, color, font=None, anchor='topleft'):
        f = font if font else self.font
        text_surf = f.render(text, True, color)
        rect = text_surf.get_rect()
        setattr(rect, anchor, pos)
        self.surface.blit(text_surf, rect)

    def get_screen_coords(self, pos):
        if hasattr(pos, 'x') and pos.x <= 2.0 and pos.y <= 2.0:
            return int(pos.x * self.res[0]), int(pos.y * self.res[1])
        return int(pos[0]), int(pos[1])

    def update_surface(self):
        self.surface.fill((0, 0, 0, 0))
        if not self.visible:
            return
        self.pulse_timer += 0.02

        ar = getattr(self.engine, 'ar_controller', None)
        if not ar:
            return

        left_pos = ar.smooth_left_pos
        right_pos = ar.smooth_right_pos
        left_pinch = ar.pinch_active_left
        right_pinch = ar.pinch_active_right
        voxel_handler = self.engine.scene.world.voxel_handler

        block_map = {
            1: "SAND", 2: "GRASS", 3: "DIRT", 4: "STONE",
            5: "SNOW", 6: "LEAVES", 7: "WOOD",
        }
        current_block = block_map.get(voxel_handler.new_voxel_id, "UNKNOWN")
        mode_str = "ADD" if voxel_handler.interaction_mode == 1 else "REMOVE"

        # Left hand status
        left_status = "STANDBY"
        left_color = (140, 140, 160)
        if ar.two_finger_up_left_active:
            left_status = "ROTATE"
            left_color = (80, 180, 255)
        elif left_pinch:
            if ar.radial_menu_active:
                left_status = "MENU"
                left_color = (255, 200, 50)
            else:
                left_status = "HOLD"
                left_color = (255, 140, 60)

        # Right hand status
        right_status = "AIM"
        right_color = (140, 140, 160)
        if ar.two_finger_up_right_active:
            right_status = "LOOK"
            right_color = (80, 180, 255)
        elif right_pinch:
            if voxel_handler.is_dragging:
                right_status = "EXTRUDE"
            else:
                right_status = "BUILD"
            right_color = (100, 255, 100)

        # Camera feed (bottom left) – larger frame
        if hasattr(ar.ar, 'image') and ar.ar.image is not None:
            cam_surf = ar.ar.image
            cam_surf = pg.transform.smoothscale(cam_surf, (340, 190))
            pip_rect = cam_surf.get_rect(bottomleft=(20, self.res[1] - 20))
            pg.draw.rect(self.surface, (10, 12, 18, 220), pip_rect.inflate(10, 10), border_radius=8)
            pg.draw.rect(self.surface, (60, 100, 160, 150), pip_rect.inflate(10, 10), width=1, border_radius=8)
            self.surface.blit(cam_surf, pip_rect)
            self.draw_text("CAMERA", (pip_rect.x + 5, pip_rect.y - 22), (180, 200, 255), self.small_font)

        # Left hand crosshair
        if left_pos is not None:
            px, py = self.get_screen_coords(left_pos)
            sub = f"Z:{left_pos.z:.2f}"
            self.draw_crosshair(px, py, left_color, left_pinch, left_status, sub)

        # Right hand crosshair
        if right_pos is not None:
            px, py = self.get_screen_coords(right_pos)
            sub = f"Z:{right_pos.z:.2f}"
            self.draw_crosshair(px, py, right_color, right_pinch, right_status, sub)
            if right_pinch and voxel_handler.is_dragging:
                self.draw_text(f"x{voxel_handler.brush_mult:.2f}", (px + 45, py - 45), (255, 255, 80), self.small_font, anchor='center')

        # Radial menu
        if ar.radial_menu_active and left_pos is not None:
            hand_screen = (left_pos.x * WIN_RES[0], left_pos.y * WIN_RES[1])
            self.radial_menu.update_selection(hand_screen)
            self.radial_menu.draw(self.surface, self.pulse_timer)

        # ---- Top left system panel - larger, centered text ----
        info_rect = pg.Rect(20, 20, 280, 190)
        self.draw_tech_panel(info_rect)
        y_offsets = [35, 60, 85, 110, 140, 165]
        texts = [
            f"FPS: {self.engine.clock.get_fps():.0f}",
            f"BLOCK: {current_block}",
            f"MODE: {mode_str}",
            f"SCALE: {self.engine.scene.world.world_scale:.2f}x",
            f"LEFT: {left_status}",
            f"RIGHT: {right_status}"
        ]
        colors = [
            (255, 255, 100),
            (200, 220, 255),
            (100, 255, 100) if mode_str == "ADD" else (255, 100, 100),
            (180, 180, 220),
            left_color,
            right_color
        ]
        for y, text, col in zip(y_offsets, texts, colors):
            self.draw_text(text, (info_rect.x + 20, y), col, self.font, anchor='topleft')

        # ---- Top right panel (build/extrude) ----
        if voxel_handler.is_dragging:
            drag_rect = pg.Rect(self.res[0] - 220, 20, 200, 70)
            self.draw_tech_panel(drag_rect)
            self.draw_text_centered("EXTRUDING", drag_rect, (255, 180, 80), self.small_font, bg=(10, 10, 15, 200))
            self.draw_text(f"BRUSH {voxel_handler.brush_mult:.2f}", (drag_rect.centerx, drag_rect.bottom - 18), (255, 255, 100), self.small_font, anchor='center')
        else:
            build_rect = pg.Rect(self.res[0] - 220, 20, 200, 60)
            self.draw_tech_panel(build_rect)
            self.draw_text_centered("BUILD", build_rect, (100, 255, 100), self.small_font, bg=(10, 10, 15, 200))
            self.draw_text(mode_str, (build_rect.centerx, build_rect.bottom - 18), (200, 220, 255), self.small_font, anchor='center')

        # ---- Bottom right (tracking status) ----
        track_rect = pg.Rect(self.res[0] - 220, self.res[1] - 80, 200, 60)
        self.draw_tech_panel(track_rect)
        left_track = "REAL" if ar._hand_type_left == "REAL" else "GHOST"
        right_track = "REAL" if ar._hand_type_right == "REAL" else "GHOST"
        self.draw_text(f"L:{left_track}", (track_rect.x + 20, track_rect.y + 18), (100, 255, 100) if ar._hand_type_left == "REAL" else (180, 180, 180), self.small_font)
        self.draw_text(f"R:{right_track}", (track_rect.x + 20, track_rect.y + 38), (100, 255, 100) if ar._hand_type_right == "REAL" else (180, 180, 180), self.small_font)

    def render(self):
        self.update_surface()
        texture_data = pg.image.tostring(self.surface, 'RGBA', False)
        self.texture.write(texture_data)
        self.texture.use(location=0)
        self.vao.render(mgl.TRIANGLE_STRIP)