# ui/hud.py
import pygame as pg
import moderngl as mgl
import array
import math
import time
from settings import WIN_RES, BRUSH_MULT_MIN, BRUSH_MULT_MAX, BOTH_HANDS_HOLD_TIME

# MediaPipe Hand Skeleton Mapping
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]

class HUD:
    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.ctx
        self.res = (int(WIN_RES[0]), int(WIN_RES[1]))
        self.visible = True  # DEPTH INTEGRATION: toggleable HUD
        
        self.surface = pg.Surface(self.res, pg.SRCALPHA)
        pg.font.init()
        self.font = pg.font.SysFont('Consolas', 16, bold=True)
        self.title_font = pg.font.SysFont('Consolas', 22, bold=True)
        self.big_font = pg.font.SysFont('Consolas', 28, bold=True)
        
        self.texture = self.ctx.texture(self.res, 4)
        self.texture.filter = (mgl.NEAREST, mgl.NEAREST)
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

    def draw_text(self, text, x, y, color, font=None, shadow=True):
        if not self.visible: return
        f = font if font else self.font
        if shadow:
            shadow_surface = f.render(text, True, (10, 10, 10))
            self.surface.blit(shadow_surface, (x + 2, y + 2))
        text_surface = f.render(text, True, color)
        self.surface.blit(text_surface, (x, y))

    def draw_glass_panel(self, rect, color=(20, 25, 30, 180)):
        if not self.visible: return
        pg.draw.rect(self.surface, color, rect, border_radius=8)
        pg.draw.rect(self.surface, (100, 150, 200, 100), rect, width=2, border_radius=8)

    def get_screen_coords(self, pos):
        if pos.x <= 2.0 and pos.y <= 2.0:
            return int(pos.x * self.res[0]), int(pos.y * self.res[1])
        return int(pos.x), int(pos.y)

    def draw_skeleton(self, landmarks, color, is_pinched):
        if not self.visible: return
        if len(landmarks) < 21: return
        screen_pts = [self.get_screen_coords(pt) for pt in landmarks]
        bone_color = (color[0]*0.7, color[1]*0.7, color[2]*0.7)
        for connection in HAND_CONNECTIONS:
            start_pt = screen_pts[connection[0]]
            end_pt = screen_pts[connection[1]]
            pg.draw.line(self.surface, bone_color, start_pt, end_pt, 3)
        for i, pt in enumerate(screen_pts):
            radius = 6 if i in (4, 8) and is_pinched else 4
            pg.draw.circle(self.surface, color, pt, radius)

    def draw_crosshair(self, x, y, color, is_pinched, label):
        if not self.visible: return
        radius = 18 if is_pinched else 24
        thickness = 4 if is_pinched else 2
        pg.draw.circle(self.surface, color, (x, y), radius, thickness)
        if is_pinched: pg.draw.circle(self.surface, color, (x, y), 6)
        length = 12
        pg.draw.line(self.surface, color, (x - radius - length, y), (x - radius, y), 2)
        pg.draw.line(self.surface, color, (x + radius, y), (x + radius + length, y), 2)
        pg.draw.line(self.surface, color, (x, y - radius - length), (x, y - radius), 2)
        pg.draw.line(self.surface, color, (x, y + radius), (x, y + radius + length), 2)
        self.draw_text(label, x + radius + 15, y - 10, color)

    def draw_text_right(self, text, right_x, y, color, font=None, shadow=True):
        if not self.visible: return
        f = font if font else self.font
        text_surface = f.render(text, True, color)
        x = right_x - text_surface.get_width()
        if shadow:
            shadow_surface = f.render(text, True, (10, 10, 10))
            self.surface.blit(shadow_surface, (x + 2, y + 2))
        self.surface.blit(text_surface, (x, y))

    def update_surface(self):
        self.surface.fill((0, 0, 0, 0))
        if not self.visible:
            return
        self.pulse_timer += 0.1
        
        ar = getattr(self.engine, 'ar_controller', None)
        if not ar: return

        l_landmarks = ar.smooth_left_landmarks
        r_landmarks = ar.smooth_right_landmarks
        left_pos = ar.smooth_left_pos
        right_pos = ar.smooth_right_pos
        left_pinch = getattr(ar, '_raw_left_pinch', False)
        right_pinch = getattr(ar, '_raw_right_pinch', False)
        voxel_handler = self.engine.scene.world.voxel_handler

        # Status strings
        l_status = "L-HAND STANDBY"
        r_status = "R-HAND AIMING"
       # Inside update_surface()
        if left_pinch and right_pinch:
            l_status = r_status = "ZOOMING WORLD"
            l_color = r_color = (255, 150, 255)
        else:
            l_status = "ROTATING WORLD" if left_pinch else "L-HAND STANDBY"
            r_status = "BUILDING" if right_pinch else "R-HAND AIMING"
            l_color = (255, 180, 50) if left_pinch else (50, 200, 255)
            r_color = (100, 255, 100) if right_pinch else (255, 100, 100)

        # Camera feed
        if hasattr(ar.ar, 'image') and ar.ar.image is not None:
            cam_surf = ar.ar.image
            cam_surf = pg.transform.scale(cam_surf, (320, 180))
            pip_rect = cam_surf.get_rect(bottomleft=(20, self.res[1] - 20))
            self.draw_glass_panel(pip_rect.inflate(10, 10))
            self.surface.blit(cam_surf, pip_rect)
            self.draw_text("AR OPTICAL FEED", pip_rect.x + 5, pip_rect.y - 25, (200, 200, 200))

        # Left hand
        if left_pos is not None:
            self.draw_skeleton(l_landmarks, l_color, left_pinch)
            px, py = self.get_screen_coords(left_pos)
            self.draw_crosshair(px, py, l_color, left_pinch, l_status)
            self.draw_text(f"Z: {left_pos.z:.2f}", px + 25, py - 15, (200, 200, 200))

        # Right hand
        if right_pos is not None:
            self.draw_skeleton(r_landmarks, r_color, right_pinch)
            px, py = self.get_screen_coords(right_pos)
            self.draw_crosshair(px, py, r_color, right_pinch, r_status)
            self.draw_text(f"Z: {right_pos.z:.2f}", px + 25, py - 30, (200, 200, 200))
            if right_pinch and voxel_handler.is_dragging:
                self.draw_text(f"BRUSH x{voxel_handler.brush_mult:.2f}", px + 25, py - 45, (255, 255, 0))

        # Left panel
        left_tracking = f'ACTIVE : {ar._hand_type_left}' if left_pos else "LOST"
        right_tracking = f'ACTIVE : {ar._hand_type_right}' if right_pos else "LOST"
        panel_rect = pg.Rect(20, 20, 320, 180)
        self.draw_glass_panel(panel_rect)
        self.draw_text("XR SPATIAL ENGINE", 35, 35, (255, 255, 255), self.title_font)
        self.draw_text(f"LEFT TRACKING:  {left_tracking}", 35, 75, (100, 255, 100) if left_pos else (255, 100, 100))
        self.draw_text(f"RIGHT TRACKING: {right_tracking}", 35, 100, (100, 255, 100) if right_pos else (255, 100, 100))
        self.draw_text(f"WORLD SCALE: {self.engine.scene.world.world_scale:.2f}x", 35, 135, (200, 200, 200))

        # Right panel
        tool_panel_rect = pg.Rect(self.res[0] - 300, 20, 280, 140)
        self.draw_glass_panel(tool_panel_rect)
        anchor_x = self.res[0] - 35
        self.draw_text_right("VOXEL TOOL", anchor_x, 35, (255, 255, 255), self.title_font)
        mode = "ADD BLOCK" if voxel_handler.interaction_mode == 1 else "REMOVE BLOCK"
        m_color = (100, 255, 100) if voxel_handler.interaction_mode == 1 else (255, 100, 100)
        self.draw_text_right(f"MODE: {mode}", anchor_x, 75, m_color, self.big_font)
        if voxel_handler.is_dragging:
            pulse_color = (255, 200, 50)
            self.draw_text_right(">> DRAG EXTRUDE <<", anchor_x, 120, pulse_color)
            self.draw_text_right(f"BRUSH {voxel_handler.brush_mult:.2f}", anchor_x, 145, (255, 255, 0))

    def render(self):
        self.update_surface()
        texture_data = pg.image.tostring(self.surface, 'RGBA', False)
        self.texture.write(texture_data)
        self.texture.use(location=0)
        self.vao.render(mgl.TRIANGLE_STRIP)