import pygame as pg
import moderngl as mgl
import array
import time
from settings import WIN_RES
from ui.hud_helpers import draw_rounded_rect, draw_glow_rect, RadialMenu
from ui.panels.info_panel import InfoPanel
from ui.panels.object_info_panel import ObjectInfoPanel
from ui.panels.hand_status_panel import HandStatusPanel
from ui.panels.crosshair_panel import CrosshairPanel
from ui.panels.radial_menu_panel import RadialMenuPanel
from ui.panels.extra_info_panel import ExtraInfoPanel
from ui.panels.fps_tracker_panel import FPSTrackerPanel
from ui.panels.tracking_status_panel import TrackingStatusPanel
from ui.panels.camera_feed_panel import CameraFeedPanel

class HUD:
    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.ctx
        self.res = (int(WIN_RES[0]), int(WIN_RES[1]))
        self.visible = True
        
        self.surface = pg.Surface(self.res, pg.SRCALPHA)
        pg.font.init()
        self.font = pg.font.SysFont('Consolas', 14, bold=True)
        self.title_font = pg.font.SysFont('Consolas', 20, bold=True)
        self.big_font = pg.font.SysFont('Consolas', 26, bold=True)
        self.small_font = pg.font.SysFont('Consolas', 12)
        self.caption_font = pg.font.SysFont('Consolas', 10)
        
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
        self.active_message = None
        
        # Initialize panels
        self.panels = [
            InfoPanel(self),
            ObjectInfoPanel(self),
            HandStatusPanel(self),
            CrosshairPanel(self),
            RadialMenuPanel(self),
            ExtraInfoPanel(self),
            FPSTrackerPanel(self),
            TrackingStatusPanel(self),
            CameraFeedPanel(self),
        ]
        self.radial_menu_panel = self.panels[4]  # reference for convenience
    
    # Drawing helpers (delegated from panels)
    def draw_glass_panel(self, surface, rect, color=(15, 20, 30, 200), accent=(80, 140, 200, 150), glow=False):
        if glow:
            draw_glow_rect(surface, rect, accent, radius=8, glow_size=8)
        draw_rounded_rect(surface, rect, color, radius=8)
        inner_rect = rect.inflate(-4, -4)
        draw_rounded_rect(surface, inner_rect, (*color[:3], 60), radius=6, border_width=1, border_color=accent)
    
    def draw_circular_status(self, surface, x, y, label, status, color, radius=20):
        pg.draw.circle(surface, (color[0], color[1], color[2], 200), (x, y), radius+2, 2)
        pg.draw.circle(surface, (color[0], color[1], color[2], 80), (x, y), radius-2)
        txt = self.caption_font.render(label, True, (220, 220, 255))
        txt_rect = txt.get_rect(center=(x, y - radius - 8))
        surface.blit(txt, txt_rect)
        status_txt = self.small_font.render(status, True, color)
        status_rect = status_txt.get_rect(center=(x, y))
        bg_rect = status_rect.inflate(8, 4)
        draw_rounded_rect(surface, bg_rect, (0, 0, 0, 150), radius=4)
        surface.blit(status_txt, status_rect)
    
    def draw_crosshair(self, surface, x, y, color, is_pinched, label, sub_label=None):
        radius = 22 if is_pinched else 28
        pulse = abs(math.sin(self.pulse_timer * 4)) * 0.3 + 0.7
        outer_color = (color[0], color[1], color[2], int(180 * pulse))
        pg.draw.circle(surface, outer_color, (x, y), radius, 3)
        if is_pinched:
            pg.draw.circle(surface, color, (x, y), 6)
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                ex = x + math.cos(rad) * radius
                ey = y + math.sin(rad) * radius
                pg.draw.line(surface, color, (x, y), (ex, ey), 1)
        else:
            pg.draw.circle(surface, color, (x, y), 4)
        length = 16
        pg.draw.line(surface, color, (x - radius - length, y), (x - radius, y), 2)
        pg.draw.line(surface, color, (x + radius, y), (x + radius + length, y), 2)
        pg.draw.line(surface, color, (x, y - radius - length), (x, y - radius), 2)
        pg.draw.line(surface, color, (x, y + radius), (x, y + radius + length), 2)
        
        label_surf = self.small_font.render(label, True, color)
        label_rect = label_surf.get_rect(midleft=(x + radius + 18, y - 10))
        shadow_surf = self.small_font.render(label, True, (0, 0, 0))
        shadow_rect = label_rect.move(2, 2)
        surface.blit(shadow_surf, shadow_rect)
        surface.blit(label_surf, label_rect)
        if sub_label:
            sub_surf = self.small_font.render(sub_label, True, (200, 200, 200))
            sub_rect = sub_surf.get_rect(midleft=(x + radius + 18, y + 8))
            shadow_sub = self.small_font.render(sub_label, True, (0, 0, 0))
            surface.blit(shadow_sub, sub_rect.move(2, 2))
            surface.blit(sub_surf, sub_rect)
    
    def draw_text(self, surface, text, pos, color, font=None, anchor='topleft', shadow=True):
        f = font if font else self.font
        text_surf = f.render(text, True, color)
        rect = text_surf.get_rect()
        setattr(rect, anchor, pos)
        if shadow:
            shadow_surf = f.render(text, True, (0, 0, 0))
            shadow_rect = rect.copy()
            shadow_rect.x += 2
            shadow_rect.y += 2
            surface.blit(shadow_surf, shadow_rect)
        surface.blit(text_surf, rect)
    
    def draw_text_centered(self, surface, text, rect, color, font=None, bg=None):
        f = font if font else self.font
        text_surf = f.render(text, True, color)
        text_rect = text_surf.get_rect(center=rect.center)
        if bg:
            bg_rect = text_rect.inflate(10, 6)
            draw_rounded_rect(surface, bg_rect, bg, radius=4)
        surface.blit(text_surf, text_rect)
    
    def get_screen_coords(self, pos):
        if hasattr(pos, 'x') and pos.x <= 2.0 and pos.y <= 2.0:
            return int(pos.x * self.res[0]), int(pos.y * self.res[1])
        return int(pos[0]), int(pos[1])
    
    def show_temp_message(self, text, duration=3.0, screen_pos=None):
        self.active_message = {
            'text': text,
            'start_time': time.time(),
            'duration': duration,
            'screen_pos': screen_pos
        }
    
    def _draw_animated_message(self, surface):
        # (unchanged, uses surface directly)
        pass
    
    def update_surface(self):
        self.surface.fill((0, 0, 0, 0))
        if not self.visible:
            return
        self.pulse_timer += 0.02
        
        # Update all panels
        for panel in self.panels:
            panel.update()
            panel.draw(self.surface)
        
        # Animated message overlay
        self._draw_animated_message(self.surface)
    
    def render(self):
        self.update_surface()
        texture_data = pg.image.tostring(self.surface, 'RGBA', False)
        self.texture.write(texture_data)
        self.texture.use(location=0)
        self.vao.render(mgl.TRIANGLE_STRIP)