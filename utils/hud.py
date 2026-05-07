import pygame as pg
import moderngl as mgl
import array
import time
from settings import WIN_RES, INTERACTION_MODE, INTERACTION_COLORS, glm, math
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
        self.fps_values = []
        self.fps_max_len = 40
    
    # Drawing helpers (delegated from panels)
    def draw_glass_panel(self, rect, color=(15, 20, 30, 200), accent=(80, 140, 200, 150), glow=False):
        if glow:
            draw_glow_rect(self.surface, rect, accent, radius=8, glow_size=8)
        draw_rounded_rect(self.surface, rect, color, radius=8)
        inner_rect = rect.inflate(-4, -4)
        draw_rounded_rect(self.surface, inner_rect, (*color[:3], 60), radius=6, border_width=1, border_color=accent)
    
    def draw_circular_status(self,x, y, label, status, color, radius=20):
        pg.draw.circle(self.surface, (color[0], color[1], color[2], 200), (x, y), radius+2, 2)
        pg.draw.circle(self.surface, (color[0], color[1], color[2], 80), (x, y), radius-2)
        txt = self.caption_font.render(label, True, (220, 220, 255))
        txt_rect = txt.get_rect(center=(x, y - radius - 8))
        self.surface.blit(txt, txt_rect)
        status_txt = self.small_font.render(status, True, color)
        status_rect = status_txt.get_rect(center=(x, y))
        bg_rect = status_rect.inflate(8, 4)
        draw_rounded_rect(self.surface, bg_rect, (0, 0, 0, 150), radius=4)
        self.surface.blit(status_txt, status_rect)
    
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
    
    def draw_text(self,text, pos, color, font=None, anchor='topleft', shadow=True):
        f = font if font else self.font
        text_surf = f.render(text, True, color)
        rect = text_surf.get_rect()
        setattr(rect, anchor, pos)
        if shadow:
            shadow_surf = f.render(text, True, (0, 0, 0))
            shadow_rect = rect.copy()
            shadow_rect.x += 2
            shadow_rect.y += 2
            self.surface.blit(shadow_surf, shadow_rect)
        self.surface.blit(text_surf, rect)
    
    def draw_text_centered(self, text, rect, color, font=None, bg=None):
        f = font if font else self.font
        text_surf = f.render(text, True, color)
        text_rect = text_surf.get_rect(center=rect.center)
        if bg:
            bg_rect = text_rect.inflate(10, 6)
            draw_rounded_rect(self.surface, bg_rect, bg, radius=4)
        self.surface.blit(text_surf, text_rect)
    
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
    
    def _draw_animated_message(self):
        if not self.active_message:
            return
        now = time.time()
        msg = self.active_message
        elapsed = now - msg['start_time']
        if elapsed > msg['duration']:
            self.active_message = None
            return

        # Animation: scale and fade
        anim_duration = 0.3
        if elapsed < anim_duration:
            progress = elapsed / anim_duration
            scale = 1.0 + math.sin(progress * math.pi) * 0.2
            alpha = int(255 * progress)
        else:
            scale = 1.0
            alpha = 255

        # Determine position
        if msg['screen_pos'] is not None:
            center_x, center_y = msg['screen_pos']
        else:
            center_x, center_y = self.res[0] // 2, self.res[1] - 50

        panel_width = 400
        panel_height = 80
        scaled_w = int(panel_width * scale)
        scaled_h = int(panel_height * scale)

        # Create panel surface
        panel_surf = pg.Surface((scaled_w, scaled_h), pg.SRCALPHA)
        panel_rect = panel_surf.get_rect()
        # Background with gradient
        for i in range(scaled_h):
            gradient_color = (15, 20, 30, int(alpha * (0.5 + 0.5 * (i/scaled_h))))
            pg.draw.line(panel_surf, gradient_color, (0, i), (scaled_w, i))
        # Border
        pg.draw.rect(panel_surf, (100, 150, 200, int(alpha*0.8)), panel_rect, 2, border_radius=8)
        # Scanline effect
        for i in range(0, scaled_h, 4):
            pg.draw.line(panel_surf, (100, 150, 200, int(alpha*0.2)), (0, i), (scaled_w, i))

        # Text
        text_surf = self.font.render(msg['text'], True, (255, 255, 100))
        text_rect = text_surf.get_rect(center=panel_rect.center)
        panel_surf.blit(text_surf, text_rect)

        # Blit to main surface
        scaled_rect = panel_surf.get_rect(center=(center_x, center_y))
        self.surface.blit(panel_surf, scaled_rect)
    
    def update_surface(self):
        self.surface.fill((0, 0, 0, 0))
        if not self.visible:
            return
        self.pulse_timer += 0.02

        ar = getattr(self.engine, 'ar_controller', None)
        if not ar:
            return

        vh = self.engine.scene.world_container.voxel_handler
        world = self.engine.scene.world_container.local_worlds[0]
        block_map = {1: "SAND", 2: "GRASS", 3: "DIRT", 4: "STONE", 5: "SNOW", 6: "LEAVES", 7: "WOOD"}
        current_block = block_map.get(vh.new_voxel_id, "UNKNOWN")
        mode_str = INTERACTION_MODE[vh.interaction_mode]

        # ---- Top‑left info panel (glass) ----
        info_rect = pg.Rect(20, 20, 300, 150)
        self.draw_glass_panel(info_rect, glow=True)
        lines = [
            f"FPS: {self.engine.clock.get_fps():.0f}",
            f"BLOCK: {current_block}",
            f"MODE: {mode_str}",
            f"SCALE: {world.world_scale:.2f}x",
            f"POS : {world.engine.player.position}",
            f"Interaction Mode :  {self.engine.scene.interaction_coordinator.mode}"
            # f"GEN: {world.generator_type.capitalize()}"
        ]
        y = info_rect.y + 20
        for i, line in enumerate(lines):
            self.draw_text(line, (info_rect.x + 20, y + i*22), (220, 230, 255), self.font, anchor='topleft')
        
        # ---- Object selection info ----
        selected = self.engine.scene.world_container.selected_object
        if selected:
            obj_rect = pg.Rect(20, 180, 300, 100)
            self.draw_glass_panel(obj_rect)
            pos = selected.position
            euler = glm.eulerAngles(selected.rotation)
            self.draw_text(f"OBJ: {type(selected).__name__}", (obj_rect.x+20, obj_rect.y+15), (255,200,100), self.font)
            self.draw_text(f"Pos: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})", (obj_rect.x+20, obj_rect.y+35), (200,220,255), self.small_font)
            self.draw_text(f"Rot: ({math.degrees(euler.x):.0f}, {math.degrees(euler.y):.0f}, {math.degrees(euler.z):.0f})", (obj_rect.x+20, obj_rect.y+55), (200,220,255), self.small_font)
            self.draw_text(f"Scl: {selected.scale.x:.2f}", (obj_rect.x+20, obj_rect.y+75), (200,220,255), self.small_font)

        # ---- Hand status (circular) ----
        left_status = "STANDBY"
        right_status = "AIM"
        left_color = (140, 140, 160)
        right_color = (140, 140, 160)
        if ar.two_finger_up_left_active:
            left_status = "ROTATE"
            left_color = (80, 180, 255)
        elif ar.pinch_active_left:
            if ar.radial_menu_active:
                left_status = "SELECT"
                left_color = (255, 200, 50)
            else:
                left_status = "HOLD"
                left_color = (255, 140, 60)

        if ar.two_finger_up_right_active:
            right_status = "LOOK"
            right_color = (80, 180, 255)
        elif ar.pinch_active_right:
            if vh.is_dragging:
                right_status = "EXTRUDE"
            else:
                right_status = "BUILD"
            right_color = (100, 255, 100)
        elif ar.is_grabbing:
            right_status = "GRAB"
            right_color = (255, 180, 80)

        # Draw circular status at bottom corners (raised to avoid camera feed)
        left_x = 70
        left_y = self.res[1] - 85
        right_x = self.res[0] - 70
        right_y = self.res[1] - 85
        self.draw_circular_status(left_x, left_y, "LEFT", left_status, left_color)
        self.draw_circular_status(right_x, right_y, "RIGHT", right_status, right_color)

        # ---- Camera feed (bottom‑left) ----
        if hasattr(ar.ar, 'image') and ar.ar.image is not None:
            cam_surf = ar.ar.image
            cam_surf = pg.transform.smoothscale(cam_surf, (260, 146))
            pip_rect = cam_surf.get_rect(bottomleft=(20, self.res[1] - 20))
            self.draw_glass_panel(pip_rect.inflate(10, 10), color=(0,0,0,100), accent=(80,140,200,200))
            self.surface.blit(cam_surf, pip_rect)

        # ---- Crosshairs ----
        left_pos = ar.smooth_left_pos
        right_pos = ar.smooth_right_pos
        if left_pos is not None:
            px, py = self.get_screen_coords(left_pos)
            sub = f"Z:{left_pos.z:.2f}"
            self.draw_crosshair(px, py, left_color, ar.pinch_active_left, left_status, sub)
        if right_pos is not None:
            px, py = self.get_screen_coords(right_pos)
            sub = f"Z:{right_pos.z:.2f}"
            self.draw_crosshair(px, py, right_color, ar.pinch_active_right, right_status, sub)
            if ar.pinch_active_right and vh.is_dragging:
                self.draw_text(f"x{vh.brush_mult:.2f}", (px + 45, py - 45), (255, 255, 80), self.small_font, anchor='center')

        # ---- Radial menu ----
        if ar.radial_menu_active:
            self.radial_menu.draw(self.surface, self.pulse_timer)

        # ---- Extra info for drag/grab (bottom‑right) ----
        extra_rect = pg.Rect(self.res[0] - 220, self.res[1] - 120, 200, 100)
        self.draw_glass_panel(extra_rect)
        if vh.is_dragging:
            self.draw_text_centered("EXTRUDING", extra_rect, (255, 180, 80), self.small_font, bg=(10,10,15,200))
            self.draw_text(f"BRUSH {vh.brush_mult:.2f}", (extra_rect.centerx, extra_rect.bottom - 18), (255, 255, 100), self.small_font, anchor='center')
        elif ar.is_grabbing:
            self.draw_text_centered("GRABBING", extra_rect, (255, 180, 80), self.small_font, bg=(10,10,15,200))
            self.draw_text(f"SIZE {ar.grab_size}", (extra_rect.centerx, extra_rect.bottom - 18), (255, 255, 100), self.small_font, anchor='center')
        else:
            self.draw_text_centered("BUILD", extra_rect, (100, 255, 100), self.small_font, bg=(10,10,15,200))
            self.draw_text(mode_str, (extra_rect.centerx, extra_rect.bottom - 18), (200, 220, 255), self.small_font, anchor='center')

        # ---- FPS Graph (top‑right) ----
        # Keep track of FPS values for graph
        fps = self.engine.clock.get_fps()
        self.fps_values.append(fps)
        if len(self.fps_values) > self.fps_max_len:
            self.fps_values.pop(0)
        if len(self.fps_values) >= 2:
            graph_width = 120
            graph_height = 40
            graph_rect = pg.Rect(self.res[0] - graph_width - 20, 20, graph_width, graph_height)
            draw_rounded_rect(self.surface, graph_rect, (10, 12, 18, 200), radius=4)
            max_fps = max(max(self.fps_values), 30)
            points = []
            for i, val in enumerate(self.fps_values):
                x = graph_rect.x + (i / len(self.fps_values)) * graph_width
                y = graph_rect.y + graph_height - (val / max_fps) * graph_height
                points.append((x, y))
            if len(points) > 1:
                pg.draw.lines(self.surface, (100, 200, 100), False, points, 2)

        # ---- Tracking status (top‑right, below graph) ----
        track_rect = pg.Rect(self.res[0] - 220, 100, 200, 60)
        self.draw_glass_panel(track_rect)
        left_track = "REAL" if ar._hand_type_left == "REAL" else "GHOST"
        right_track = "REAL" if ar._hand_type_right == "REAL" else "GHOST"
        self.draw_text(f"L:{left_track}", (track_rect.x + 20, track_rect.y + 18),
                    (100, 255, 100) if ar._hand_type_left == "REAL" else (180, 180, 180), self.small_font)
        self.draw_text(f"R:{right_track}", (track_rect.x + 20, track_rect.y + 38),
                    (100, 255, 100) if ar._hand_type_right == "REAL" else (180, 180, 180), self.small_font)

        # ---- Animated message ----
        self._draw_animated_message()

        # ---- Loading overlay ----
        # if getattr(self.engine.scene.world, 'world_swapping', False):
        #     overlay = pg.Surface(self.res, pg.SRCALPHA)
        #     overlay.fill((0, 0, 0, 180))
        #     self.surface.blit(overlay, (0, 0))
        #     self.draw_text_centered("REGENERATING WORLD...", pg.Rect(0,0,self.res[0],self.res[1]), (255,255,100), self.big_font)

    def render(self):
        self.update_surface()
        texture_data = pg.image.tostring(self.surface, 'RGBA', False)
        self.texture.write(texture_data)
        self.texture.use(location=0)
        self.vao.render(mgl.TRIANGLE_STRIP)