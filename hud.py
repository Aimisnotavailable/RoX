import pygame as pg
import moderngl as mgl
import numpy as np
from settings import *
from scripts.arconfig import *
from mediapipe.python.solutions.hands import HAND_CONNECTIONS

class HUD:
    def __init__(self, engine):
        self.engine = engine
        self.ctx = engine.ctx
        
        # Grab the shader from your centralized shader program!
        self.program = self.engine.shader_program.hud
        
        # Fix: Ensure WIN_RES is strictly cast to integers for ModernGL
        self.res = (int(WIN_RES[0]), int(WIN_RES[1]))
        
        self.surface = pg.Surface(self.res, pg.SRCALPHA)
        self.font = pg.font.SysFont('courier', 18, bold=True)
        self.title_font = pg.font.SysFont('courier', 24, bold=True)
        
        self.texture = self.ctx.texture(self.res, 4)
        self.texture.filter = (mgl.NEAREST, mgl.NEAREST)
        
        # Fullscreen quad vertices
        vertices = np.array([
            -1.0,  1.0,  0.0, 0.0,
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 1.0,
             1.0,  1.0,  1.0, 0.0,
        ], dtype='f4')
        
        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.vertex_array(self.program, [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')])

    def render_feed_to_texture(self, surf, fit_to_screen=False):
        img = self.engine.ar_controller.ar_system.image
        if img is not None:
            image = img
            if fit_to_screen:
                image = pygame.transform.scale(img, WIN_RES)
            surf.blit(image, (0, 0) if fit_to_screen else (WIN_RES[0] - image.get_width(), 0))

    def render_hands(self, surf):
        hand_pts  = self.engine.ar_controller.ar_data['POSITION_DATA']
        for label in ("LEFT", "RIGHT"):
            pts = hand_pts[label]
            if not pts:
                continue

            try:
                max_idx = max(max(c) for c in HAND_CONNECTIONS)
            except Exception:
                max_idx = -1

            if len(pts) > max_idx:
                for a_idx, b_idx in HAND_CONNECTIONS:
                    if a_idx >= len(pts) or b_idx >= len(pts):
                        continue
                    pa = pts[a_idx]
                    pb = pts[b_idx]
                    if pa and pb:
                        try:
                            pygame.draw.line(surf, (0, 0, 255), (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), 1)
                        except Exception:
                            continue

            for p in pts:
                if not p:
                    continue
                try:
                    cx = int(round(p[0])); cy = int(round(p[1]))
                    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 2)
                except Exception:
                    continue

                color = (200, 80, 80) if label == "LEFT" else (80, 80, 200)
                p = pts[WRIST_IDX] if len(pts) > WRIST_IDX else max_idx
                pygame.draw.circle(surf, color, p, 10, 2)
                font = pygame.font.SysFont("Arial", 14)
                txt = font.render(f"{label}", True, color)
                surf.blit(txt, (max(0, p[0]-20), max(0, p[1]-30)))

    def update_surface(self):
        self.surface.fill((0, 0, 0, 0)) # Clear transparent
        
        fps_text = f"FPS: {self.engine.clock.get_fps():.0f}"
        
        ar_state = "INACTIVE"
        action = "IDLE"
        if hasattr(self.engine, 'ar_controller'):
            ar_state = "ACTIVE" if self.engine.ar_controller.is_running else "WAITING"
            action = self.engine.ar_controller.current_action_label

        # Draw UI Panel background
        pg.draw.rect(self.surface, (20, 20, 20, 180), (10, 10, 320, 160), border_radius=8)
        pg.draw.rect(self.surface, (0, 255, 150, 255), (10, 10, 320, 160), width=2, border_radius=8)
        
        # Render Text
        self.surface.blit(self.title_font.render("[ RoX Engine ]", True, (0, 255, 150)), (25, 20))
        self.surface.blit(self.font.render(fps_text, True, (255, 255, 255)), (25, 60))
        self.surface.blit(self.font.render(f"AR:  {ar_state}", True, (255, 255, 255)), (25, 110))
        self.surface.blit(self.font.render(f"ACT: {action}", True, (255, 255, 0)), (25, 135))
        self.render_feed_to_texture(self.surface)
        self.render_hands(self.surface)
        # Draw AR Hand Cursors dynamically on screen!
        if hasattr(self.engine, 'ar_controller'):
            ctrl = self.engine.ar_controller
            if ctrl.l_pos:
                pg.draw.circle(self.surface, (0, 150, 255), (int(ctrl.l_pos[0]), int(ctrl.l_pos[1])), 10, 2)
            if ctrl.r_pos:
                color = (255, 50, 50) if ctrl.r_click else (0, 255, 150)
                px, py = int(ctrl.r_pos[0]), int(ctrl.r_pos[1])
                pg.draw.circle(self.surface, color, (px, py), 12, 3)
                pg.draw.line(self.surface, color, (px-15, py), (px+15, py), 2)
                pg.draw.line(self.surface, color, (px, py-15), (px, py+15), 2)

        # Upload Pygame canvas to OpenGL Texture
        texture_data = pg.image.tobytes(self.surface, 'RGBA')
        self.texture.write(texture_data)

    def render(self):
        self.update_surface()
        self.ctx.enable(mgl.BLEND)
        self.texture.use(location=0)
        self.vao.render()