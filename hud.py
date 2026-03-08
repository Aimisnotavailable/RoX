import pygame as pg
import moderngl as mgl
import numpy as np
from settings import *

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
        self.texture.swizzle = 'BGRA'
        
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