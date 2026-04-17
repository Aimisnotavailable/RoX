import pygame as pg
from ui.base_panel import BasePanel

class CameraFeedPanel(BasePanel):
    def __init__(self, hud):
        super().__init__(hud)
        self.rect = pg.Rect(20, hud.res[1] - 166, 270, 156)
    
    def update(self):
        pass
    
    def draw(self, surface):
        ar = self.engine.ar_controller
        if hasattr(ar, 'hand_tracker') and hasattr(ar.hand_tracker, 'image'):
            cam_surf = ar.hand_tracker.image
            if cam_surf:
                cam_surf = pg.transform.smoothscale(cam_surf, (260, 146))
                pip_rect = cam_surf.get_rect(bottomleft=(20, self.hud.res[1] - 20))
                self.hud.draw_glass_panel(surface, pip_rect.inflate(10, 10),
                                         color=(0,0,0,100), accent=(80,140,200,200))
                surface.blit(cam_surf, pip_rect)