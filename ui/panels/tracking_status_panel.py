import pygame as pg
from ui.base_panel import BasePanel

class TrackingStatusPanel(BasePanel):
    def __init__(self, hud):
        super().__init__(hud)
        self.rect = pg.Rect(hud.res[0] - 220, 100, 200, 60)
    
    def update(self):
        pass
    
    def draw(self, surface):
        self.hud.draw_glass_panel(surface, self.rect)
        ar = self.engine.ar_controller
        left_track = "REAL" if ar._hand_type_left == "REAL" else "GHOST"
        right_track = "REAL" if ar._hand_type_right == "REAL" else "GHOST"
        left_color = (100, 255, 100) if left_track == "REAL" else (180, 180, 180)
        right_color = (100, 255, 100) if right_track == "REAL" else (180, 180, 180)
        
        self.hud.draw_text(surface, f"L:{left_track}",
                          (self.rect.x + 20, self.rect.y + 18),
                          left_color, self.hud.small_font)
        self.hud.draw_text(surface, f"R:{right_track}",
                          (self.rect.x + 20, self.rect.y + 38),
                          right_color, self.hud.small_font)