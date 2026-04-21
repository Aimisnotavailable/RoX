import pygame as pg
from ui.base_panel import BasePanel
from settings import INTERACTION_MODE

class ExtraInfoPanel(BasePanel):
    def __init__(self, hud):
        super().__init__(hud)
        self.rect = pg.Rect(hud.res[0] - 220, hud.res[1] - 120, 200, 100)
    
    def update(self):
        pass
    
    def draw(self, surface):
        self.hud.draw_glass_panel(surface, self.rect)
        
        vh = self.engine.scene.world_container.voxel_handler
        ar = self.engine.ar_controller
        mode_str = INTERACTION_MODE[vh.interaction_mode]
        
        if vh.is_dragging:
            self.hud.draw_text_centered(surface, "EXTRUDING", self.rect,
                                       (255, 180, 80), self.hud.small_font, bg=(10,10,15,200))
            self.hud.draw_text(surface, f"BRUSH {vh.brush_mult:.2f}",
                              (self.rect.centerx, self.rect.bottom - 18),
                              (255, 255, 100), self.hud.small_font, anchor='center')
        elif ar.is_grabbing:
            self.hud.draw_text_centered(surface, "GRABBING", self.rect,
                                       (255, 180, 80), self.hud.small_font, bg=(10,10,15,200))
            self.hud.draw_text(surface, f"SIZE {ar.grab_size}",
                              (self.rect.centerx, self.rect.bottom - 18),
                              (255, 255, 100), self.hud.small_font, anchor='center')
        else:
            self.hud.draw_text_centered(surface, "BUILD", self.rect,
                                       (100, 255, 100), self.hud.small_font, bg=(10,10,15,200))
            self.hud.draw_text(surface, mode_str,
                              (self.rect.centerx, self.rect.bottom - 18),
                              (200, 220, 255), self.hud.small_font, anchor='center')