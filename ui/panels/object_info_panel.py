import pygame as pg
import glm
import math
from ui.base_panel import BasePanel

class ObjectInfoPanel(BasePanel):
    def __init__(self, hud):
        super().__init__(hud)
        self.rect = pg.Rect(20, 180, 300, 100)
    
    def update(self):
        pass
    
    def draw(self, surface):
        pass
        # selected = self.engine.scene.global_scene.selected_object
        # if not selected or not self.visible:
        #     return
        
        # self.hud.draw_glass_panel(surface, self.rect)
        # pos = selected.position
        # euler = glm.eulerAngles(selected.rotation)
        
        # self.hud.draw_text(surface, f"OBJ: {type(selected).__name__}",
        #                   (self.rect.x + 20, self.rect.y + 15), (255, 200, 100), self.hud.font)
        # self.hud.draw_text(surface, f"Pos: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})",
        #                   (self.rect.x + 20, self.rect.y + 35), (200, 220, 255), self.hud.small_font)
        # self.hud.draw_text(surface, f"Rot: ({math.degrees(euler.x):.0f}, {math.degrees(euler.y):.0f}, {math.degrees(euler.z):.0f})",
        #                   (self.rect.x + 20, self.rect.y + 55), (200, 220, 255), self.hud.small_font)
        # self.hud.draw_text(surface, f"Scl: {selected.scale.x:.2f}",
        #                   (self.rect.x + 20, self.rect.y + 75), (200, 220, 255), self.hud.small_font)