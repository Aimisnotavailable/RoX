import pygame as pg
from ui.base_panel import BasePanel

class CrosshairPanel(BasePanel):
    def __init__(self, hud):
        super().__init__(hud)
    
    def update(self):
        pass
    
    def draw(self, surface):
        ar = self.engine.ar_controller
        vh = self.engine.scene.world_container.voxel_handler
        
        left_pos = ar.smooth_left_pos
        right_pos = ar.smooth_right_pos
        
        if left_pos is not None:
            px, py = self.hud.get_screen_coords(left_pos)
            sub = f"Z:{left_pos.z:.2f}"
            self.hud.draw_crosshair(surface, px, py, self._get_left_color(),
                                   ar.pinch_active_left, self._get_left_status(), sub)
        
        if right_pos is not None:
            px, py = self.hud.get_screen_coords(right_pos)
            sub = f"Z:{right_pos.z:.2f}"
            self.hud.draw_crosshair(surface, px, py, self._get_right_color(),
                                   ar.pinch_active_right, self._get_right_status(), sub)
            if ar.pinch_active_right and vh.is_dragging:
                self.hud.draw_text(surface, f"x{vh.brush_mult:.2f}",
                                  (px + 45, py - 45), (255, 255, 80),
                                  self.hud.small_font, anchor='center')
    
    def _get_left_status(self):
        # Delegate to hand status panel or recompute
        return "STANDBY"
    
    def _get_left_color(self):
        return (140, 140, 160)
    
    def _get_right_status(self):
        return "AIM"
    
    def _get_right_color(self):
        return (140, 140, 160)