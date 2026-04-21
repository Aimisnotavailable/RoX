import pygame as pg
from ui.base_panel import BasePanel

class HandStatusPanel(BasePanel):
    def __init__(self, hud):
        super().__init__(hud)
        self.left_color = (140, 140, 160)
        self.right_color = (140, 140, 160)
        self.left_status = "STANDBY"
        self.right_status = "AIM"
    
    def update(self):
        ar = self.engine.ar_controller
        vh = self.engine.scene.world_container.voxel_handler
        
        # Left hand
        if ar.two_finger_up_left_active:
            self.left_status = "ROTATE"
            self.left_color = (80, 180, 255)
        elif ar.pinch_active_left:
            if ar.radial_menu_active:
                self.left_status = "SELECT"
                self.left_color = (255, 200, 50)
            else:
                self.left_status = "HOLD"
                self.left_color = (255, 140, 60)
        else:
            self.left_status = "STANDBY"
            self.left_color = (140, 140, 160)
        
        # Right hand
        if ar.two_finger_up_right_active:
            self.right_status = "LOOK"
            self.right_color = (80, 180, 255)
        elif ar.pinch_active_right:
            if vh.is_dragging:
                self.right_status = "EXTRUDE"
            else:
                self.right_status = "BUILD"
            self.right_color = (100, 255, 100)
        elif ar.is_grabbing:
            self.right_status = "GRAB"
            self.right_color = (255, 180, 80)
        else:
            self.right_status = "AIM"
            self.right_color = (140, 140, 160)
    
    def draw(self, surface):
        if not self.visible:
            return
        
        res = self.hud.res
        left_x, left_y = 70, res[1] - 85
        right_x, right_y = res[0] - 70, res[1] - 85
        
        self.hud.draw_circular_status(surface, left_x, left_y, "LEFT", self.left_status, self.left_color)
        self.hud.draw_circular_status(surface, right_x, right_y, "RIGHT", self.right_status, self.right_color)