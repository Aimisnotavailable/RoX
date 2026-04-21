import pygame as pg
from ui.base_panel import BasePanel
from ui.hud_helpers import RadialMenu

class RadialMenuPanel(BasePanel):
    def __init__(self, hud):
        super().__init__(hud)
        self.menu = RadialMenu()
    
    def update(self):
        pass
    
    def draw(self, surface):
        ar = self.engine.ar_controller
        if ar.radial_menu_active:
            self.menu.draw(surface, self.hud.pulse_timer)
    
    def activate(self, center, options):
        self.menu.activate(center, options)
    
    def deactivate(self):
        self.menu.deactivate()
    
    def update_selection(self, screen_point):
        self.menu.update_selection(screen_point)
    
    @property
    def active(self):
        return self.menu.active
    
    @property
    def selected_index(self):
        return self.menu.selected_index
    
    @property
    def current_options(self):
        return self.menu.current_options