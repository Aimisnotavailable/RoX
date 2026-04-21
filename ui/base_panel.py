from abc import ABC, abstractmethod
import pygame as pg

class BasePanel(ABC):
    """Abstract base for all HUD panels."""
    
    def __init__(self, hud):
        self.hud = hud
        self.engine = hud.engine
        self.visible = True
        self.rect = pg.Rect(0, 0, 0, 0)
    
    @abstractmethod
    def update(self):
        """Update panel state (called once per frame)."""
        pass
    
    @abstractmethod
    def draw(self, surface: pg.Surface):
        """Draw the panel onto the given surface."""
        pass
    
    def handle_event(self, event):
        """Optional event handling."""
        pass