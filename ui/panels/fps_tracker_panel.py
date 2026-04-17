import pygame as pg
from ui.base_panel import BasePanel
from ui.hud_helpers import draw_rounded_rect

class FPSTrackerPanel(BasePanel):
    def __init__(self, hud):
        super().__init__(hud)
        self.fps_values = []
        self.max_len = 60
        self.rect = pg.Rect(hud.res[0] - 140, 20, 120, 40)
    
    def update(self):
        fps = self.engine.clock.get_fps()
        self.fps_values.append(fps)
        if len(self.fps_values) > self.max_len:
            self.fps_values.pop(0)
    
    def draw(self, surface):
        if len(self.fps_values) < 2:
            return
        draw_rounded_rect(surface, self.rect, (10, 12, 18, 200), radius=4)
        max_fps = max(max(self.fps_values), 30)
        points = []
        for i, val in enumerate(self.fps_values):
            x = self.rect.x + (i / len(self.fps_values)) * self.rect.width
            y = self.rect.y + self.rect.height - (val / max_fps) * self.rect.height
            points.append((x, y))
        if len(points) > 1:
            pg.draw.lines(surface, (100, 200, 100), False, points, 2)