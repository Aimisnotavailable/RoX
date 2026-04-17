import pygame as pg
from settings import INTERACTION_MODE
from ui.base_panel import BasePanel

class InfoPanel(BasePanel):
    def __init__(self, hud):
        super().__init__(hud)
        self.rect = pg.Rect(20, 20, 300, 150)
    
    def update(self):
        pass
    
    def draw(self, surface):
        if not self.visible:
            return
        
        # Draw glass panel
        self.hud.draw_glass_panel(surface, self.rect, glow=True)
        
        # Get data
        fps = self.engine.clock.get_fps()
        vh = self.engine.scene.world_container.voxel_handler
        block_map = {1: "SAND", 2: "GRASS", 3: "DIRT", 4: "STONE", 5: "SNOW", 6: "LEAVES", 7: "WOOD"}
        current_block = block_map.get(vh.new_voxel_id, "UNKNOWN")
        mode_str = INTERACTION_MODE[vh.interaction_mode]
        world = self.engine.scene.world_container.local_worlds[0]
        
        lines = [
            f"FPS: {fps:.0f}",
            f"BLOCK: {current_block}",
            f"MODE: {mode_str}",
            f"SCALE: {world.world_scale:.2f}x",
        ]
        y = self.rect.y + 20
        for line in lines:
            self.hud.draw_text(surface, line, (self.rect.x + 20, y), (220, 230, 255), self.hud.font)
            y += 22