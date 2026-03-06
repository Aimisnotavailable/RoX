import pygame
import sys
from src.builder import BLOCK_TYPES

class AppInputHandler:
    def __init__(self, engine):
        self.engine = engine

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.engine.mesh.vbo.release()
                self.engine.prog.release()
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                self.engine.builder.toggle_mode()
                
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.engine.switch_camera_mode()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.engine.clicking = True
                    if self.engine.is_rts_mode:
                        self.engine.builder.stop_raycast = True    
                if event.button == 4:
                    self.engine.builder.current_block_index = (self.engine.builder.current_block_index + 1) % len(BLOCK_TYPES)
                if event.button == 5:
                    self.engine.builder.current_block_index = (self.engine.builder.current_block_index - 1) % len(BLOCK_TYPES)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.engine.clicking = False
                    self.engine.builder.stop_raycast = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                self.engine.use_aspect_ratio = not self.engine.use_aspect_ratio