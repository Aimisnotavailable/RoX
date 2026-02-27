import pygame
import sys
import random
import math
import time

from src.arinput import ARInputHandler 
from scripts.config import *
from scripts.ar import AR
from src.camera import Camera2D

class BlockWorld:
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        self.blocks = set()
        
        # Create some random initial blocks
        for x in range(20):
            for y in range(20):
                if random.random() < 0.1: 
                    self.blocks.add((x, y))

    def world_to_grid(self, wx, wy):
        gx = int(round(wx / self.grid_size))
        gy = int(round(wy / self.grid_size))
        return (gx, gy)

    def add_block_at_world(self, wx, wy):
        gx, gy = self.world_to_grid(wx, wy)
        if (gx, gy) in self.blocks:
            return False
        self.blocks.add((gx, gy))
        return True
        
    def remove_block_at_world(self, wx, wy):
        gx, gy = self.world_to_grid(wx, wy)
        if (gx, gy) in self.blocks:
            self.blocks.remove((gx, gy))
            return True
        return False

class GraphicsEngine2D:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 1280, 720
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("RoX 2D Mode")
        self.clock = pygame.time.Clock()

        # AR & Input Setup
        self.ar = AR()
        self.input_handler = ARInputHandler(self) 

        # Game World
        self.camera = Camera2D(pos=(0, 0), zoom=1.0)
        self.block_world = BlockWorld()
        
        # Interaction State
        self.camera_target = [0, 0]
        self.last_zoom_dist = None
        self.last_pan_pos = None
        self.debug_mode = True

    def handle_input(self):
        # 1. Update AR Input Handler
        self.input_handler.update(self.ar.ar_data)
        
        # 2. Get Data from Handler
        l_state = self.input_handler.left_state
        r_state = self.input_handler.right_state
        l_pos = self.input_handler.left_finger_ema
        r_pos = self.input_handler.right_finger_ema
        
        # ZOOM (Two Hands)
        if l_state.active and r_state.active and l_pos and r_pos:
            dx = l_pos[0] - r_pos[0]
            dy = l_pos[1] - r_pos[1]
            dist = math.hypot(dx, dy)
            
            if self.last_zoom_dist is not None:
                delta = dist - self.last_zoom_dist
                self.camera.zoom += delta * 0.01
                self.camera.zoom = max(0.1, min(5.0, self.camera.zoom))
            
            self.last_zoom_dist = dist
            self.last_pan_pos = None 
            return 
        else:
            self.last_zoom_dist = None

        # PAN CAMERA (Left Hand Pinch)
        if l_state.active and l_pos:
            if self.last_pan_pos is None:
                self.last_pan_pos = l_pos
            
            dx = l_pos[0] - self.last_pan_pos[0]
            dy = l_pos[1] - self.last_pan_pos[1]
            
            pan_speed = 2.0 / self.camera.zoom 
            self.camera.pos[0] -= dx * pan_speed
            self.camera.pos[1] -= dy * pan_speed
            
            self.last_pan_pos = l_pos
        else:
            self.last_pan_pos = None

        # BUILD/DELETE (Right Hand)
        gesture = self.ar.ar_data.get("GESTURES", {}).get("RIGHT", "NONE")
        
        if r_state.active and r_pos:
            wx, wy = self.camera.screen_to_world(r_pos[0], r_pos[1], self.WIDTH, self.HEIGHT)
            
            if gesture == "OPEN_PALM":
                 self.block_world.remove_block_at_world(wx, wy)
            else:
                 self.block_world.add_block_at_world(wx, wy)

    def draw_grid(self):
        cam_x, cam_y = self.camera.pos
        zoom = self.camera.zoom
        grid_sz = self.block_world.grid_size
        
        for gx, gy in self.block_world.blocks:
            wx = gx * grid_sz
            wy = gy * grid_sz
            
            sx, sy = self.camera.world_to_screen(wx, wy, self.WIDTH, self.HEIGHT)
            size = grid_sz * zoom
            
            if -size < sx < self.WIDTH + size and -size < sy < self.HEIGHT + size:
                # Draw Semi-Transparent Blocks so we can see AR feed behind?
                # Pygame rects don't support alpha directly unless drawing to surface
                # For performance, we stick to solid for now, or use a shape
                
                # Main Block
                pygame.draw.rect(self.screen, (100, 200, 100), (sx - size/2, sy - size/2, size, size))
                # Border
                pygame.draw.rect(self.screen, (50, 100, 50), (sx - size/2, sy - size/2, size, size), 2)

    def render(self):
        self.screen.fill((0, 0, 0, 0))
        # 2. Draw blocks on top of camera feed
        # CORE LOOP
        # 1. Draw AR Feed FIRST (This acts as the background clear)
        self.ar.render(self.screen) 
        
        # 2. Logic
        self.handle_input()
        self.draw_grid()
        
        # 3. Draw AR Cursors (Visual Feedback)
        l_pos = self.input_handler.left_finger_ema
        r_pos = self.input_handler.right_finger_ema
        l_act = self.input_handler.left_state.active
        r_act = self.input_handler.right_state.active

        if l_pos:
            col = (0, 255, 255) if l_act else (0, 100, 100)
            pygame.draw.circle(self.screen, col, (int(l_pos[0]), int(l_pos[1])), 10)
            if l_act: pygame.draw.circle(self.screen, (255, 255, 255), (int(l_pos[0]), int(l_pos[1])), 15, 2)

        if r_pos:
            gesture = self.ar.ar_data.get("GESTURES", {}).get("RIGHT", "NONE")
            if gesture == "OPEN_PALM":
                col = (255, 50, 50) # Red for delete
            else:
                col = (0, 255, 0) if r_act else (0, 100, 0) # Green for build
            
            pygame.draw.circle(self.screen, col, (int(r_pos[0]), int(r_pos[1])), 10)
            if r_act: pygame.draw.circle(self.screen, (255, 255, 255), (int(r_pos[0]), int(r_pos[1])), 15, 2)

        # 4. UI / Debug
        if self.debug_mode:
            fps = int(self.clock.get_fps())
            font = pygame.font.SysFont('Arial', 18)
            info = [
                f"FPS: {fps}",
                f"Zoom: {self.camera.zoom:.2f}",
                f"Blocks: {len(self.block_world.blocks)}",
                f"Right Gesture: {self.ar.ar_data.get('GESTURES', {}).get('RIGHT', 'NONE')}"
            ]
            for i, text in enumerate(info):
                surf = font.render(text, True, (255, 255, 0))
                self.screen.blit(surf, (10, 10 + i * 20))

        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.ar.stop()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.ar.stop()
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_d:
                        self.debug_mode = not self.debug_mode

            self.render()
            
            self.clock.tick(60)