import pygame
import moderngl
import sys
import glm
from PIL import Image
from src.camera import Camera
from src.mesh import CubeMesh
from src.builder import VoxelBuilder
from src.quad import ScreenQuad  # <--- NEW

class GraphicsEngine:
    def __init__(self, win_size=(1280, 720)):
        pygame.init()
        self.WIN_SIZE = win_size
        
        # OpenGL Setup
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.set_mode(self.WIN_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
        
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        # Enable Blending for UI transparency
        self.ctx.enable(moderngl.BLEND) 
        
        self.clock = pygame.time.Clock()
        self.time = 0
        self.delta_time = 0
        
        # --- 1. Load Main Assets ---
        self.prog = self.create_shader_program('shaders/default')
        self.camera = Camera(self)
        self.mesh = CubeMesh(self)
        self.texture = self.load_texture('assets/textures/test.png')
        self.builder = VoxelBuilder(self)
        
        # --- 2. Background Setup ---
        self.quad = ScreenQuad(self)
        # Replace with your image file
        self.bg_texture = self.load_texture('assets/textures/sky.png') 
        
        # --- 3. UI Setup (Pygame Surface) ---
        self.font = pygame.font.SysFont('arial', 30, bold=True)
        # Create a surface with 'SRGB' (standard colors) and 'Alpha' (transparency)
        self.ui_surface = pygame.Surface(self.WIN_SIZE, flags=pygame.SRCALPHA)
        # Create a texture from that surface
        self.ui_texture = self.ctx.texture(self.WIN_SIZE, 4)

    def load_program(self, path):
        # Helper to load shaders easily
        with open(f'{path}.vert') as f: vertex_src = f.read()
        with open(f'{path}.frag') as f: fragment_src = f.read()
        return self.ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)

    def load_texture(self, path):
        try:
            img = Image.open(path).convert('RGBA') # Force RGBA
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            texture = self.ctx.texture(img.size, 4, img.tobytes())
            texture.build_mipmaps()
            return texture
        except FileNotFoundError:
            print(f"Error: {path} not found.")
            return self.ctx.texture((2,2), 4, b'\xff'*16)

    def create_shader_program(self, path):
        return self.load_program(path)

    def update_ui(self):
        """Draws Pygame text onto a Surface, then converts it to a GL Texture"""
        # 1. Clear the surface with transparent color (0,0,0,0)
        self.ui_surface.fill((0, 0, 0, 0))
        
        # 2. Draw Text (Standard Pygame)
        text = f"FPS: {int(self.clock.get_fps())} | Mode: {self.builder.mode}"
        text_surf = self.font.render(text, True, (255, 255, 0)) # Yellow Text
        self.ui_surface.blit(text_surf, (20, 20))
        
        # 3. Draw Instructions
        help_text = self.font.render("TAB: Toggle Mode | Left Click: Action", True, (255, 255, 255))
        self.ui_surface.blit(help_text, (20, 60))

        # 4. Upload this surface data to the GPU Texture
        # We must flip it because OpenGL expects bottom-left origin
        flipped_data = pygame.image.tostring(pygame.transform.flip(self.ui_surface, False, True), 'RGBA')
        self.ui_texture.write(flipped_data)

    def update(self):
        self.camera.update()
        self.camera.move()
        self.builder.update()
        self.time = pygame.time.get_ticks() * 0.001
        
        # Update 3D Uniforms
        self.prog['m_proj'].write(self.camera.m_proj)
        self.prog['m_view'].write(self.camera.m_view)
        self.prog['light_pos'].write(glm.vec3(0, 5, 5))
        self.prog['cam_pos'].write(self.camera.position)

    def render(self):
        # 1. Clear Screen
        self.ctx.clear(0.0, 0.0, 0.0)

        # 2. Render Background (Depth Test OFF)
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # FIX: Pass the actual size of the background image!
        # self.bg_texture.size returns (width, height)
        self.quad.render(self.bg_texture, image_size=self.bg_texture.size)
        
        # 3. Render 3D Scene (Depth Test ON)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.texture.use(location=0)
        self.builder.render()
        
        # 4. Render UI Overlay (Depth Test OFF, Blending ON)
        self.update_ui() 
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # The UI texture is the same size as the screen, so we don't need to pass a size
        # (It will default to screen size in the quad class)
        self.quad.render(self.ui_texture)
        
        pygame.display.flip()

    def handle_events(self):
        # ... (Keep existing code) ...
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                # Clean up resources
                self.mesh.vbo.release()
                self.prog.release()
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                self.builder.toggle_mode()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    self.builder.handle_click()
    
    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.render()
            self.delta_time = self.clock.tick(60)