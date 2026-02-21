import pygame
import moderngl
import sys
import glm
from PIL import Image
from src.rtscamera import RTSCamera
from src.fpscamera import FPSCamera 
from src.mesh import CubeMesh
from src.builder import VoxelBuilder
from src.quad import ScreenQuad  # <--- NEW
from src.builder import BLOCK_TYPES

class GraphicsEngine3D:
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
        self.ctx.enable(moderngl.BLEND) 
        
        self.clock = pygame.time.Clock()
        self.time = 0
        self.delta_time = 0
        
        # MODE SWITCHING
        # --- SINGLE BUILDER, TWO CAMERAS ---
        self.builder = VoxelBuilder(self)
        self.cam_fps = FPSCamera(self)
        self.cam_rts = RTSCamera(self)

        # Assets
        self.prog = self.create_shader_program('shaders/default')
        self.mesh = CubeMesh(self)

        # Load the new array instead of the old test.png
        self.texture_array = self.load_texture_array('assets/textures/tex_array_1.png')
        
        # Background
        self.quad = ScreenQuad(self)
        self.bg_texture = self.load_texture('assets/textures/sky.png') 
        self.use_aspect_ratio = True  # <--- NEW TOGGLE STATE
        
        # UI
        self.font = pygame.font.SysFont('arial', 30, bold=True)
        self.ui_surface = pygame.Surface(self.WIN_SIZE, flags=pygame.SRCALPHA)
        self.ui_texture = self.ctx.texture(self.WIN_SIZE, 4)

        # Start in FPS Mode
        self.is_rts_mode = False
        self.camera = self.cam_fps
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)


        # Controls
        self.clicking = False
        self.delay = 0

    def switch_camera_mode(self):
        """
        Toggle between RTS and FPS modes while preserving the player's FPS position.
        - When switching to RTS: save FPS state and base RTS camera on FPS position.
        - When switching back to FPS: restore the saved FPS state.
        """
        self.is_rts_mode = not self.is_rts_mode

        if self.is_rts_mode:
            # Going into RTS: save FPS state and position RTS above the FPS player
            print("Mode: RTS")
            # Save FPS so we can restore later
            self.cam_fps.save_state()
            # Base RTS camera on the FPS camera so the user doesn't get lost
            self.cam_rts.from_fps(self.cam_fps)
            self.camera = self.cam_rts
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
        else:
            # Going back to FPS: restore the saved FPS state
            print("Mode: FPS")
            self.cam_fps.restore_state()
            self.camera = self.cam_fps
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)


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
    
    def load_texture_array(self, path):
        try:
            img = Image.open(path).convert('RGBA')
            # remove the flip for now to keep logic simple - we can flip in shader or individual tiles
            # img = img.transpose(Image.FLIP_TOP_BOTTOM) 
            
            # 1. Define Tile Size MANUALLY
            # Since your image is a grid, we can't guess the size from the width.
            tile_size = 512 
            
            cols = img.width // tile_size
            rows = img.height // tile_size
            
            print(f"Loading Grid: {path}")
            print(f"  - Grid: {cols} cols x {rows} rows")
            
            # 2. Slice the Grid into separate tiles
            # OpenGL Array expects: [Tile 0 Bytes] + [Tile 1 Bytes] + ...
            tile_data = bytearray()
            
            for y in range(rows):
                for x in range(cols):
                    # Calculate coordinates
                    left = x * tile_size
                    upper = y * tile_size
                    right = left + tile_size
                    lower = upper + tile_size
                    
                    # Crop the tile
                    tile = img.crop((left, upper, right, lower))
                    
                    # Optional: Flip tile for OpenGL (Bottom-Left origin)
                    tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
                    
                    tile_data.extend(tile.tobytes())
            
            # 3. Create the Array
            # Total layers = rows * cols (e.g., 8 * 3 = 24 layers)
            num_layers = rows * cols
            
            texture_array = self.ctx.texture_array(
                (tile_size, tile_size, num_layers), 
                4, 
                tile_data
            )
            
            texture_array.build_mipmaps()
            texture_array.filter = (moderngl.NEAREST_MIPMAP_LINEAR, moderngl.NEAREST)
            return texture_array
            
        except Exception as e:
            print(f"Error: {e}")
            return None

    def create_shader_program(self, path):
        return self.load_program(path)

    def update_ui(self):
        """Draws Pygame text onto a Surface, then converts it to a GL Texture"""
        # 1. Clear the surface with transparent color (0,0,0,0)
        self.ui_surface.fill((0, 0, 0, 0))
        
        # 2. Draw Text (Standard Pygame)
        text = f"FPS: {int(self.clock.get_fps())} | Camera : {'RTS' if self.is_rts_mode else 'FPS'}"
        text_surf = self.font.render(text, True, (255, 255, 0)) # Yellow Text
        self.ui_surface.blit(text_surf, (20, 20))
        
        # 3. Draw Instructions
        current_data = BLOCK_TYPES[self.builder.current_block_index]
        block_name = current_data['name']
        text_surf = self.font.render(f"Selected: {block_name}", True, (255, 255, 255))
        self.ui_surface.blit(text_surf, (20, 100))

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
        # self.prog['cam_pos'].write(self.camera.position)

    def render(self):
        self.ctx.clear(0.0, 0.0, 0.0)
        

        # Background
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Pass the toggle state here!
        self.quad.render(self.bg_texture, 
                         image_size=self.bg_texture.size, 
                         keep_aspect=self.use_aspect_ratio)
                         
        # 3. Render 3D Scene (Depth Test ON)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.texture_array.use(location=0)
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
                
            # Mode Switch
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.switch_camera_mode()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.clicking = True    

                if event.button == 4:
                    self.builder.current_block_index = (self.builder.current_block_index + 1) % len(BLOCK_TYPES)
                if event.button == 5:
                    self.builder.current_block_index = (self.builder.current_block_index - 1) % len(BLOCK_TYPES)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.clicking = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                self.use_aspect_ratio = not self.use_aspect_ratio
        
        if self.clicking:
            if not self.delay:
                self.builder.handle_click()
                self.delay = 5
        self.delay = max(0, self.delay - 0.06 * self.delta_time)
        
    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.render()
            self.delta_time = self.clock.tick()