import pygame
import moderngl
import sys
import glm
from PIL import Image
from src.camera import FPSCamera, RTSCamera
from src.mesh import CubeMesh
from src.builder import VoxelBuilder
from src.quad import ScreenQuad
from src.builder import BLOCK_TYPES
from src.arinput import ARInputHandler
from scripts.ar import AR

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
        self.builder = VoxelBuilder(self)
        self.cam_fps = FPSCamera(self)
        self.cam_rts = RTSCamera(self)

        # Assets
        self.prog = self.create_shader_program('shaders/default')
        self.mesh = CubeMesh(self)
        self.texture_array = self.load_texture_array('assets/textures/tex_array_1.png')
        
        # Background
        self.quad = ScreenQuad(self)
        self.bg_texture = self.load_texture('assets/textures/sky.png') 
        self.use_aspect_ratio = True
        
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
        
        # AR
        self.ar = AR()
        self.input_handler = ARInputHandler(self)
        
        # AR Interaction State
        self.last_pinch_dist = None
        self.ar_cursor_pos = None 
        self.last_rotate_pos = None
        self.last_build_hand_pos = None  # <--- NEW: For tracking drag velocity

    def switch_camera_mode(self):
        self.is_rts_mode = not self.is_rts_mode

        if self.is_rts_mode:
            print("Mode: RTS")
            self.cam_fps.save_state()
            self.cam_rts.from_fps(self.cam_fps)
            self.camera = self.cam_rts
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)
        else:
            print("Mode: FPS")
            self.cam_fps.restore_state()
            self.camera = self.cam_fps
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)

    def convert_hand_inputs_to_world_inputs(self):
        """
        Interprets ARInputHandler states into 3D camera/builder actions.
        """
        l_state = self.input_handler.left_state
        r_state = self.input_handler.right_state
        l_pos = self.input_handler.left_finger_ema
        r_pos = self.input_handler.right_finger_ema
        
        # Always update cursor if Right Hand is visible
        if r_pos:
            self.ar_cursor_pos = r_pos
        else:
            self.ar_cursor_pos = None

        # -------------------------
        # CASE 1: ZOOM (Both Hands Pinched)
        # -------------------------
        if l_state.active and r_state.active and l_pos and r_pos:
            dx = l_pos[0] - r_pos[0]
            dy = l_pos[1] - r_pos[1]
            current_dist = (dx**2 + dy**2)**0.5
            
            if self.last_pinch_dist is not None:
                delta = current_dist - self.last_pinch_dist
                self.camera.position += self.camera.forward * (delta * 0.05)
                
            self.last_pinch_dist = current_dist
            self.last_rotate_pos = None
            self.clicking = False
            self.builder.stop_raycast = False
            return
        else:
            self.last_pinch_dist = None

        # -------------------------
        # CASE 2: ROTATE WORLD (Left Hand Pinched)
        # -------------------------
        if l_state.active and l_pos:
            if self.last_rotate_pos is None:
                self.last_rotate_pos = l_pos
            
            dx = l_pos[0] - self.last_rotate_pos[0]
            dy = l_pos[1] - self.last_rotate_pos[1]
            
            rot_speed = 0.005
            self.builder.rotation.y += dx * rot_speed
            self.builder.rotation.x += dy * rot_speed
            
            self.last_rotate_pos = l_pos
        else:
            self.last_rotate_pos = None

        # -------------------------
        # CASE 3: BUILD/SNAP (Right Hand Pinched)
        # -------------------------
        if r_state.active and r_pos:
            self.clicking = True
            self.builder.stop_raycast = True 
            
            # --- FIX: Calculate Hand Velocity (Delta) for Snapping ---
            if self.last_build_hand_pos is None:
                # First frame of pinch, no movement yet
                self.last_build_hand_pos = r_pos
                rel_x, rel_y = 0, 0
            else:
                # Calculate movement since last frame
                rel_x = r_pos[0] - self.last_build_hand_pos[0]
                rel_y = r_pos[1] - self.last_build_hand_pos[1]
            
            # Inject this into Camera so Builder can see "Mouse Movement"
            self.camera.movement_rel = (rel_x, rel_y)
            
            # Store current pos for next frame
            self.last_build_hand_pos = r_pos
            
        else:
            self.clicking = False
            self.builder.stop_raycast = False
            self.last_build_hand_pos = None # Reset history

    def load_program(self, path):
        with open(f'{path}.vert') as f: vertex_src = f.read()
        with open(f'{path}.frag') as f: fragment_src = f.read()
        return self.ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)

    def load_texture(self, path):
        try:
            img = Image.open(path).convert('RGBA')
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
            tile_size = 512 
            cols = img.width // tile_size
            rows = img.height // tile_size
            tile_data = bytearray()
            for y in range(rows):
                for x in range(cols):
                    left = x * tile_size
                    upper = y * tile_size
                    right = left + tile_size
                    lower = upper + tile_size
                    tile = img.crop((left, upper, right, lower))
                    tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
                    tile_data.extend(tile.tobytes())
            num_layers = rows * cols
            texture_array = self.ctx.texture_array((tile_size, tile_size, num_layers), 4, tile_data)
            texture_array.build_mipmaps()
            texture_array.filter = (moderngl.NEAREST_MIPMAP_LINEAR, moderngl.NEAREST)
            return texture_array
        except Exception as e:
            print(f"Error: {e}")
            return None

    def create_shader_program(self, path):
        return self.load_program(path)

    def draw_ui_text(self):
        text = f"FPS: {int(self.clock.get_fps())} | Camera : {'RTS' if self.is_rts_mode else 'FPS'}"
        text_surf = self.font.render(text, True, (255, 255, 0))
        self.ui_surface.blit(text_surf, (20, 20))
        
        current_data = BLOCK_TYPES[self.builder.current_block_index]
        block_name = current_data['name']
        text_surf = self.font.render(f"Selected: {block_name}", True, (255, 255, 255))
        self.ui_surface.blit(text_surf, (20, 100))

    def update(self):
        # 1. Update AR State
        self.input_handler.update(self.ar.ar_data)
        
        # 2. Update Camera (Calculate Matrices & Mouse Input)
        self.camera.update()
        self.camera.move()

        # 3. OVERRIDE Inputs with AR (Must happen AFTER camera.update)
        #    This ensures our hand calculated 'movement_rel' overwrites the stationary mouse.
        if self.is_rts_mode and self.ar.ar_data.get("HAND_PRESENCE", False):
            self.convert_hand_inputs_to_world_inputs()
        
        # 4. Update Builder
        self.builder.update(screen_pos=self.ar_cursor_pos)
        
        # 5. Handle Clicking / Snapping
        if self.clicking:
            if not self.delay:
                self.builder.handle_click()
                self.delay = 5  # Speed of placing blocks (lower = faster)
        self.delay = max(0, self.delay - 0.06 * self.delta_time)
        
        self.time = pygame.time.get_ticks() * 0.001
        
        self.prog['m_proj'].write(self.camera.m_proj)
        self.prog['m_view'].write(self.camera.m_view)
        self.prog['light_pos'].write(glm.vec3(0, 5, 5))
    
    def render_2d(self):        
        flipped_data = pygame.image.tostring(pygame.transform.flip(self.ui_surface, False, True), 'RGBA')
        self.ui_texture.write(flipped_data)

    def render(self):
        self.ctx.clear(0.0, 0.0, 0.0)
        
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ui_surface.fill((0, 0, 0, 0))
        self.ar.render(self.ui_surface)
        self.draw_ui_text()
        self.render_2d()
        self.ctx.disable(moderngl.DEPTH_TEST)

        self.quad.render(self.ui_texture)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.texture_array.use(location=0)
        self.builder.render()
                
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.mesh.vbo.release()
                self.prog.release()
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                self.builder.toggle_mode()
                
            if event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                self.switch_camera_mode()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.clicking = True
                    if self.is_rts_mode:
                        self.builder.stop_raycast = True    

                if event.button == 4:
                    self.builder.current_block_index = (self.builder.current_block_index + 1) % len(BLOCK_TYPES)
                if event.button == 5:
                    self.builder.current_block_index = (self.builder.current_block_index - 1) % len(BLOCK_TYPES)

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.clicking = False
                    self.builder.stop_raycast = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                self.use_aspect_ratio = not self.use_aspect_ratio
        
    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.render()
            self.delta_time = self.clock.tick()