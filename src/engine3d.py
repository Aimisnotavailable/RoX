import pygame
import moderngl
import sys
import glm
import math
import random
from PIL import Image
from src.camera import FPSCamera, RTSCamera
from src.mesh import CubeMesh
from src.builder import VoxelBuilder
from src.quad import ScreenQuad
from src.builder import BLOCK_TYPES
from src.arinput import ARInputHandler
from scripts.ar import AR

# --- HUD COLORS ---
COLOR_ZOOM = (255, 200, 0)   # Amber
COLOR_ROTATE = (0, 255, 255) # Cyan
COLOR_BUILD = (0, 255, 0)    # Green
COLOR_TEXT = (255, 255, 255)
COLOR_UI_BG = (0, 0, 0, 150) # Semi-transparent black

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
        self.quad = ScreenQuad(self)
        
        # UI & Fonts
        self.font = pygame.font.SysFont('arial', 20, bold=True)
        self.font_large = pygame.font.SysFont('arial', 48, bold=True)
        
        # SURFACES & TEXTURES
        self.ui_surface = pygame.Surface(self.WIN_SIZE, flags=pygame.SRCALPHA)
        self.feed_surface = pygame.Surface(self.WIN_SIZE, pygame.SRCALPHA)
        self.ui_texture = self.ctx.texture(self.WIN_SIZE, 4)
        self.feed_texture = self.ctx.texture(self.WIN_SIZE, 3)

        # Start in FPS Mode
        self.is_rts_mode = False
        self.camera = self.cam_fps
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

        # Controls
        self.clicking = False
        self.delay = 0
        self.current_action_label = "" 
        
        # AR System
        self.ar = AR()
        self.input_handler = ARInputHandler(self)
        
        # AR Interaction State
        self.last_pinch_dist = None
        self.ar_cursor_pos = None 
        self.last_rotate_pos = None
        self.last_build_hand_pos = None
        
        self.trails = {'left': [], 'right': []}
        
        # Visual Effects State
        self.zoom_line_visible = False
        self.zoom_line_coords = ((0,0), (0,0))
        self.zoom_pct_display = 0

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

    def update_trails(self, l_pos, r_pos):
        if l_pos:
            self.trails['left'].append(l_pos)
            if len(self.trails['left']) > 10: self.trails['left'].pop(0)
        else:
            self.trails['left'].clear()
        
        if r_pos:
            self.trails['right'].append(r_pos)
            if len(self.trails['right']) > 10: self.trails['right'].pop(0)
        else:
            self.trails['right'].clear()

    def convert_hand_inputs_to_world_inputs(self):
        l_state = self.input_handler.left_state
        r_state = self.input_handler.right_state
        l_pos = self.input_handler.left_finger_ema
        r_pos = self.input_handler.right_finger_ema
        
        self.update_trails(l_pos, r_pos)
        self.ar_cursor_pos = r_pos if r_pos else None
        self.current_action_label = "" 
        
        # Reset visual flags
        self.zoom_line_visible = False

        # ZOOM
        if l_state.active and r_state.active and l_pos and r_pos:
            self.current_action_label = "ZOOMING"
            dx = l_pos[0] - r_pos[0]
            dy = l_pos[1] - r_pos[1]
            current_dist = (dx**2 + dy**2)**0.5
            
            # Store coords for rendering in render_hud()
            self.zoom_line_visible = True
            self.zoom_line_coords = (l_pos, r_pos)
            self.zoom_pct_display = int(current_dist / 5)

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

        # ROTATE
        if l_state.active and l_pos:
            self.current_action_label = "ROTATING"
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

        # BUILD
        if r_state.active and r_pos:
            self.current_action_label = "BUILDING"
            self.clicking = True
            self.builder.stop_raycast = True 
            
            if self.last_build_hand_pos is None:
                self.last_build_hand_pos = r_pos
                rel_x, rel_y = 0, 0
            else:
                # RAW MOVEMENT RESTORED (No Damping)
                rel_x = r_pos[0] - self.last_build_hand_pos[0]
                rel_y = r_pos[1] - self.last_build_hand_pos[1]
            
            self.camera.movement_rel = (rel_x, rel_y)
            self.last_build_hand_pos = r_pos
        else:
            self.clicking = False
            self.builder.stop_raycast = False
            self.last_build_hand_pos = None

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
        
    def draw_dynamic_cursor(self, pos_list, color, active):
        if not pos_list: return
        if len(pos_list) > 2:
            pygame.draw.lines(self.ui_surface, color, False, pos_list, 3)
        current_pos = pos_list[-1]
        radius = 15 if active else 20
        width = 0 if active else 3
        pygame.draw.circle(self.ui_surface, (0,0,0,100), (int(current_pos[0])+2, int(current_pos[1])+2), radius)
        pygame.draw.circle(self.ui_surface, color, (int(current_pos[0]), int(current_pos[1])), radius, width)

    def render_hud(self):
        # 1. Zoom Line (Draw this first so text is on top)
        if self.zoom_line_visible:
            l_pos, r_pos = self.zoom_line_coords
            start_pos = (int(l_pos[0]), int(l_pos[1]))
            end_pos = (int(r_pos[0]), int(r_pos[1]))
            
            # Draw Main Beam
            pygame.draw.line(self.ui_surface, COLOR_ZOOM, start_pos, end_pos, 4)
            # Draw Glow
            pygame.draw.line(self.ui_surface, (255, 255, 200, 100), start_pos, end_pos, 8)
            
            mid_x = (start_pos[0] + end_pos[0]) // 2
            mid_y = (start_pos[1] + end_pos[1]) // 2
            text_surf = self.font.render(f"{self.zoom_pct_display}%", True, COLOR_ZOOM)
            self.ui_surface.blit(text_surf, (mid_x - 20, mid_y - 40))

        # 2. Action Badge
        if self.current_action_label:
            color = COLOR_TEXT
            if self.current_action_label == "ZOOMING": color = COLOR_ZOOM
            elif self.current_action_label == "ROTATING": color = COLOR_ROTATE
            elif self.current_action_label == "BUILDING": color = COLOR_BUILD
            
            txt_surf = self.font_large.render(self.current_action_label, True, color)
            x_pos = (self.WIN_SIZE[0] - txt_surf.get_width()) // 2
            shadow_surf = self.font_large.render(self.current_action_label, True, (0,0,0))
            self.ui_surface.blit(shadow_surf, (x_pos+2, 52))
            self.ui_surface.blit(txt_surf, (x_pos, 50))

        # 3. Block Info
        current_data = BLOCK_TYPES[self.builder.current_block_index]
        block_name = current_data['name']
        pygame.draw.rect(self.ui_surface, COLOR_UI_BG, (10, 80, 200, 35), border_radius=5)
        text_surf = self.font.render(f"Block: {block_name}", True, COLOR_TEXT)
        self.ui_surface.blit(text_surf, (20, 86))

        # 4. FPS & Mode
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        mode_text = f"Mode: {'AR / RTS' if self.is_rts_mode else 'FPS'}"
        box_w = 220
        box_h = 60
        x_base = self.WIN_SIZE[0] - box_w - 10
        pygame.draw.rect(self.ui_surface, COLOR_UI_BG, (x_base, 10, box_w, box_h), border_radius=5)
        self.ui_surface.blit(self.font.render(fps_text, True, COLOR_ZOOM), (x_base + 10, 15))
        self.ui_surface.blit(self.font.render(mode_text, True, COLOR_TEXT), (x_base + 10, 40))

        # 5. Compass
        cx, cy = 60, self.WIN_SIZE[1] - 60
        radius = 30
        pygame.draw.circle(self.ui_surface, COLOR_UI_BG, (cx, cy), radius)
        pygame.draw.circle(self.ui_surface, (200, 200, 200), (cx, cy), radius, 2)
        angle = self.builder.rotation.y
        end_x = cx + math.sin(angle) * radius
        end_y = cy + math.cos(angle) * radius 
        pygame.draw.line(self.ui_surface, (255, 50, 50), (cx, cy), (end_x, end_y), 3)
        self.ui_surface.blit(self.font.render("N", True, (255, 50, 50)), (end_x-5, end_y-10))
        
        # 6. Cursors
        l_active = self.input_handler.left_state.active
        self.draw_dynamic_cursor(self.trails['left'], COLOR_ROTATE, l_active)
        r_active = self.input_handler.right_state.active
        self.draw_dynamic_cursor(self.trails['right'], COLOR_BUILD, r_active)

    def update(self):
        # 1. Update AR State
        self.input_handler.update(self.ar.ar_data)
        
        # 2. Camera Physics
        self.camera.update()
        self.camera.move()
        
        # 3. Game Logic
        if self.is_rts_mode and self.ar.ar_data.get("HAND_PRESENCE", False):
            self.convert_hand_inputs_to_world_inputs()
        else:
            self.trails['left'].clear()
            self.trails['right'].clear()
            self.zoom_line_visible = False
        
        # 4. Builder Logic
        self.builder.update(screen_pos=self.ar_cursor_pos)
        
        if self.clicking:
            if not self.delay:
                self.builder.handle_click()
                self.delay = 5  
        self.delay = max(0, self.delay - 0.06 * self.delta_time)
        
        self.time = pygame.time.get_ticks() * 0.001
        
        self.prog['m_proj'].write(self.camera.m_proj)
        self.prog['m_view'].write(self.camera.m_view)
        self.prog['light_pos'].write(glm.vec3(0, 5, 5))
    
    def render_feed_to_texture(self):
        if self.ar.image:
            self.feed_surface.blit(self.ar.image, (0, 0))
            img_flipped = pygame.transform.flip(self.feed_surface, False, True)
            data = pygame.image.tostring(img_flipped, 'RGB')
            self.feed_texture.write(data)

    def render_ui_to_texture(self):        
        flipped_data = pygame.image.tostring(pygame.transform.flip(self.ui_surface, False, True), 'RGBA')
        self.ui_texture.write(flipped_data)

    def render(self):
        # 1. CLEAR
        self.ctx.clear(0.0, 0.0, 0.0)
        self.ui_surface.fill((0, 0, 0, 0)) 
        
        # 2. HAND DRAWING
        self.ar.render(self.ui_surface) 
        
        # 3. HUD (Draws ON TOP of hands)
        self.render_hud()
        
        # 4. BACKGROUND
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.render_feed_to_texture()
        self.quad.render(self.feed_texture)

        # 5. 3D WORLD
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.texture_array.use(location=0)
        self.builder.render()

        # 6. FOREGROUND (Hands + HUD)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.render_ui_to_texture()
        self.quad.render(self.ui_texture)
                
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