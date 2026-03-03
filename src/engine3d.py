from scripts.config import *
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
        self.cap = cv2.VideoCapture("demo/demo.mp4")
        self.ar = AR(win_size)
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

        self._draw_status_panel(self.ui_surface) # Existing
        self._draw_hotbar(self.ui_surface)       # Existing
        
        # --- The "AR Stuff" ---
        self._draw_ar_status(self.ui_surface)    # NEW: Hand tracking info
        self._draw_compass(self.ui_surface)      # NEW: Orientation info

        # Upload and render to OpenGL
        texture_data = pygame.image.tostring(self.ui_surface, "RGBA", True)
        self.ui_texture.write(texture_data)

    def _draw_ar_status(self, surface):
        # Position: Top Right
        x, y = self.WIN_SIZE[0] - 270, 20
        panel_rect = pygame.Rect(x, y, 250, 100)
        pygame.draw.rect(surface, COLOR_PANEL_BG, panel_rect, border_radius=10)

        # Fetch data from arinput/ar
        ar_data = self.ar.ar_data # Or wherever you store the latest update
        
        # Hand presence "Signal" lights
        for i, label in enumerate(["LEFT", "RIGHT"]):
            is_present = ar_data["POSITION_DATA"][label] != []
            is_ghost = ar_data["FRAME_TYPE"][label] == "GHOST"
            
            # Color logic
            color = (255, 50, 50) # Red (Off)
            if is_present:
                color = (120, 120, 255) if is_ghost else (0, 255, 150) # Blue for Ghost, Green for Real
            
            # Draw status circle
            circle_y = y + 35 + (i * 30)
            pygame.draw.circle(surface, color, (x + 30, circle_y), 8)
            self._draw_text(surface, f"{label} HAND", (x + 50, circle_y - 8), 16, (200, 200, 200))
    
    def _draw_compass(self, surface):
        # Position: Bottom Right
        center_x, center_y = self.WIN_SIZE[0] - 80, self.WIN_SIZE[1] - 80
        radius = 40
        
        # Get rotation from camera
        yaw = self.camera.yaw # Assuming your camera has a yaw attribute
        
        # Draw background ring
        pygame.draw.circle(surface, (60, 60, 60, 100), (center_x, center_y), radius, width=2)
        
        # Calculate North (N) position
        # We negate yaw because camera rotation is usually inverted compared to UI space
        rad = math.radians(yaw)
        nx = center_x + math.sin(rad) * radius
        ny = center_y - math.cos(rad) * radius
        
        # Draw the "North" needle
        pygame.draw.line(surface, (255, 50, 50), (center_x, center_y), (nx, ny), 3)
        self._draw_text(surface, "N", (nx - 5, ny - 20), 14, (255, 50, 50))
    
        # Draw a little text for the angle
        self._draw_text(surface, f"{int(yaw % 360)}°", (center_x - 15, center_y + 45), 14, (200, 200, 200))
            
    def _draw_status_panel(self, surface):
        # Make the panel slightly taller to fit the FPS
        panel_rect = pygame.Rect(20, 20, 250, 130) 
        pygame.draw.rect(surface, COLOR_PANEL_BG, panel_rect, border_radius=10)
        
        # Get current FPS from pygame clock
        fps_val = int(self.clock.get_fps())
        fps_text = f"FPS: {fps_val}"
        
        # Choose a color based on performance (Optional but cool)
        fps_color = (0, 255, 150) if fps_val > 50 else (255, 200, 0)
        if fps_val < 30: fps_color = (255, 50, 50)

        mode_text = "MODE: " + self.builder.mode
        cam_text  = "CAM: " + ("RTS" if self.is_rts_mode else "FPS")
        
        # Draw the lines
        self._draw_text(surface, fps_text, (40, 35), 20, fps_color) # FPS back on top!
        self._draw_text(surface, mode_text, (40, 65), 18, (200, 200, 200))
        self._draw_text(surface, cam_text, (40, 90), 16, (150, 150, 150))

    def _draw_hotbar(self, surface):
        bar_w = 500
        bar_h = 70
        x = (self.WIN_SIZE[0] - bar_w) // 2
        y = self.WIN_SIZE[1] - bar_h - 20
        
        # Draw Hotbar Background
        pygame.draw.rect(surface, COLOR_PANEL_BG, (x, y, bar_w, bar_h), border_radius=15)
        
        # Draw Voxel Slots
        slot_size = 50
        spacing = 15
        start_x = x + (bar_w - (len(BLOCK_TYPES) * (slot_size + spacing))) // 2
        
        for i, block in enumerate(BLOCK_TYPES):
            slot_x = start_x + i * (slot_size + spacing)
            slot_y = y + (bar_h - slot_size) // 2
            
            # Highlight active slot
            is_active = (i == self.builder.current_block_index)
            border_col = COLOR_ACCENT if is_active else (100, 100, 100)
            border_width = 3 if is_active else 1
            
            pygame.draw.rect(surface, (40, 40, 40), (slot_x, slot_y, slot_size, slot_size), border_radius=5)
            pygame.draw.rect(surface, border_col, (slot_x, slot_y, slot_size, slot_size), width=border_width, border_radius=5)
            
            # Draw small block initial or ID
            self._draw_text(surface, block['name'][0], (slot_x + 18, slot_y + 12), 20, (255, 255, 255))

    def _draw_text(self, surface, text, pos, size, color):
        font = pygame.font.SysFont("Arial", size, bold=True)
        txt_surf = font.render(text, True, color)
        surface.blit(txt_surf, pos)

    def render_hands(self, surf):
        """
        Render hand landmarks stored in self.ar.ar_data onto the given pygame surface.
        Expects POSITION_DATA entries to be lists of pixel tuples or the sentinel (-1,-1)
        for invalid/missing joints. Uses self.ar.INVALID_POINT as the sentinel.
        """
        if not self.ar.ar_data.get("HAND_PRESENCE"):
            get_logger_info('AR', '[AR] FAILED TO DETECT HANDS', True)
            return

        INVALID = getattr(self.ar, "INVALID_POINT", (-1, -1))

        # Try to obtain HAND_CONNECTIONS in a safe way
        try:
            HAND_CONNECTIONS = self.ar.mp_hands.HAND_CONNECTIONS
        except Exception:
            try:
                from mediapipe.python.solutions.hands import HAND_CONNECTIONS
            except Exception:
                HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,9),(9,13),(13,17),(17,0)]

        for label in ("LEFT", "RIGHT"):
            pts = self.ar.ar_data.get('POSITION_DATA', {}).get(label, [])
            if not pts:
                continue

            # draw connections only if both endpoints are valid
            # compute max index safely
            try:
                max_idx = max(max(c) for c in HAND_CONNECTIONS)
            except Exception:
                max_idx = -1

            if len(pts) > max_idx:
                for a_idx, b_idx in HAND_CONNECTIONS:
                    # ensure indices exist
                    if a_idx >= len(pts) or b_idx >= len(pts):
                        continue
                    pa = pts[a_idx]
                    pb = pts[b_idx]
                    # both endpoints must be valid (not sentinel and not None)
                    if pa and pb and pa != INVALID and pb != INVALID:
                        try:
                            pygame.draw.line(surf, (0, 0, 255), (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), 1)
                        except Exception:
                            # defensive: skip any drawing errors
                            continue

            # draw landmark points (preserve index positions; skip invalids)
            for p in pts:
                if not p or p == INVALID:
                    continue
                try:
                    cx = int(round(p[0])); cy = int(round(p[1]))
                    pygame.draw.circle(surf, (255, 255, 255), (cx, cy), 2)
                except Exception:
                    # if p is malformed, skip it
                    continue


                color = (200, 80, 80) if label == "LEFT" else (80, 80, 200)
                p = pts[WRIST_IDX] if len(pts) > WRIST_IDX else max_idx
                pygame.draw.circle(surf, color, p, 10, 2)
                font = pygame.font.SysFont("Arial", 14)
                txt = font.render(f"{label}", True, color)
                surf.blit(txt, (max(0, p[0]-20), max(0, p[1]-30)))

    def update(self):
        # 1. Update AR State
        ret, frame = self.cap.read()
        if ret:
            self.ar.update(frame)
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
    
    def render_feed_to_texture(self, fit_to_screen=False):
        if self.ar.image:
            image = self.ar.image
            if fit_to_screen:
                image = pygame.transform.scale(self.ar.image, self.WIN_SIZE)

            self.feed_surface.blit(image, (0, 0) if fit_to_screen else (self.WIN_SIZE[0] - image.get_width(), 0))
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
        self.render_hands(self.ui_surface)
        
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