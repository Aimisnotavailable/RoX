import glm
import pygame

# --- NEW: Block Definitions ---
# Format: ID, Name, (Side, Top, Bottom) texture layers
# Assuming your image rows are: 0=Air, 1=Sand, 2=Grass, etc.
# And columns are: 0=Side, 1=Top, 2=Bottom
BLOCK_TYPES = [
    # ID 1: Sand (Row 1)
    {"id": 1, "name": "Sand",  "layers": (3, 4, 5)},   
    
    # ID 2: Grass (Row 2)
    {"id": 2, "name": "Grass", "layers": (6, 7, 8)},   
    
    # ID 3: Dirt (Row 3)
    {"id": 3, "name": "Dirt",  "layers": (9, 10, 11)},   
    
    # ID 4: Stone (Row 4)
    {"id": 4, "name": "Stone", "layers": (12, 13, 14)}, 
    
    # ID 5: Snow (Row 5)
    {"id": 5, "name": "Snow",  "layers": (15, 16, 17)},

    # ID 6: Leaves (Row 6)
    {"id": 6, "name": "Leaves","layers": (18, 19, 20)},

    # ID 7: Wood (Row 7)
    {"id": 7, "name": "Wood",  "layers": (21, 22, 23)},
]
class VoxelBuilder:
    def __init__(self, app):
        self.app = app
        # Dictionary now stores TYPE:  {(x,y,z): block_type_id}
        # 0=Grass, 1=Dirt, 2=Stone, 3=Wood
        self.cubes = {
            (0, 0, 0): 0, 
            (1, 0, 0): 1, 
            (-1, 0, 0): 2,
        }
        self.scale = 1.0
        self.rotation = glm.vec2(0, 0)
        
        self.hovered_block = None
        self.place_pos = None
        self.mode = 'BUILD'
        
        # Default to Grass (Index 1 in our list, which is ID 2)
        self.current_block_index = 1
        
        # ATLAS CONFIG (2x2 Grid)
        self.atlas_rows = 2
        self.atlas_cols = 2

    # ... (Keep get_model_matrix and raycast SAME as before) ...
    def get_model_matrix(self):
        model = glm.mat4(1.0)
        model = glm.rotate(model, self.rotation.y, glm.vec3(0, 1, 0))
        model = glm.rotate(model, self.rotation.x, glm.vec3(1, 0, 0))
        model = glm.scale(model, glm.vec3(self.scale))
        return model

    def raycast(self):
        # (Copy your existing raycast logic exactly as it was)
        # Just ensure checking `if check_pos in self.cubes:` works (it does, keys are checked)
        cam_pos = self.app.camera.position
        cam_dir = self.app.camera.forward
        model_mat = self.get_model_matrix()
        inv_model = glm.inverse(model_mat)
        ray_origin = inv_model * glm.vec4(cam_pos, 1.0)
        ray_origin = glm.vec3(ray_origin)
        ray_dir = inv_model * glm.vec4(cam_dir, 0.0)
        ray_dir = glm.vec3(ray_dir)
        ray_dir = glm.normalize(ray_dir)
        step_size = 0.05
        max_dist = 20.0
        current_pos = ray_origin
        last_pos = None
        for i in range(int(max_dist / step_size)):
            current_pos += ray_dir * step_size
            check_pos = (int(round(current_pos.x)), int(round(current_pos.y)), int(round(current_pos.z)))
            if check_pos in self.cubes:
                if last_pos is None: return None, None
                lx, ly, lz = int(round(last_pos.x)), int(round(last_pos.y)), int(round(last_pos.z))
                return check_pos, (lx, ly, lz)
            last_pos = glm.vec3(current_pos)
        return None, None

    def handle_click(self):
        if self.mode == 'BUILD' and self.place_pos:
            # Store the actual BLOCK ID (e.g., 2 for Grass)
            block_data = BLOCK_TYPES[self.current_block_index]
            self.cubes[self.place_pos] = block_data['id']
        elif self.mode == 'DELETE':
            if self.hovered_block:
                # .pop(key) removes it safely
                self.cubes.pop(self.hovered_block, None)

    def toggle_mode(self):
        self.mode = 'DELETE' if self.mode == 'BUILD' else 'BUILD'

    def update(self):
        keys = pygame.key.get_pressed()
        buttons = pygame.key.get
        
        # Rotation
        speed = 2.0 * self.app.delta_time * 0.001
        if keys[pygame.K_LEFT]:  self.rotation.y -= speed
        if keys[pygame.K_RIGHT]: self.rotation.y += speed
        if keys[pygame.K_UP]:    self.rotation.x -= speed
        if keys[pygame.K_DOWN]:  self.rotation.x += speed
        if keys[pygame.K_z]:     self.scale += speed * 0.5
        if keys[pygame.K_x]:     self.scale = max(0.1, self.scale - speed * 0.5)
        
        # UI Update for Window Title
        title = f"RoX | Mode: {self.mode} | Block ID: {self.current_block_index} | FPS: {int(self.app.clock.get_fps())}"
        pygame.display.set_caption(title)

        self.hovered_block, self.place_pos = self.raycast()

    def get_uv_offset(self, block_id):
        """Calculates (u, v) offset for a given block ID"""
        # ID 0 -> Row 0, Col 0
        # ID 1 -> Row 0, Col 1
        # ID 2 -> Row 1, Col 0 (In OpenGL usually bottom-up, but let's assume top-down for ease)
        
        # If we assume standard left-to-right reading:
        col = block_id % self.atlas_cols
        row = block_id // self.atlas_cols
        
        # Size of one tile (e.g. 0.5 for 2x2)
        step = 1.0 / self.atlas_cols 
        
        # Note: Depending on how your texture is flipped, 'row' might need to be inverted
        return glm.vec2(col * step, row * step)

    def render(self):
        base_model = self.get_model_matrix()
        
        # Enable blending for ghost transparency
        self.app.ctx.enable(self.app.ctx.BLEND)
        self.app.texture_array.use(location=0)
        
        # 1. RENDER WORLD BLOCKS
        self.app.prog['is_ghost'].value = 0
        self.app.prog['is_wireframe'].value = 0
        
        for pos, block_id in self.cubes.items():
            # Find block data
            data = next((b for b in BLOCK_TYPES if b['id'] == block_id), BLOCK_TYPES[0])
            
            # --- HIGHLIGHT LOGIC (Restored) ---
            if self.mode == 'DELETE' and pos == self.hovered_block:
                # Turn the BLOCK ITSELF red (no wireframe cage)
                self.app.prog['objectColor'].write(glm.vec3(1.0, 0.2, 0.2)) 
            else:
                # IMPORTANT: Reset to White for normal blocks!
                self.app.prog['objectColor'].write(glm.vec3(1.0, 1.0, 1.0))

            self.render_block(pos, base_model, data['layers'])

        # 2. RENDER GHOST BLOCK (Build Mode Only)
        if self.mode == 'BUILD' and self.place_pos:
            data = BLOCK_TYPES[self.current_block_index]
            
            self.app.prog['is_ghost'].value = 1 
            # Make the ghost slightly pulsing or just white tint
            self.app.prog['objectColor'].write(glm.vec3(1.0, 1.0, 1.0))
            
            self.render_block(self.place_pos, base_model, data['layers'])
            
            # Reset ghost mode immediately
            self.app.prog['is_ghost'].value = 0

        # 3. BUILD CURSOR (Optional Wireframe)
        # Only show the green wireframe in BUILD mode so it doesn't annoy you in Delete mode
        if self.mode == 'BUILD' and self.place_pos:
             self.app.prog['is_wireframe'].value = 1
             
             model = glm.translate(base_model, glm.vec3(self.place_pos))
             # Scale 1.05 makes it "clip" around. Scale 1.00 makes it "phase" inside.
             # Let's use 1.005 for a snug fit.
             model = glm.scale(model, glm.vec3(1.005)) 
             
             self.app.prog['m_model'].write(model)
             self.app.prog['objectColor'].write(glm.vec3(0.0, 1.0, 0.0))
             
             self.app.mesh.render_lines()
             self.app.prog['is_wireframe'].value = 0

    def render_block(self, pos, base_model, layers):
        model = glm.translate(base_model, glm.vec3(pos))
        self.app.prog['m_model'].write(model)
        
        # layers[0] is Bottom (e.g., 3)
        # layers[1] is Side   (e.g., 4)
        # layers[2] is Top    (e.g., 5)
        
        self.app.prog['u_layer_bottom'].value=layers[0] # Use .write() for ints (safer) or .value
        self.app.prog['u_layer_side'].value=layers[1]
        self.app.prog['u_layer_top'].value=layers[2]
        
        self.app.mesh.render()