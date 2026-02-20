import glm
import pygame

class VoxelBuilder:
    def __init__(self, app):
        self.app = app
        self.cubes = {(0, 0, 0)}
        self.scale = 1.0
        self.rotation = glm.vec2(0, 0)
        
        self.hovered_block = None
        self.place_pos = None
        
        # New State
        self.mode = 'BUILD' # 'BUILD' or 'DELETE'
        
        # Font for UI
        self.font = pygame.font.SysFont('arial', 24, bold=True)

    def get_model_matrix(self):
        """Reconstructs the matrix used to position the whole group."""
        model = glm.mat4(1.0)
        model = glm.rotate(model, self.rotation.y, glm.vec3(0, 1, 0))
        model = glm.rotate(model, self.rotation.x, glm.vec3(1, 0, 0))
        model = glm.scale(model, glm.vec3(self.scale))
        return model

    def raycast(self):
        """
        Casts a ray from the camera into the Local Model Space of the cubes.
        Returns: (hit_block_coord, empty_face_coord)
        """
        # 1. Get Camera Ray in World Space
        cam_pos = self.app.camera.position
        cam_dir = self.app.camera.forward
        
        # 2. Convert Ray to Model Space
        # We apply the INVERSE of the model matrix to the ray.
        # This effectively 'un-rotates' and 'un-scales' the ray so it matches the simple integer grid.
        model_mat = self.get_model_matrix()
        inv_model = glm.inverse(model_mat)
        
        # Transform Origin (Position needs w=1)
        ray_origin = inv_model * glm.vec4(cam_pos, 1.0)
        ray_origin = glm.vec3(ray_origin)
        
        # Transform Direction (Direction needs w=0 so translation is ignored)
        ray_dir = inv_model * glm.vec4(cam_dir, 0.0)
        ray_dir = glm.vec3(ray_dir)
        ray_dir = glm.normalize(ray_dir) # Normalize after scaling
        
        # 3. Step through the grid (Simple Ray Marching)
        # We step a tiny bit forward and check if we are inside a cube.
        step_size = 0.05
        max_dist = 20.0 # How far we can reach
        
        current_pos = ray_origin
        last_pos = None # Keeps track of the "empty" air block before we hit something
        
        # Maximum iterations = distance / step
        for i in range(int(max_dist / step_size)):
            current_pos += ray_dir * step_size
            
            # Round to nearest integer grid coordinate
            x = int(round(current_pos.x))
            y = int(round(current_pos.y))
            z = int(round(current_pos.z))
            check_pos = (x, y, z)
            
            if check_pos in self.cubes:
                # HIT! We found a block.
                # Return (The Block We Hit, The Empty Space Before It)
                
                # If we hit something immediately (inside a block), return None
                if last_pos is None:
                    return None, None
                    
                # Calculate integer coords for the previous step
                lx = int(round(last_pos.x))
                ly = int(round(last_pos.y))
                lz = int(round(last_pos.z))
                return check_pos, (lx, ly, lz)
            
            last_pos = glm.vec3(current_pos)

        # Ray went too far and hit nothing
        return None, None

    def handle_click(self):
        """Single function to handle action based on mode"""
        if self.mode == 'BUILD':
            if self.place_pos:
                self.cubes.add(self.place_pos)
        elif self.mode == 'DELETE':
            if self.hovered_block:
                self.cubes.discard(self.hovered_block)
    
    def toggle_mode(self):
        if self.mode == 'BUILD':
            self.mode = 'DELETE'
        else:
            self.mode = 'BUILD'
            
    def add_cube(self):
        if self.place_pos:
            self.cubes.add(self.place_pos)

    def remove_cube(self):
        if self.hovered_block:
            self.cubes.discard(self.hovered_block)

    def update(self):
        # Rotate logic
        keys = pygame.key.get_pressed()
        speed = 2.0 * self.app.delta_time * 0.001
        
        if keys[pygame.K_LEFT]:  self.rotation.y -= speed
        if keys[pygame.K_RIGHT]: self.rotation.y += speed
        if keys[pygame.K_UP]:    self.rotation.x -= speed
        if keys[pygame.K_DOWN]:  self.rotation.x += speed
        if keys[pygame.K_z]:     self.scale += speed * 0.5
        if keys[pygame.K_x]:     self.scale = max(0.1, self.scale - speed * 0.5)

        # Perform Raycast every frame
        self.hovered_block, self.place_pos = self.raycast()

        # --- UI UPDATE ---
        # Instead of rendering text (hard), we update the window title (easy & clean)
        title = f"RoX Engine | MODE: {self.mode} | Cubes: {len(self.cubes)} | FPS: {int(self.app.clock.get_fps())}"
        pygame.display.set_caption(title)

    def render(self):
        base_model = self.get_model_matrix()
        
        # 1. Render Solid Blocks
        self.app.prog['is_wireframe'].value = 0 # Tell shader to draw textures
        
        for pos in self.cubes:
            model = glm.translate(base_model, glm.vec3(pos))
            self.app.prog['m_model'].write(model)
            
            # Highlight Logic (Red tint for delete)
            if self.mode == 'DELETE' and pos == self.hovered_block:
                self.app.prog['objectColor'].write(glm.vec3(1.0, 0.3, 0.3))
            else:
                self.app.prog['objectColor'].write(glm.vec3(1.0, 1.0, 1.0))
            
            self.app.mesh.render() # Renders Solid Triangles

        # 2. Render Cursor (Real Lines)
        # Only show cursor if we have a valid place position AND we are in BUILD mode
        if self.mode == 'BUILD' and self.place_pos:
            self.app.prog['is_wireframe'].value = 1 # Tell shader to draw solid color
            
            model = glm.translate(base_model, glm.vec3(self.place_pos))
            # Optional: Scale it slightly up so it doesn't z-fight if overlapping
            model = glm.scale(model, glm.vec3(1.05)) 
            
            self.app.prog['m_model'].write(model)
            self.app.prog['objectColor'].write(glm.vec3(0.0, 1.0, 0.0)) # Bright Green
            
            self.app.mesh.render_lines() # <--- Use the new Line Renderer