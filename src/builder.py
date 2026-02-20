# builder.py
import glm
import pygame
import math

# --- Block Definitions ---
# Format: ID, Name, (Bottom, Side, Top) texture layers
BLOCK_TYPES = [
    {"id": 1, "name": "Sand",  "layers": (3, 4, 5)},
    {"id": 2, "name": "Grass", "layers": (6, 7, 8)},
    {"id": 3, "name": "Dirt",  "layers": (9, 10, 11)},
    {"id": 4, "name": "Stone", "layers": (12, 13, 14)},
    {"id": 5, "name": "Snow",  "layers": (15, 16, 17)},
    {"id": 6, "name": "Leaves","layers": (18, 19, 20)},
    {"id": 7, "name": "Wood",  "layers": (21, 22, 23)},
]

class VoxelBuilder:
    def __init__(self, app):
        self.app = app

        # Local/model-space voxel dictionary: (x,y,z) -> block_id
        self.cubes = {
            (0, 0, 0): 1,
            (1, 0, 0): 2,
            (-1, 0, 0): 3,
        }

        # Visual transform of the whole builder (applied when rendering)
        self.scale = 1.0
        self.rotation = glm.vec2(0.0, 0.0)  # (x, y) rotation angles in radians

        # Interaction state (all positions are in local/model voxel coordinates)
        self.hovered_block = None   # block under cursor in DELETE mode (local coords)
        self.place_pos = None       # where a block would be placed (local coords)
        self.delete_pos = None      # which block would be deleted (local coords)
        self.mode = 'BUILD'         # 'BUILD' or 'DELETE'

        # Block selection
        self.current_block_index = 1  # index into BLOCK_TYPES

        # Texture atlas config (example)
        self.atlas_rows = 2
        self.atlas_cols = 2

        # If your cube mesh is centered at (0.5,0.5,0.5) vs at (0,0,0) corner,
        # set this to glm.vec3(0.5) to align ghost/wireframe with rendered cubes.
        # If your mesh is built from (0..1) with origin at corner, keep glm.vec3(0.0).
        self.voxel_center_offset = glm.vec3(0.0, 0.0, 0.0)

        # Debug flag: set True to render visited voxels or ray (requires extra debug draw code)
        self.debug_draw = False

    # -------------------------
    # Model matrix used for rendering
    # -------------------------
    def get_model_matrix(self):
        model = glm.mat4(1.0)
        model = glm.rotate(model, self.rotation.y, glm.vec3(0.0, 1.0, 0.0))
        model = glm.rotate(model, self.rotation.x, glm.vec3(1.0, 0.0, 0.0))
        model = glm.scale(model, glm.vec3(self.scale))
        return model

    # -------------------------
    # World <-> Model ray helpers
    # -------------------------
    def world_to_model_ray(self, origin, direction):
        """
        Transform a world-space ray into model/local space using the inverse model matrix.
        origin: glm.vec3 (world)
        direction: glm.vec3 (world)
        returns: (local_origin: glm.vec3, local_direction: glm.vec3)
        """
        model = self.get_model_matrix()
        inv_model = glm.inverse(model)

        # Transform origin as position (w = 1)
        o4 = glm.vec4(origin.x, origin.y, origin.z, 1.0)
        local_o4 = inv_model * o4
        local_origin = glm.vec3(local_o4.x, local_o4.y, local_o4.z)

        # Transform direction as vector (w = 0)
        d4 = glm.vec4(direction.x, direction.y, direction.z, 0.0)
        local_d4 = inv_model * d4
        local_direction = glm.normalize(glm.vec3(local_d4.x, local_d4.y, local_d4.z))

        return local_origin, local_direction

    def model_to_world_pos(self, local_pos):
        """
        Convert a local voxel coordinate (tuple or glm.vec3) to world-space position
        using the model matrix. Returns glm.vec3.
        """
        model = self.get_model_matrix()
        p4 = glm.vec4(local_pos[0], local_pos[1], local_pos[2], 1.0)
        w4 = model * p4
        return glm.vec3(w4.x, w4.y, w4.z)

    # -------------------------
    # Mouse -> world ray
    # -------------------------
    def get_mouse_ray(self):
        """
        Returns (origin, direction) in world space for the current mouse position.
        """
        mouse_x, mouse_y = pygame.mouse.get_pos()
        width, height = self.app.WIN_SIZE

        # Normalized Device Coordinates
        x = (2.0 * mouse_x) / width - 1.0
        y = 1.0 - (2.0 * mouse_y) / height

        clip_coords = glm.vec4(x, y, -1.0, 1.0)
        eye_coords = glm.inverse(self.app.camera.m_proj) * clip_coords
        eye_coords = glm.vec4(eye_coords.x, eye_coords.y, -1.0, 0.0)

        world_ray = glm.inverse(self.app.camera.m_view) * eye_coords
        ray_direction = glm.normalize(glm.vec3(world_ray))
        return self.app.camera.position, ray_direction

    # -------------------------
    # Raycast entry points
    # -------------------------
    def raycast_fps(self, origin, direction):
        # FPS uses camera forward ray (already transformed to model space by caller)
        self.raycast_generic(origin, direction, is_rts=False)

    def raycast_rts(self, origin, direction):
        # RTS uses mouse ray (already transformed to model space by caller)
        self.raycast_generic(origin, direction, is_rts=True)

    # -------------------------
    # Voxel traversal (Amanatides-Woo DDA)
    # Works in local/model space voxel coordinates
    # -------------------------
    def raycast_generic(self, origin, direction, is_rts=False):
        """
        origin, direction are in model/local space.
        Sets self.delete_pos and self.place_pos (both in local voxel coords) or None.
        place_pos will only be set when a valid placement is allowed:
          - adjacent to a hit block (delete_pos != None), or
          - a valid ground-plane hit (RTS mode).
        """

        # Normalize direction so t is in model units
        direction = glm.normalize(direction)

        # Small epsilon to avoid self-intersection when origin is exactly on a boundary
        eps = 1e-6
        origin = glm.vec3(origin.x + direction.x * eps,
                          origin.y + direction.y * eps,
                          origin.z + direction.z * eps)

        # Current voxel coordinates (local/model space)
        x = math.floor(origin.x)
        y = math.floor(origin.y)
        z = math.floor(origin.z)

        step_x = 1 if direction.x >= 0 else -1
        step_y = 1 if direction.y >= 0 else -1
        step_z = 1 if direction.z >= 0 else -1

        # tDelta: how far along the ray we must move for the ray to cross one voxel in that axis
        t_delta_x = abs(1.0 / direction.x) if abs(direction.x) > 1e-12 else 1e30
        t_delta_y = abs(1.0 / direction.y) if abs(direction.y) > 1e-12 else 1e30
        t_delta_z = abs(1.0 / direction.z) if abs(direction.z) > 1e-12 else 1e30

        # tMax: distance along ray to the first voxel boundary on each axis
        if step_x > 0:
            t_max_x = (x + 1.0 - origin.x) * t_delta_x
        else:
            t_max_x = (origin.x - x) * t_delta_x

        if step_y > 0:
            t_max_y = (y + 1.0 - origin.y) * t_delta_y
        else:
            t_max_y = (origin.y - y) * t_delta_y

        if step_z > 0:
            t_max_z = (z + 1.0 - origin.z) * t_delta_z
        else:
            t_max_z = (origin.z - z) * t_delta_z

        max_dist = 60.0 if is_rts else 8.0
        last_pos = None

        # If origin starts inside a block, treat as immediate hit (common FPS behavior)
        if (x, y, z) in self.cubes:
            self.delete_pos = (x, y, z)
            self.place_pos = None
            return

        # Main traversal loop
        while True:
            # RTS ground plane check: compute t where ray crosses y = 0 (model-space ground)
            if is_rts and direction.y < 0:
                t_ground = (0.0 - origin.y) / direction.y
                nearest_boundary_t = min(t_max_x, t_max_y, t_max_z)
                if 0.0 <= t_ground <= nearest_boundary_t and t_ground <= max_dist:
                    # Ground intersection occurs before any voxel boundary and within range
                    gx = x
                    gz = z
                    # If there's a block exactly at ground voxel, prefer that block
                    if (gx, 0, gz) in self.cubes:
                        self.delete_pos = (gx, 0, gz)
                        self.place_pos = last_pos
                        return
                    # Otherwise place on top of ground (y = 1)
                    self.delete_pos = None
                    self.place_pos = (gx, 1, gz)
                    return

            # If current voxel contains a block, we hit it
            if (x, y, z) in self.cubes:
                self.delete_pos = (x, y, z)
                self.place_pos = last_pos
                return

            # Stop if nearest boundary is beyond max distance
            nearest_t = min(t_max_x, t_max_y, t_max_z)
            if nearest_t > max_dist:
                break

            # Save current voxel as last empty before stepping
            last_pos = (x, y, z)

            # Step along the smallest t_max
            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                x += step_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_z:
                y += step_y
                t_max_y += t_delta_y
            else:
                z += step_z
                t_max_z += t_delta_z

        # No hit found within range
        self.place_pos = None
        self.delete_pos = None

    # -------------------------
    # Input handling
    # -------------------------
    def handle_click(self):
        """
        Place or delete blocks depending on mode.
        BUILD: only place if place_pos is valid and either adjacent to a block (delete_pos exists)
               or was set by a valid ground hit (delete_pos is None but place_pos.y == 1).
        DELETE: delete hovered_block if present.
        """
        if self.mode == 'BUILD' and self.place_pos:
            # Allow placement only if adjacent to an existing block (delete_pos != None)
            # or if it was a valid ground placement (delete_pos is None and place_pos[1] == 1)
            if self.delete_pos is not None or (self.delete_pos is None and self.place_pos[1] == 1):
                block_data = BLOCK_TYPES[self.current_block_index]
                self.cubes[self.place_pos] = block_data['id']
        elif self.mode == 'DELETE':
            if self.hovered_block:
                self.cubes.pop(self.hovered_block, None)

    def toggle_mode(self):
        self.mode = 'DELETE' if self.mode == 'BUILD' else 'BUILD'

    # -------------------------
    # Update loop: choose ray strategy and update UI state
    # -------------------------
    def update(self):
        keys = pygame.key.get_pressed()

        # Rotation and scale controls
        speed = 2.0 * self.app.delta_time * 0.001
        if keys[pygame.K_LEFT]:
            self.rotation.y -= speed
        if keys[pygame.K_RIGHT]:
            self.rotation.y += speed
        if keys[pygame.K_UP]:
            self.rotation.x -= speed
        if keys[pygame.K_DOWN]:
            self.rotation.x += speed
        if keys[pygame.K_z]:
            self.scale += speed * 0.5
        if keys[pygame.K_x]:
            self.scale = max(0.1, self.scale - speed * 0.5)

        # UI Update for Window Title
        title = f"RoX | Mode: {self.mode} | Block ID: {self.current_block_index} | FPS: {int(self.app.clock.get_fps())}"
        pygame.display.set_caption(title)

        # Choose Raycast Strategy based on Engine Mode
        if self.app.is_rts_mode:
            # Mouse ray in world space -> transform to model/local space
            ray_origin_world, ray_direction_world = self.get_mouse_ray()
            local_origin, local_direction = self.world_to_model_ray(ray_origin_world, ray_direction_world)
            self.raycast_rts(local_origin, local_direction)
        else:
            # FPS uses camera forward (world) -> transform to model/local space
            ray_origin_world = self.app.camera.position
            ray_direction_world = self.app.camera.forward
            local_origin, local_direction = self.world_to_model_ray(ray_origin_world, ray_direction_world)
            self.raycast_fps(local_origin, local_direction)

        # Update hovered_block for delete mode (convert delete_pos to hovered)
        if self.mode == 'DELETE':
            self.hovered_block = self.delete_pos
        else:
            self.hovered_block = None

    # -------------------------
    # Texture atlas helper
    # -------------------------
    def get_uv_offset(self, block_id):
        col = block_id % self.atlas_cols
        row = block_id // self.atlas_cols
        step = 1.0 / self.atlas_cols
        return glm.vec2(col * step, row * step)

    # -------------------------
    # Rendering
    # -------------------------
    def render(self):
        base_model = self.get_model_matrix()

        self.app.ctx.enable(self.app.ctx.BLEND)
        self.app.texture_array.use(location=0)

        # 1. RENDER WORLD BLOCKS
        self.app.prog['is_ghost'].value = 0
        self.app.prog['is_wireframe'].value = 0

        for pos, block_id in self.cubes.items():
            data = next((b for b in BLOCK_TYPES if b['id'] == block_id), BLOCK_TYPES[0])

            if self.mode == 'DELETE' and pos == self.hovered_block:
                self.app.prog['objectColor'].write(glm.vec3(1.0, 0.2, 0.2))
            else:
                self.app.prog['objectColor'].write(glm.vec3(1.0, 1.0, 1.0))

            self.render_block(pos, base_model, data['layers'])

        # 2. RENDER GHOST BLOCK (Build Mode Only)
        if self.mode == 'BUILD' and self.place_pos:
            data = BLOCK_TYPES[self.current_block_index]

            self.app.prog['is_ghost'].value = 1
            self.app.prog['objectColor'].write(glm.vec3(1.0, 1.0, 1.0))

            # Render ghost at place_pos; apply voxel_center_offset so it aligns with mesh origin convention
            ghost_pos = (self.place_pos[0] + self.voxel_center_offset.x,
                         self.place_pos[1] + self.voxel_center_offset.y,
                         self.place_pos[2] + self.voxel_center_offset.z)
            self.render_block(ghost_pos, base_model, data['layers'])

            self.app.prog['is_ghost'].value = 0

        # 3. BUILD CURSOR (Optional Wireframe)
        if self.mode == 'BUILD' and self.place_pos:
            self.app.prog['is_wireframe'].value = 1

            # Build a model for the wireframe that matches how we rendered the ghost
            wire_pos = glm.vec3(self.place_pos[0] + self.voxel_center_offset.x,
                                self.place_pos[1] + self.voxel_center_offset.y,
                                self.place_pos[2] + self.voxel_center_offset.z)
            model = glm.translate(base_model, wire_pos)
            model = glm.scale(model, glm.vec3(1.005))

            self.app.prog['m_model'].write(model)
            self.app.prog['objectColor'].write(glm.vec3(0.0, 1.0, 0.0))

            self.app.mesh.render_lines()
            self.app.prog['is_wireframe'].value = 0

    def render_block(self, pos, base_model, layers):
        """
        pos: either integer voxel tuple (x,y,z) or a glm.vec3 for ghost/wireframe.
        base_model: model matrix for the whole builder.
        layers: tuple of (bottom, side, top) layer indices.
        """
        # If pos is a tuple of ints, convert to glm.vec3
        if not isinstance(pos, glm.vec3):
            model_pos = glm.vec3(pos[0], pos[1], pos[2])
        else:
            model_pos = pos

        model = glm.translate(base_model, model_pos)
        self.app.prog['m_model'].write(model)

        # layers ordering: bottom, side, top
        self.app.prog['u_layer_bottom'].value = layers[0]
        self.app.prog['u_layer_side'].value = layers[1]
        self.app.prog['u_layer_top'].value = layers[2]

        self.app.mesh.render()
