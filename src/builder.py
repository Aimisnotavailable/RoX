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

        # Voxel center offset for alignment
        self.voxel_center_offset = glm.vec3(0.0, 0.0, 0.0)

        # Debug flags
        self.debug_draw = False
        self.snap_axis = (0, 0)
        self._snapped_place_pos = (0, 0, 0)
        self.stop_raycast = False
        self.sensitivity = 0.2

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
        Convert a local voxel coordinate to world-space position.
        """
        model = self.get_model_matrix()
        p4 = glm.vec4(local_pos[0], local_pos[1], local_pos[2], 1.0)
        w4 = model * p4
        return glm.vec3(w4.x, w4.y, w4.z)

    # -------------------------
    # 2D Screen -> RTS RAY
    # -------------------------
    def get_rts_ray(self, screen_pos=None):
        """
        Returns (origin, direction) in world space for the current mouse OR hand position.
        Accepts optional screen_pos (x, y) for AR hand integration.
        """
        if screen_pos is None:
            _x, _y = pygame.mouse.get_pos()
        else:
            _x, _y = screen_pos  # Use the AR hand coordinates

        width, height = self.app.WIN_SIZE

        # Normalized Device Coordinates
        x = (2.0 * _x) / width - 1.0
        y = 1.0 - (2.0 * _y) / height

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
        self.raycast_generic(origin, direction, is_rts=False)

    def raycast_rts(self, origin, direction):
        self.raycast_generic(origin, direction, is_rts=True)

    # -------------------------
    # Voxel traversal (Amanatides-Woo DDA)
    # -------------------------
    def raycast_generic(self, origin, direction, is_rts=False):
        """
        Sets self.delete_pos and self.place_pos (both in local voxel coords) or None.
        """
        direction = glm.normalize(direction)

        # Small epsilon to avoid self-intersection
        eps = 1e-6
        origin = glm.vec3(origin.x + direction.x * eps,
                          origin.y + direction.y * eps,
                          origin.z + direction.z * eps)

        x = math.floor(origin.x)
        y = math.floor(origin.y)
        z = math.floor(origin.z)

        step_x = 1 if direction.x >= 0 else -1
        step_y = 1 if direction.y >= 0 else -1
        step_z = 1 if direction.z >= 0 else -1

        t_delta_x = abs(1.0 / direction.x) if abs(direction.x) > 1e-12 else 1e30
        t_delta_y = abs(1.0 / direction.y) if abs(direction.y) > 1e-12 else 1e30
        t_delta_z = abs(1.0 / direction.z) if abs(direction.z) > 1e-12 else 1e30

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

        if (x, y, z) in self.cubes:
            self.delete_pos = (x, y, z)
            self.place_pos = None
            return

        while True:
            if (x, y, z) in self.cubes:
                self.delete_pos = (x, y, z)
                self.place_pos = last_pos
                if last_pos is None:
                    self.snap_axis = None
                else:
                    delta = (last_pos[0] - x, last_pos[1] - y, last_pos[2] - z)
                    axis = max(range(3), key=lambda i: abs(delta[i]))
                    sign = 1 if delta[axis] > 0 else -1 if delta[axis] < 0 else 0
                    self.snap_axis = (axis, sign)
                return

            nearest_t = min(t_max_x, t_max_y, t_max_z)
            if nearest_t > max_dist:
                break

            last_pos = (x, y, z)

            if t_max_x <= t_max_y and t_max_x <= t_max_z:
                x += step_x
                t_max_x += t_delta_x
            elif t_max_y <= t_max_z:
                y += step_y
                t_max_y += t_delta_y
            else:
                z += step_z
                t_max_z += t_delta_z

        self.place_pos = None
        self.delete_pos = None

    # -------------------------
    # Input handling
    # -------------------------
    def handle_click(self):
        """
        Place or delete blocks.
        """
        # --- Mouse/Raycast Snapping Logic ---
        # Only applies if we are 'holding' the click in RTS mode
        if self.app.is_rts_mode and self.stop_raycast and self.place_pos is not None:
            pos = [float(self.place_pos[0]), float(self.place_pos[1]), float(self.place_pos[2])]
            
            # Note: For AR, movement_rel might be 0, so this fine-tune snapping
            # might not work perfectly without AR delta injection, but basic placement works.
            rel_x, rel_y = self.app.camera.movement_rel

            if getattr(self, 'snap_axis', None) is not None:
                axis, sign = self.snap_axis
                if axis == 0:
                    delta = -rel_x * self.sensitivity
                elif axis == 1:
                    delta = rel_y * self.sensitivity
                else:
                    delta = -rel_y * self.sensitivity
                pos[axis] -= delta * sign

            snapped = [round(p) for p in pos]
            self.place_pos = glm.vec3(pos[0], pos[1], pos[2])
            self._snapped_place_pos = tuple(int(v) for v in snapped)

        # --- Actual Block Placement ---
        if self.mode == 'BUILD' and self.place_pos:
            if self.delete_pos is not None or (self.delete_pos is None and self.place_pos[1] == 1):
                block_data = BLOCK_TYPES[self.current_block_index]
                
                if hasattr(self, '_snapped_place_pos') and self.app.is_rts_mode:
                    place_pos_int = self._snapped_place_pos
                else:
                    place_pos_int = tuple(int(round(p)) for p in self.place_pos)

                if not place_pos_int in self.cubes:
                    self.cubes[place_pos_int] = block_data['id']

        elif self.mode == 'DELETE':
            if self.hovered_block:
                self.cubes.pop(self.hovered_block, None)

    def toggle_mode(self):
        self.mode = 'DELETE' if self.mode == 'BUILD' else 'BUILD'

    # -------------------------
    # Update loop
    # -------------------------
    def update(self, screen_pos=None):
        """
        Accepts optional screen_pos (x,y) to drive the RTS ray from AR hands.
        """
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

        title = f"RoX | Mode: {self.mode} | Block ID: {self.current_block_index} | FPS: {int(self.app.clock.get_fps())}"
        pygame.display.set_caption(title)

        # Choose Raycast Strategy based on Engine Mode
        if self.app.is_rts_mode:
            # Pass screen_pos (Hand) or None (Mouse) to the ray calculator
            ray_origin_world, ray_direction_world = self.get_rts_ray(screen_pos)
            
            local_origin, local_direction = self.world_to_model_ray(ray_origin_world, ray_direction_world)
            if not self.stop_raycast:
                self.raycast_rts(local_origin, local_direction)
        else:
            # FPS uses camera forward
            ray_origin_world = self.app.camera.position
            ray_direction_world = self.app.camera.forward
            local_origin, local_direction = self.world_to_model_ray(ray_origin_world, ray_direction_world)
            self.raycast_fps(local_origin, local_direction)

        if self.mode == 'DELETE':
            self.hovered_block = self.delete_pos
        else:
            self.hovered_block = None

    def get_uv_offset(self, block_id):
        col = block_id % self.atlas_cols
        row = block_id // self.atlas_cols
        step = 1.0 / self.atlas_cols
        return glm.vec2(col * step, row * step)

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

            ghost_pos = (self.place_pos[0] + self.voxel_center_offset.x,
                         self.place_pos[1] + self.voxel_center_offset.y,
                         self.place_pos[2] + self.voxel_center_offset.z)
            self.render_block(ghost_pos, base_model, data['layers'])

            self.app.prog['is_ghost'].value = 0

        # 3. BUILD CURSOR
        if self.mode == 'BUILD' and self.place_pos:
            self.app.prog['is_wireframe'].value = 1

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
        if not isinstance(pos, glm.vec3):
            model_pos = glm.vec3(pos[0], pos[1], pos[2])
        else:
            model_pos = pos

        model = glm.translate(base_model, model_pos)
        self.app.prog['m_model'].write(model)

        self.app.prog['u_layer_bottom'].value = layers[0]
        self.app.prog['u_layer_side'].value = layers[1]
        self.app.prog['u_layer_top'].value = layers[2]

        self.app.mesh.render()