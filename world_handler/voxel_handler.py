# voxel_handler.py
import pygame as pg
import glm
import math
from settings import *
from scripts.logger import get_logger_info

class VoxelHandler:
    def __init__(self, world_container):
        self.world_container = world_container
        self.engine = world_container.engine

        # --- Ray Casting Result State ---
        self.local_world = None
        self.chunk = None
        self.voxel_id: int | None = None
        self.voxel_index: int | None = None
        self.voxel_local_pos: glm.ivec3 | None = None
        self.voxel_world_pos: glm.ivec3 | None = None
        self.voxel_normal: glm.ivec3 | None = None

        self.interaction_mode = 0  # 0: remove, 1: add, 2: grab
        self.new_voxel_id = WOOD

        # --- Drag state ---
        self.is_dragging = False
        self.snap_normal: glm.vec3 | None = None
        self.exact_pos: glm.vec3 | None = None
        self.place_pos: glm.ivec3 | None = None
        self.sensitivity = 0.05
        self.last_ar_mouse_pos: glm.vec2 | None = None

        # --- DEPTH INTEGRATION: brush multiplier ---
        self.brush_mult = 1.0
        self.drag_start_depth: float | None = None

    def update(self):
        self.handle_input()

        ar_controller = getattr(self.engine, 'ar_controller', None)
        ar_mouse_pos = getattr(ar_controller, 'ar_mouse_pos', None)

        if not self.is_dragging:
            if self.engine.player.mode == "FPS":
                if self.interaction_mode < 2:
                    self.raycast_fps(self.engine.player.position, self.engine.player.forward)
                elif self.interaction_mode == 2 and not ar_controller.is_grabbing:
                    self.ray_cast_from_hands()
                else:
                    self.ray_cast_from_hands()
            elif self.engine.player.mode == "RTS":
                ray_origin, ray_direction = self.get_rts_ray(screen_pos=ar_mouse_pos)
                self.raycast_rts(ray_origin, ray_direction)

    def handle_input(self):
        """Process mouse / AR input for dragging and interaction."""
        if self.engine.player.mode != "RTS":
            self.is_dragging = False
            return

        ar_controller = getattr(self.engine, 'ar_controller', None)
        ar_mouse_pos = getattr(ar_controller, 'ar_mouse_pos', None)
        ar_right_click = getattr(ar_controller, 'ar_right_click', False)

        if ar_mouse_pos is not None:
            current_mouse_pos = glm.vec2(ar_mouse_pos[0], ar_mouse_pos[1])
            mouse_pressed = ar_right_click
            if self.last_ar_mouse_pos is None:
                self.last_ar_mouse_pos = current_mouse_pos
            mouse_delta = current_mouse_pos - self.last_ar_mouse_pos
            self.last_ar_mouse_pos = current_mouse_pos
        else:
            self.last_ar_mouse_pos = None
            mouse_pressed = pg.mouse.get_pressed()[0]
            rel_x, rel_y = pg.mouse.get_rel()
            mouse_delta = glm.vec2(rel_x, rel_y)

        # --- Depth delta for brush multiplier ---
        current_depth = None
        if ar_controller and ar_controller.smooth_right_pos is not None:
            current_depth = ar_controller.smooth_right_pos.z

        # 1. START DRAG
        if mouse_pressed and not self.is_dragging:
            if self.voxel_id and self.voxel_normal:
                self.is_dragging = True
                if self.interaction_mode == 0:  # REMOVE
                    self.snap_normal = -glm.vec3(self.voxel_normal)
                    self.exact_pos = glm.vec3(self.voxel_world_pos)
                elif self.interaction_mode == 1:  # ADD
                    self.snap_normal = glm.vec3(self.voxel_normal)
                    self.exact_pos = glm.vec3(self.voxel_world_pos) + self.snap_normal
                else:  # GRAB
                    self.snap_normal = -glm.vec3(self.voxel_normal)
                    self.exact_pos = glm.vec3(self.voxel_world_pos)

                self.place_pos = glm.ivec3(glm.round(self.exact_pos))
                self._apply_drag_modification(self.place_pos)

                if current_depth is not None:
                    self.drag_start_depth = current_depth
                    self.brush_mult = 1.0
                else:
                    self.drag_start_depth = None
                get_logger_info('DEBUG', f'Drag START at {self.place_pos}')

        # 2. DURING DRAG
        elif mouse_pressed and self.is_dragging:
            if self.snap_normal is not None:
                screen_dir = self.get_screen_axis_direction()

                speed_factor = 1.0
                # Optional depth scaling for speed
                # if current_depth is not None:
                #     speed_factor = max(Z_DRAG_SPEED_MIN, min(Z_DRAG_SPEED_MAX, 1.0 + (current_depth - 0.5) * 1.5))

                if self.drag_start_depth is not None and current_depth is not None:
                    delta_z = current_depth - self.drag_start_depth
                    self.brush_mult = 1.0 + delta_z * RIGHT_BRUSH_SENSITIVITY
                    self.brush_mult = max(BRUSH_MULT_MIN, min(BRUSH_MULT_MAX, self.brush_mult))

                pixel_movement = glm.dot(mouse_delta, screen_dir) * self.sensitivity * speed_factor * self.brush_mult
                self.exact_pos += self.snap_normal * pixel_movement
                snapped = glm.ivec3(glm.round(self.exact_pos))

                if snapped != self.place_pos:
                    self.place_pos = snapped
                    self._apply_drag_modification(self.place_pos)

        # 3. END DRAG
        elif not mouse_pressed and self.is_dragging:
            get_logger_info('DEBUG', 'Ended Drag Action.')
            self.is_dragging = False
            self.snap_normal = None
            self.exact_pos = None
            self.drag_start_depth = None
            self.brush_mult = 1.0

    def _apply_drag_modification(self, pos: glm.ivec3):
        """Place or remove a block at the given world position during dragging."""
        if self.interaction_mode == 1:  # ADD
            self._place_block_at(pos)
        else:  # REMOVE
            self._remove_block_at(pos)

    def get_screen_axis_direction(self) -> glm.vec2:
        """Compute the 2D screen-space direction of the drag normal."""
        if self.snap_normal is None or self.exact_pos is None:
            return glm.vec2(0)

        # Use integer world position for world lookup
        exact_ivec = glm.ivec3(glm.round(self.exact_pos))
        lw = self.world_container.get_local_world_at(exact_ivec)
        m_model = lw.m_model if lw else glm.mat4(1.0)

        # Continue using self.exact_pos (float) for precise screen projection
        p1 = m_model * glm.vec4(self.exact_pos, 1.0)
        p2 = m_model * glm.vec4(self.exact_pos + glm.vec3(self.snap_normal), 1.0)

        m_vp = self.engine.player.m_proj * self.engine.player.m_view
        c1 = m_vp * p1
        c2 = m_vp * p2

        if c1.w == 0 or c2.w == 0:
            return glm.vec2(0)

        n1 = glm.vec3(c1) / c1.w
        n2 = glm.vec3(c2) / c2.w

        screen_dir = glm.vec2(n2.x - n1.x, -(n2.y - n1.y))
        length = glm.length(screen_dir)
        return screen_dir / length if length > 0.0001 else glm.vec2(0)

    # ==========================================
    # --- 3D RAYCASTING ---
    # ==========================================

    def get_rts_ray(self, screen_pos=None):
        """Convert screen coordinates to a world-space ray."""
        if screen_pos is None:
            if getattr(self.engine, 'ar_mouse_pos', None) is not None:
                _x, _y = self.engine.ar_mouse_pos
            else:
                _x, _y = pg.mouse.get_pos()
        else:
            _x, _y = screen_pos

        width, height = WIN_RES
        x = (2.0 * _x) / width - 1.0
        y = 1.0 - (2.0 * _y) / height
        clip_coords = glm.vec4(x, y, -1.0, 1.0)
        eye_coords = glm.inverse(self.engine.player.m_proj) * clip_coords

        eye_coords = glm.vec4(eye_coords.x, eye_coords.y, -1.0, 0.0)
        world_ray = glm.inverse(self.engine.player.m_view) * eye_coords
        ray_direction = glm.normalize(glm.vec3(world_ray))

        return self.engine.player.position, ray_direction

    def ray_cast_from_hands(self):
        """Raycast from camera through right hand index tip (FPS mode)."""
        ar_controller = getattr(self.engine, 'ar_controller', None)
        if ar_controller is None:
            return
        landmarks = ar_controller.smooth_right_landmarks
        if not landmarks or len(landmarks) < 21:
            return
        tip_norm = landmarks[8]  # index tip
        screen_x = tip_norm.x * WIN_RES[0]
        screen_y = tip_norm.y * WIN_RES[1]

        origin, direction = self.get_rts_ray(screen_pos=(screen_x, screen_y))
        self.raycast_generic(origin, direction, is_rts=False)

    def raycast_fps(self, origin, direction):
        self.raycast_generic(origin, direction, is_rts=False)

    def raycast_rts(self, origin, direction):
        self.raycast_generic(origin, direction, is_rts=True)

    def raycast_generic(self, origin: glm.vec3, direction: glm.vec3, is_rts: bool):
        """Perform DDA raycasting in world space, querying WorldContainer."""
        max_dist = MAX_RAY_DIST
        x1, y1, z1 = origin
        x2, y2, z2 = origin + direction * max_dist

        current_voxel_pos = glm.ivec3(glm.floor(origin))
        step_dir = -1

        dx = glm.sign(x2 - x1)
        delta_x = min(dx / (x2 - x1), 1e7) if dx != 0 else 1e7
        max_x = delta_x * (1.0 - glm.fract(x1)) if dx > 0 else delta_x * glm.fract(x1)

        dy = glm.sign(y2 - y1)
        delta_y = min(dy / (y2 - y1), 1e7) if dy != 0 else 1e7
        max_y = delta_y * (1.0 - glm.fract(y1)) if dy > 0 else delta_y * glm.fract(y1)

        dz = glm.sign(z2 - z1)
        delta_z = min(dz / (z2 - z1), 1e7) if dz != 0 else 1e7
        max_z = delta_z * (1.0 - glm.fract(z1)) if dz > 0 else delta_z * glm.fract(z1)

        # Reset results
        self.voxel_id = None
        self.voxel_normal = None
        self.local_world = None
        self.chunk = None
        self.voxel_index = None
        self.voxel_local_pos = None
        self.voxel_world_pos = None

        while not (max_x > 1.0 and max_y > 1.0 and max_z > 1.0):
            result = self.world_container.get_voxel(current_voxel_pos)
            if result is not None:
                voxel_id, local_pos, chunk, voxel_index = result
                self.voxel_id = voxel_id
                self.voxel_local_pos = local_pos
                self.chunk = chunk
                self.voxel_world_pos = current_voxel_pos
                self.voxel_index = voxel_index
                self.local_world = self.world_container.get_local_world_at(current_voxel_pos)

                get_logger_info('GAME', f"{' | '.join(map(str, result))}")
                if step_dir == 0:
                    self.voxel_normal = glm.ivec3(-dx, 0, 0)
                elif step_dir == 1:
                    self.voxel_normal = glm.ivec3(0, -dy, 0)
                else:
                    self.voxel_normal = glm.ivec3(0, 0, -dz)
                return True

            if max_x < max_y:
                if max_x < max_z:
                    current_voxel_pos.x += dx
                    max_x += delta_x
                    step_dir = 0
                else:
                    current_voxel_pos.z += dz
                    max_z += delta_z
                    step_dir = 2
            else:
                if max_y < max_z:
                    current_voxel_pos.y += dy
                    max_y += delta_y
                    step_dir = 1
                else:
                    current_voxel_pos.z += dz
                    max_z += delta_z
                    step_dir = 2

        return False

    # ==========================================
    # --- VOXEL MODIFICATION & MESH UPDATING ---
    # ==========================================

    def set_voxel(self):
        """Triggered by single clicks (quick pinch)."""
        if self.is_dragging:
            get_logger_info('DEBUG', 'set_voxel ignored: dragging')
            return

        if self.voxel_id:
            if self.interaction_mode == 1:
                self.add_voxel()
            elif self.interaction_mode == 0:
                self.remove_voxel()
        else:
            get_logger_info('DEBUG', 'set_voxel: no voxel targeted')

    def add_voxel(self):
        if self.voxel_id:
            target_pos = self.voxel_world_pos + self.voxel_normal
            self._place_block_at(target_pos)

    def remove_voxel(self):
        if self.voxel_id:
            self._remove_block_at(self.voxel_world_pos)

    def _place_block_at(self, pos: glm.ivec3):
        """Place a block at the given world position if empty."""
        result = self.world_container.get_voxel(pos)
        if result is not None:
            get_logger_info('DEBUG', f'Cannot place block at {pos}: space occupied')
            return

        lw = self.world_container.get_local_world_at(pos)
        if lw is None:
            return

        local_pos = pos - lw.position
        wx, wy, wz = int(local_pos.x), int(local_pos.y), int(local_pos.z)
        cx = wx // CHUNK_SIZE
        cy = wy // CHUNK_SIZE
        cz = wz // CHUNK_SIZE

        chunk_index = cx + WORLD_W * cz + WORLD_AREA * cy
        chunk = lw.chunks[chunk_index]
        if chunk is None:
            return

        lx = wx - cx * CHUNK_SIZE
        ly = wy - cy * CHUNK_SIZE
        lz = wz - cz * CHUNK_SIZE
        voxel_index = lx + CHUNK_SIZE * lz + CHUNK_AREA * ly

        chunk.voxels[voxel_index] = self.new_voxel_id
        chunk.mesh.rebuild()
        if chunk.is_empty:
            chunk.is_empty = False
        self._rebuild_adjacent_chunks(lw, glm.ivec3(lx, ly, lz), pos)
        get_logger_info('DEBUG', f'Placed block {self.new_voxel_id} at {pos}')

    def _remove_block_at(self, pos: glm.ivec3):
        """Remove the block at the given world position if present."""
        result = self.world_container.get_voxel(pos)
        if result is None:
            return
        voxel_id, local_pos, chunk, voxel_index = result
        if voxel_id == 0:
            return

        chunk.voxels[voxel_index] = 0
        chunk.mesh.rebuild()

        lw = self.world_container.get_local_world_at(pos)
        if lw is not None:
            self._rebuild_adjacent_chunks(lw, local_pos, pos)
        get_logger_info('DEBUG', f'Removed block at {pos}')

    def _rebuild_adjacent_chunks(self, lw, local_pos: glm.ivec3, world_pos: glm.ivec3):
        """Rebuild neighboring chunks if the modified block is on a chunk border."""
        lx, ly, lz = local_pos.x, local_pos.y, local_pos.z
        wx, wy, wz = int(world_pos.x), int(world_pos.y), int(world_pos.z)

        # Determine chunk coordinates of the modified block within the local world
        local_world_pos = world_pos - lw.position
        cx = int(local_world_pos.x // CHUNK_SIZE)
        cy = int(local_world_pos.y // CHUNK_SIZE)
        cz = int(local_world_pos.z // CHUNK_SIZE)

        neighbors = []
        if lx == 0:
            neighbors.append((cx - 1, cy, cz))
        elif lx == CHUNK_SIZE - 1:
            neighbors.append((cx + 1, cy, cz))
        if ly == 0:
            neighbors.append((cx, cy - 1, cz))
        elif ly == CHUNK_SIZE - 1:
            neighbors.append((cx, cy + 1, cz))
        if lz == 0:
            neighbors.append((cx, cy, cz - 1))
        elif lz == CHUNK_SIZE - 1:
            neighbors.append((cx, cy, cz + 1))

        for nx, ny, nz in neighbors:
            if 0 <= nx < WORLD_W and 0 <= ny < WORLD_H and 0 <= nz < WORLD_D:
                idx = nx + WORLD_W * nz + WORLD_AREA * ny
                chunk = lw.chunks[idx]
                if chunk is not None:
                    chunk.mesh.rebuild()

    def switch_mode(self):
        self.interaction_mode = (self.interaction_mode + 1) % len(INTERACTION_MODE)
        get_logger_info('GAME', f'Interaction Mode switched to: {INTERACTION_MODE[self.interaction_mode]}')