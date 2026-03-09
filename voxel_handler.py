import pygame as pg
import glm
import math
from settings import *
from meshes.chunk_mesh_builder import get_chunk_index

class VoxelHandler:
    def __init__(self, world):
        self.world = world
        self.engine = world.engine
        self.chunks = world.chunks

        # --- Ray Casting Result State ---
        self.chunk = None
        self.voxel_id = None
        self.voxel_index = None
        self.voxel_local_pos = None
        self.voxel_world_pos = None
        self.voxel_normal = None

        self.interaction_mode = 0  # 0: remove voxel   1: add voxel
        self.new_voxel_id = WOOD   # Assuming WOOD is defined in settings.py
        
        # --- Smart Drag-to-Build State ---
        self.is_dragging = False
        self.snap_normal = None    # The 3D axis we lock onto (e.g., (1, 0, 0))
        self.exact_pos = None      # The float position during the drag
        self.place_pos = None      # The snapped integer position
        self.sensitivity = 0.05    # Adjust to make mouse dragging faster/slower
        self.last_ar_mouse_pos = None

    # ==========================================
    # --- CORE UPDATE LOOP ---
    # ==========================================
    
    def update(self):
        self.handle_input()
        
        # Safely fetch active AR cursor state
        ar_controller = getattr(self.engine, 'ar_controller', None)
        ar_mouse_pos = getattr(ar_controller, 'ar_mouse_pos', None)
        
        if not self.is_dragging:
            if self.engine.player.mode == "FPS":
                self.raycast_fps(self.engine.player.position, self.engine.player.forward)
            elif self.engine.player.mode == "RTS":
                # get_rts_ray accepts screen_pos override naturally
                ray_origin, ray_direction = self.get_rts_ray(screen_pos=ar_mouse_pos)
                self.raycast_rts(ray_origin, ray_direction)


    # ==========================================
    # --- INPUT HANDLING ---
    # ==========================================

    def handle_input(self):
        # FIX: Only restrict by Player Mode, allowing both Add (1) and Remove (0) modes
        if self.engine.player.mode != "RTS":
            self.is_dragging = False
            return

        # --- SEAMLESS INPUT FALLBACK SYSTEM ---
        ar_controller = getattr(self.engine, 'ar_controller', None)
        ar_mouse_pos = getattr(ar_controller, 'ar_mouse_pos', None)
        ar_right_click = getattr(ar_controller, 'ar_right_click', False)

        if ar_mouse_pos is not None:
            current_mouse_pos = glm.vec2(ar_mouse_pos[0] * WIN_RES[0], ar_mouse_pos[1] * WIN_RES[1])
            mouse_pressed = ar_right_click
            
            if self.last_ar_mouse_pos is None:
                self.last_ar_mouse_pos = current_mouse_pos
                
            # --- DAMPEN AR EXTRUSION SPEED ---
            # Multiply delta by 0.15 to prevent hyperspeed movement
            mouse_delta = (current_mouse_pos - self.last_ar_mouse_pos) * 0.15 
            self.last_ar_mouse_pos = current_mouse_pos
        else:
            # PHYSICAL MOUSE FALLBACK
            self.last_ar_mouse_pos = None 
            mouse_pressed = pg.mouse.get_pressed()[0]
            rel_x, rel_y = pg.mouse.get_rel()
            mouse_delta = glm.vec2(rel_x, rel_y)

        # 1. START DRAG
        if mouse_pressed and not self.is_dragging:
            if self.voxel_id and self.voxel_normal: 
                self.is_dragging = True
                
                # BRANCH: Start position and direction based on mode
                if self.interaction_mode == 1: # ADD
                    self.snap_normal = glm.vec3(self.voxel_normal)
                    self.exact_pos = glm.vec3(self.voxel_world_pos) + self.snap_normal
                else: # REMOVE
                    # Inverse the normal to drag "into" the geometry
                    self.snap_normal = -glm.vec3(self.voxel_normal) 
                    self.exact_pos = glm.vec3(self.voxel_world_pos)
                    
                self.place_pos = glm.vec3(self.exact_pos)
                self._apply_drag_modification(self.place_pos)

        # 2. DURING DRAG
        elif mouse_pressed and self.is_dragging:
            if self.snap_normal is not None:
                screen_dir = self.get_screen_axis_direction()
                pixel_movement = glm.dot(mouse_delta, screen_dir)
                
                self.exact_pos += self.snap_normal * (pixel_movement * self.sensitivity)
                snapped = glm.round(self.exact_pos)
                
                if snapped != self.place_pos:
                    self.place_pos = snapped
                    self._apply_drag_modification(self.place_pos)

        # 3. END DRAG
        elif not mouse_pressed and self.is_dragging:
            get_logger_info('DEBUG', 'Ended Drag Action.')
            self.is_dragging = False
            self.snap_normal = None
            self.exact_pos = None

    def _apply_drag_modification(self, pos):
        """Internal helper to route drag updates to the correct voxel method."""
        if self.interaction_mode == 1:
            # Call your existing single-block placement helper
            self._place_block_at(pos) 
        else:
            # Manual removal logic for specific coordinates
            voxel_id, voxel_index, local_pos, chunk = self.get_voxel_id(pos)
            if chunk is not None and voxel_id != 0:
                chunk.voxels[int(voxel_index)] = 0
                chunk.mesh.rebuild()
                self._rebuild_adj_for_pos(local_pos, pos)

    def get_screen_axis_direction(self):
        if self.snap_normal is None or self.exact_pos is None:
            return glm.vec2(0)

        m_model = self.world.m_model 
        p1 = m_model * glm.vec4(self.exact_pos, 1.0)
        p2 = m_model * glm.vec4(self.exact_pos + glm.vec3(self.snap_normal), 1.0)
        
        m_vp = self.engine.player.m_proj * self.engine.player.m_view
        c1 = m_vp * p1
        c2 = m_vp * p2
        
        if c1.w == 0 or c2.w == 0: return glm.vec2(0)
        
        n1 = glm.vec3(c1) / c1.w
        n2 = glm.vec3(c2) / c2.w
        
        screen_dir = glm.vec2(n2.x - n1.x, -(n2.y - n1.y)) 
        length = glm.length(screen_dir)
        return screen_dir / length if length > 0.0001 else glm.vec2(0)


    # ==========================================
    # --- 3D RAYCASTING ---
    # ==========================================
    
    def get_rts_ray(self, screen_pos=None):
        if screen_pos is None:
            # Prioritize AR Virtual Cursor, fallback to physical mouse
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

    def raycast_fps(self, origin, direction):
        self.raycast_generic(origin, direction, is_rts=False)

    def raycast_rts(self, origin, direction):
        self.raycast_generic(origin, direction, is_rts=True)

    def raycast_generic(self, origin, direction, is_rts=False):
        inv_model = glm.inverse(self.world.m_model)
        local_origin = glm.vec3(inv_model * glm.vec4(origin, 1.0))
        local_dir = glm.normalize(glm.vec3(inv_model * glm.vec4(direction, 0.0)))

        max_dist = 60.0 if is_rts else 8.0
        x1, y1, z1 = local_origin
        x2, y2, z2 = local_origin + local_dir * max_dist

        current_voxel_pos = glm.ivec3(x1, y1, z1)
        self.voxel_id = 0
        self.voxel_normal = glm.ivec3(0)
        step_dir = -1

        dx = glm.sign(x2 - x1)
        delta_x = min(dx / (x2 - x1), 10000000.0) if dx != 0 else 10000000.0
        max_x = delta_x * (1.0 - glm.fract(x1)) if dx > 0 else delta_x * glm.fract(x1)

        dy = glm.sign(y2 - y1)
        delta_y = min(dy / (y2 - y1), 10000000.0) if dy != 0 else 10000000.0
        max_y = delta_y * (1.0 - glm.fract(y1)) if dy > 0 else delta_y * glm.fract(y1)

        dz = glm.sign(z2 - z1)
        delta_z = min(dz / (z2 - z1), 10000000.0) if dz != 0 else 10000000.0
        max_z = delta_z * (1.0 - glm.fract(z1)) if dz > 0 else delta_z * glm.fract(z1)

        while not (max_x > 1.0 and max_y > 1.0 and max_z > 1.0):
            result = self.get_voxel_id(voxel_world_pos=current_voxel_pos)
            if result[0]:
                self.voxel_id, self.voxel_index, self.voxel_local_pos, self.chunk = result
                self.voxel_world_pos = current_voxel_pos

                if step_dir == 0: self.voxel_normal.x = -dx
                elif step_dir == 1: self.voxel_normal.y = -dy
                else: self.voxel_normal.z = -dz
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
        """Triggered by single clicks."""
        if self.is_dragging: return
            
        if self.interaction_mode:
            self.add_voxel()
        else:
            self.remove_voxel()

    def add_voxel(self):
        if self.voxel_id:
            self._place_block_at(self.voxel_world_pos + self.voxel_normal)

    def remove_voxel(self):
        if self.voxel_id:
            self._remove_block_at(self.voxel_world_pos)

    def _place_block_at(self, pos):
        """Universal helper to safely inject a block at a coordinate and rebuild meshes."""
        result = self.get_voxel_id(pos)
        if not result[0]: # If space is empty
            _, voxel_index, voxel_local_pos, chunk = result
            if chunk is not None:
                chunk.voxels[int(voxel_index)] = self.new_voxel_id
                chunk.mesh.rebuild()
                if chunk.is_empty: chunk.is_empty = False
                self._rebuild_adj_for_pos(voxel_local_pos, pos)

    def remove_voxel(self):
        if self.voxel_id:
            self.chunk.voxels[int(self.voxel_index)] = 0
            self.chunk.mesh.rebuild()
            self._rebuild_adj_for_pos(self.voxel_local_pos, self.voxel_world_pos)

    def switch_mode(self):
        self.interaction_mode = not self.interaction_mode
        mode_str = "ADD" if self.interaction_mode else "REMOVE"
        get_logger_info('GAME', f'Interaction Mode switched to: {mode_str}')

    def _rebuild_adj_for_pos(self, local_pos, world_pos):
        lx, ly, lz = int(math.floor(local_pos[0])), int(math.floor(local_pos[1])), int(math.floor(local_pos[2]))
        wx, wy, wz = int(math.floor(world_pos[0])), int(math.floor(world_pos[1])), int(math.floor(world_pos[2]))

        if lx == 0: self.rebuild_adj_chunk((wx - 1, wy, wz))
        elif lx == CHUNK_SIZE - 1: self.rebuild_adj_chunk((wx + 1, wy, wz))
        if ly == 0: self.rebuild_adj_chunk((wx, wy - 1, wz))
        elif ly == CHUNK_SIZE - 1: self.rebuild_adj_chunk((wx, wy + 1, wz))
        if lz == 0: self.rebuild_adj_chunk((wx, wy, wz - 1))
        elif lz == CHUNK_SIZE - 1: self.rebuild_adj_chunk((wx, wy, wz + 1))

    def rebuild_adj_chunk(self, adj_voxel_pos):
        sanitized_pos = (int(adj_voxel_pos[0]), int(adj_voxel_pos[1]), int(adj_voxel_pos[2]))
        index = int(get_chunk_index(sanitized_pos))
        if index != -1:
            self.chunks[index].mesh.rebuild()

    def get_voxel_id(self, voxel_world_pos):
        wx, wy, wz = int(math.floor(voxel_world_pos[0])), int(math.floor(voxel_world_pos[1])), int(math.floor(voxel_world_pos[2]))
        cx, cy, cz = wx // CHUNK_SIZE, wy // CHUNK_SIZE, wz // CHUNK_SIZE

        if 0 <= cx < WORLD_W and 0 <= cy < WORLD_H and 0 <= cz < WORLD_D:
            chunk_index = int(cx + WORLD_W * cz + WORLD_AREA * cy)
            chunk = self.chunks[chunk_index]
            if chunk is not None:
                lx, ly, lz = int(wx - cx * CHUNK_SIZE), int(wy - cy * CHUNK_SIZE), int(wz - cz * CHUNK_SIZE)
                voxel_index = int(lx + CHUNK_SIZE * lz + CHUNK_AREA * ly)
                return chunk.voxels[voxel_index], voxel_index, (lx, ly, lz), chunk
        return 0, 0, (0, 0, 0), None