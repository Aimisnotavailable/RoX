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

    # ==========================================
    # --- CORE UPDATE LOOP ---
    # ==========================================
    
    def update(self):
        self.handle_input()
        
        # Only shoot the raycast if we aren't currently dragging a line of blocks
        if not self.is_dragging:
            if self.engine.player.mode == "FPS":
                self.raycast_fps(self.engine.player.position, self.engine.player.forward)
            elif self.engine.player.mode == "RTS":
                ray_origin, ray_direction = self.get_rts_ray()
                self.raycast_rts(ray_origin, ray_direction)


    # ==========================================
    # --- SMART DRAG-TO-BUILD LOGIC ---
    # ==========================================

    def handle_input(self):
        # We only enable drag-to-build in RTS mode while trying to ADD blocks
        if self.engine.player.mode != "RTS" or self.interaction_mode != 1:
            self.is_dragging = False
            return

        mouse_pressed = pg.mouse.get_pressed()[0] # Left click
        
        # Fetch mouse movement.
        rel_x, rel_y = pg.mouse.get_rel()
        mouse_delta = glm.vec2(rel_x, rel_y)

        # 1. START DRAG: Just clicked on a block face
        if mouse_pressed and not self.is_dragging:
            if self.voxel_id and self.voxel_normal: 
                self.is_dragging = True
                self.snap_normal = glm.vec3(self.voxel_normal)
                
                # Start building exactly one block OUT from the face we clicked
                self.exact_pos = glm.vec3(self.voxel_world_pos) + self.snap_normal
                self.place_pos = glm.vec3(self.exact_pos)
                
                # Instantly place the first block
                self._place_block_at(self.place_pos)

        # 2. DURING DRAG: Holding the button and moving the mouse
        elif mouse_pressed and self.is_dragging:
            if self.snap_normal is not None:
                screen_dir = self.get_screen_axis_direction()
                pixel_movement = glm.dot(mouse_delta, screen_dir)
                
                # Move our exact float position
                self.exact_pos += self.snap_normal * (pixel_movement * self.sensitivity)

                # Snap to integer grid block
                snapped = glm.round(self.exact_pos)
                
                # If snapped position moved to a new block coordinate, place it!
                if snapped != self.place_pos:
                    self.place_pos = snapped
                    self._place_block_at(self.place_pos)

        # 3. END DRAG: Released the mouse
        elif not mouse_pressed and self.is_dragging:
            self.is_dragging = False
            self.snap_normal = None
            self.exact_pos = None

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
        if length < 0.0001:
            return glm.vec2(0)
            
        return screen_dir / length


    # ==========================================
    # --- 3D RAYCASTING ---
    # ==========================================
    
    def get_rts_ray(self, screen_pos=None):
        if screen_pos is None:
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
        
        local_origin = inv_model * glm.vec4(origin, 1.0)
        origin = glm.vec3(local_origin)
        
        local_dir = inv_model * glm.vec4(direction, 0.0)
        direction = glm.normalize(glm.vec3(local_dir))

        max_dist = 60.0 if is_rts else 8.0
        
        x1, y1, z1 = origin
        x2 = x1 + direction.x * max_dist
        y2 = y1 + direction.y * max_dist
        z2 = z1 + direction.z * max_dist

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

                if step_dir == 0:
                    self.voxel_normal.x = -dx
                elif step_dir == 1:
                    self.voxel_normal.y = -dy
                else:
                    self.voxel_normal.z = -dz
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
        if self.engine.player.mode == "RTS" and self.interaction_mode == 1:
            return 
            
        if self.interaction_mode:
            self.add_voxel()
        else:
            self.remove_voxel()

    def add_voxel(self):
        if self.voxel_id:
            self._place_block_at(self.voxel_world_pos + self.voxel_normal)

    def _place_block_at(self, pos):
        """Universal helper to safely inject a block at a coordinate and rebuild meshes."""
        result = self.get_voxel_id(pos)
        if not result[0]: # If space is empty
            _, voxel_index, voxel_local_pos, chunk = result
            if chunk is not None:
                # 100% Guaranteed Integer Index
                safe_index = int(voxel_index)
                
                chunk.voxels[safe_index] = self.new_voxel_id
                chunk.mesh.rebuild()
                if chunk.is_empty:
                    chunk.is_empty = False
                
                self._rebuild_adj_for_pos(voxel_local_pos, pos)

    def remove_voxel(self):
        if self.voxel_id:
            # 100% Guaranteed Integer Index
            safe_index = int(self.voxel_index)
            
            self.chunk.voxels[safe_index] = 0
            self.chunk.mesh.rebuild()
            self._rebuild_adj_for_pos(self.voxel_local_pos, self.voxel_world_pos)

    def switch_mode(self):
        self.interaction_mode = not self.interaction_mode

    def _rebuild_adj_for_pos(self, local_pos, world_pos):
        """Checks if a modification touched a chunk border and rebuilds neighbors."""
        # Sanitize floats out of unpacking
        lx, ly, lz = int(math.floor(local_pos[0])), int(math.floor(local_pos[1])), int(math.floor(local_pos[2]))
        wx, wy, wz = int(math.floor(world_pos[0])), int(math.floor(world_pos[1])), int(math.floor(world_pos[2]))

        if lx == 0: self.rebuild_adj_chunk((wx - 1, wy, wz))
        elif lx == CHUNK_SIZE - 1: self.rebuild_adj_chunk((wx + 1, wy, wz))

        if ly == 0: self.rebuild_adj_chunk((wx, wy - 1, wz))
        elif ly == CHUNK_SIZE - 1: self.rebuild_adj_chunk((wx, wy + 1, wz))

        if lz == 0: self.rebuild_adj_chunk((wx, wy, wz - 1))
        elif lz == CHUNK_SIZE - 1: self.rebuild_adj_chunk((wx, wy, wz + 1))

    def rebuild_adj_chunk(self, adj_voxel_pos):
        # Convert tuple back to pure int
        sanitized_pos = (int(adj_voxel_pos[0]), int(adj_voxel_pos[1]), int(adj_voxel_pos[2]))
        index = int(get_chunk_index(sanitized_pos))
        if index != -1:
            self.chunks[index].mesh.rebuild()

    def get_voxel_id(self, voxel_world_pos):
        # --- THE INTEGER FIREWALL ---
        # Safely extract and floor the incoming world position coordinates
        wx = int(math.floor(voxel_world_pos[0]))
        wy = int(math.floor(voxel_world_pos[1]))
        wz = int(math.floor(voxel_world_pos[2]))
        
        # Pure integer division for chunks
        cx, cy, cz = wx // CHUNK_SIZE, wy // CHUNK_SIZE, wz // CHUNK_SIZE

        if 0 <= cx < WORLD_W and 0 <= cy < WORLD_H and 0 <= cz < WORLD_D:
            # Calculate integer chunk index
            chunk_index = int(cx + WORLD_W * cz + WORLD_AREA * cy)
            chunk = self.chunks[chunk_index]
            
            if chunk is not None:
                # Calculate strictly integer local positions
                lx = int(wx - cx * CHUNK_SIZE)
                ly = int(wy - cy * CHUNK_SIZE)
                lz = int(wz - cz * CHUNK_SIZE)
                
                # Calculate integer voxel array index
                voxel_index = int(lx + CHUNK_SIZE * lz + CHUNK_AREA * ly)
                
                voxel_id = chunk.voxels[voxel_index]
                
                # Return guaranteed pure int values/tuples
                return voxel_id, voxel_index, (lx, ly, lz), chunk
                
        return 0, 0, (0, 0, 0), None