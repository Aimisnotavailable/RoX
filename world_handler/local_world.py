from settings import *
from world_objects.chunk import Chunk
from world_handler.voxel_handler import VoxelHandler
from world_handler.world_data_handler import save_chunk, load_chunk
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from objects.world_objects import WorldObjects
from world_objects.selectable_object import SelectableObject
import threading


class LocalWorld(SelectableObject):
    def __init__(self, engine, dimensions=(0, 0, 0), obj_type = 'sphere'):
        super().__init__()
        self.engine = engine
        self.dimensions = dimensions
        self.world_area = dimensions[0] * dimensions[2]
        self.world_vol = self.world_area * dimensions[1]
        self.chunks = [None for _ in range(self.world_vol)]
        self.voxels = np.empty([self.world_vol, CHUNK_VOL], dtype='uint8')
        self.position = glm.vec3(0, 0, 0)
        self.objects : List[WorldObjects] = [WorldObjects(dimensions, obj_type)]
        self.new_world = False

        # Initial rotation (random)
        init_yaw = 0 # random.random() * math.pi * 2
        init_pitch = 0 # random.random() + 0.2
        self.rotation = glm.quat(glm.vec3(init_pitch, init_yaw, 0.0))
        self.scale = glm.vec3(1.0)

        build_meshes = True
        new_world = True
        self.new_world = new_world

        if new_world:
            get_logger_info("DEBUG", f"Time : {datetime.now()}")
            get_logger_info("DEBUG", f"WORLD BUILDING STARTED")
            self.build_chunks(build_meshes=build_meshes)
            if build_meshes:
                self.build_chunk_mesh()
            get_logger_info("DEBUG", f"WORLD BUILDING DONE")
            get_logger_info("DEBUG", f"Time : {datetime.now()}")

        self.world_swapping = False

    # ------------------------------------------------------------------
    # Compatibility properties for existing code
    # ------------------------------------------------------------------
    @property
    def world_yaw(self):
        return glm.eulerAngles(self.rotation).y

    @world_yaw.setter
    def world_yaw(self, value):
        euler = glm.eulerAngles(self.rotation)
        euler.y = value
        self.rotation = glm.quat(euler)

    @property
    def world_pitch(self):
        return glm.eulerAngles(self.rotation).x

    @world_pitch.setter
    def world_pitch(self, value):
        euler = glm.eulerAngles(self.rotation)
        euler.x = value
        self.rotation = glm.quat(euler)

    @property
    def world_scale(self):
        return self.scale.x

    @world_scale.setter
    def world_scale(self, value):
        self.scale = glm.vec3(value)

    # ------------------------------------------------------------------
    # SelectableObject implementation
    # ------------------------------------------------------------------
    def get_local_aabb(self) -> tuple[glm.vec3, glm.vec3]:
        """Local bounding box covers the entire voxel volume."""
        return (
            glm.vec3(0.0),
            glm.vec3(self.dimensions[0] * CHUNK_SIZE, self.dimensions[1]* CHUNK_SIZE, self.dimensions[2] * CHUNK_SIZE)
        )

    @property
    def m_model(self):
        """Compatibility alias for model_matrix."""
        return self.model_matrix

    # ------------------------------------------------------------------
    # Transform‑aware methods for global ↔ local conversion
    # ------------------------------------------------------------------
    def global_to_local(self, global_pos: glm.vec3 | glm.ivec3) -> glm.vec3:
        """Convert a global position to this world's local coordinates."""
        if isinstance(global_pos, glm.ivec3):
            global_pos = glm.vec3(global_pos)
        inv_model = glm.inverse(self.m_model)
        return glm.vec3(inv_model * glm.vec4(global_pos, 1.0))

    def local_to_global(self, local_pos: glm.vec3 | glm.ivec3) -> glm.vec3:
        """Convert a local position to global coordinates."""
        if isinstance(local_pos, glm.ivec3):
            local_pos = glm.vec3(local_pos)
        return glm.vec3(self.m_model * glm.vec4(local_pos, 1.0))

    def contains_global(self, global_pos: glm.vec3 | glm.ivec3) -> bool:
        """Check if a global position lies inside this world's bounds."""
        local = self.global_to_local(global_pos)
        return (0 <= local.x < self.dimensions[0] * CHUNK_SIZE and
                0 <= local.y < self.dimensions[1] * CHUNK_SIZE and
                0 <= local.z < self.dimensions[2] * CHUNK_SIZE)

    def get_voxel_global(self, global_pos: glm.vec3 | glm.ivec3) -> tuple[int, glm.ivec3, Chunk, int] | None:
        """
        Return (voxel_id, local_position_in_chunk, chunk, voxel_index)
        for the voxel at the given global position, or None if air/outside.
        """
        if isinstance(global_pos, glm.ivec3):
            global_pos = glm.vec3(global_pos)

        if not self.contains_global(global_pos):
            return None

        local = self.global_to_local(global_pos)
        wx = int(local.x)
        wy = int(local.y)
        wz = int(local.z)

        cx = wx // CHUNK_SIZE
        cy = wy // CHUNK_SIZE
        cz = wz // CHUNK_SIZE

        chunk_index = cx + self.dimensions[0] * cz + self.world_area * cy
        chunk = self.chunks[chunk_index]
        if chunk is None:
            return None

        lx = wx - cx * CHUNK_SIZE
        ly = wy - cy * CHUNK_SIZE
        lz = wz - cz * CHUNK_SIZE
        voxel_index = lx + CHUNK_SIZE * lz + CHUNK_AREA * ly
        voxel_id = chunk.voxels[voxel_index]

        if voxel_id == 0:
            return None
        return voxel_id, glm.ivec3(lx, ly, lz), chunk, voxel_index

    # ------------------------------------------------------------------
    # Original methods (unchanged except where noted)
    # ------------------------------------------------------------------
    def update(self):
        if not self.new_world:
            inv_model = glm.inverse(self.m_model)
            local_player_pos = inv_model * glm.vec4(self.engine.player.position, 1.0)
            x = int(local_player_pos.x // CHUNK_SIZE)
            y = int(local_player_pos.y // CHUNK_SIZE)
            z = int(local_player_pos.z // CHUNK_SIZE)
            self.load_visible_chunks(x, y, z)

        # Auto-rotate (optional) – using quaternion
        rot_speed = 0.001 * self.engine.delta_time
        # self.rotation = glm.quat(glm.vec3(0.0, rot_speed, 0.0)) * self.rotation

    def load_visible_chunks(self, center_x, center_y, center_z):
        WORLD_W_local = self.dimensions[0]
        WORLD_H_local = self.dimensions[1]
        WORLD_D_local = self.dimensions[2]
        WORLD_AREA_local = self.world_area
        R = RENDER_DISTANCE // 2

        x0 = max(0, center_x - R)
        x1 = min(WORLD_W_local, center_x + R)
        z0 = max(0, center_z - R)
        z1 = min(WORLD_D_local, center_z + R)
        y0 = 0
        y1 = WORLD_H_local

        mesh_to_build = []
        load_tasks = []

        with ThreadPoolExecutor(max_workers=4) as ex:
            for xi in range(x0, x1):
                for yi in range(y0, y1):
                    for zi in range(z0, z1):
                        idx = xi + WORLD_W_local * zi + WORLD_AREA_local * yi
                        if self.chunks[idx] is not None:
                            continue

                        chunk_path = CHUNK_FILE_BASE_DIR / f"chunk_{idx}.npz"
                        future = ex.submit(load_chunk, chunk_path)
                        load_tasks.append((future, idx, (xi, yi, zi)))

            for future, idx, pos in ((f, i, p) for f, i, p in load_tasks):
                try:
                    vox = future.result()
                except Exception:
                    vox = np.zeros(CHUNK_VOL, dtype=np.uint8)

                if vox.size != CHUNK_VOL:
                    if vox.size < CHUNK_VOL:
                        padded = np.zeros(CHUNK_VOL, dtype=np.uint8)
                        padded[:vox.size] = vox
                        vox = padded
                    else:
                        vox = vox[:CHUNK_VOL]

                chunk = Chunk(self, position=pos, generator=self.generator)
                self.chunks[idx] = chunk
                self.voxels[idx] = vox
                chunk.voxels = self.voxels[idx]
                chunk.is_empty = not np.any(vox)
                mesh_to_build.append(chunk)

        self.rebuild_chunk_mesh(mesh_to_build)

    def build_chunks(self, build_meshes=True):
        for i in range(self.world_vol):
            self.chunks[i] = None
        self.voxels.fill(0)

        generator = self.objects[0].generator
        bbox = generator.get_bounding_box() if hasattr(generator, 'get_bounding_box') else None
        get_logger_info('GAME', f'BBOX SIZE : {bbox}')
        positions = []
        indices = []

        if bbox is not None:
            min_cx = int(max(0, bbox[0] // CHUNK_SIZE))
            max_cx = int(min(self.dimensions[0] - 1, bbox[1] // CHUNK_SIZE))
            min_cy = int(max(0, bbox[2] // CHUNK_SIZE))
            max_cy = int(min(self.dimensions[1] - 1, bbox[3] // CHUNK_SIZE))
            min_cz = int(max(0, bbox[4] // CHUNK_SIZE))
            max_cz = int(min(self.dimensions[2] - 1, bbox[5] // CHUNK_SIZE))
            for cx in range(min_cx, max_cx + 1):
                for cy in range(min_cy, max_cy + 1):
                    for cz in range(min_cz, max_cz + 1):
                        idx = cx + self.dimensions[0] * cz + self.world_area * cy
                        positions.append((cx, cy, cz))
                        indices.append(idx)
        else:
            for x in range(self.dimensions[0]):
                for y in range(self.dimensions[1]):
                    for z in range(self.dimensions[2]):
                        idx = x + self.dimensions[0] * z + self.world_area * y
                        positions.append(glm.vec3(x, y, z))
                        indices.append(idx)

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for pos, idx in zip(positions, indices):
                future = executor.submit(self._generate_chunk_data, pos, idx)
                futures[future] = (idx, pos)

            for future in as_completed(futures):
                idx, pos = futures[future]
                try:
                    chunk, voxels = future.result()
                    if not chunk:
                        continue
                    self.chunks[idx] = chunk
                    self.voxels[idx] = voxels
                    chunk.voxels = self.voxels[idx]
                    chunk.is_empty = not np.any(voxels)
                except Exception as e:
                    get_logger_info("ERROR", f"Error generating chunk at {pos}: {e}")
        

        if build_meshes:
            for chunk in self.chunks:
                if chunk:
                    chunk.build_mesh()

    def _generate_chunk_data(self, pos, idx):
        chunk = Chunk(self, position=pos, generator=self.objects[0].generator)
        voxels = chunk.build_voxels()
        chunk.voxels = voxels

        if np.any(voxels):
            self._save_chunk(idx, voxels)
            return chunk, voxels
        else:
            return None, voxels

    def _save_chunk(self, idx, voxels):
        chunk_path = CHUNK_FILE_BASE_DIR / f"chunk_{idx}.npz"
        save_chunk(chunk_path, voxels)

    def build_chunk_mesh(self):
        for chunk in self.chunks:
            if chunk:
                chunk.build_mesh()

    def rebuild_chunk_mesh(self, chunks):
        for chunk in chunks:
            if chunk:
                chunk.build_mesh()

    def render(self):
        for chunk in self.chunks:
            if chunk:
                chunk.world_position = chunk.position + glm.ivec3(self.position)
                chunk.render()

    # Deprecated methods kept for compatibility
    def contains(self, world_pos: glm.vec3) -> bool:
        return self.contains_global(world_pos)

    def get_voxel(self, world_pos: glm.vec3) -> tuple[int, glm.ivec3, Chunk] | None:
        result = self.get_voxel_global(world_pos)
        if result is None:
            return None
        return result[0], result[1], result[2]