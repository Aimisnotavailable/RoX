from settings import *
from world_objects.chunk import Chunk
from world_handler.voxel_handler import VoxelHandler
from world_handler.world_data_handler import save_chunk, load_chunk   # new imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from world_handler.world_generators import TerrainWorldGenerator, FunctionWorldGenerator

class World:
    def __init__(self, engine, new_world=False, generator_type='terrain'):
        self.engine = engine
        self.chunks = [None for _ in range(WORLD_VOL)]
        self.voxels = np.empty([WORLD_VOL, CHUNK_VOL], dtype='uint8')
        self.new_world = new_world

        # --- Create the appropriate generator ---
        if generator_type == 'terrain':
            self.generator = TerrainWorldGenerator()
        elif generator_type == 'sphere':
            center = (WORLD_W * CHUNK_SIZE // 2, WORLD_H * CHUNK_SIZE // 2, WORLD_D * CHUNK_SIZE // 2)
            radius = 100
            def sphere_func(x, y, z, center_x, center_y, center_z, radius):
                dx = x - center_x
                dy = y - center_y
                dz = z - center_z
                return dx*dx + dy*dy + dz*dz <= radius*radius

            self.generator = FunctionWorldGenerator(sphere_func,
                                    voxel_id=STONE,
                                    center_x=center[0],
                                    center_y=center[1],
                                    center_z=center[2],
                                    radius=radius)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

        if new_world:
            print("WORLD BUILDING STARTED")
            self.build_chunks()
            self.build_chunk_mesh()
            print("WORLD BUILDING DONE")
        self.voxel_handler = VoxelHandler(self)

        self.world_yaw = 0.0
        self.world_pitch = 0.0
        self.world_scale = 1.0

    @property
    def m_model(self):
        m_model = glm.mat4(1.0)
        center = glm.vec3(WORLD_W * CHUNK_SIZE / 2, WORLD_H * CHUNK_SIZE / 2, WORLD_D * CHUNK_SIZE / 2)
        m_model = glm.translate(m_model, center)
        m_model = glm.rotate(m_model, self.world_pitch, glm.vec3(1, 0, 0))
        m_model = glm.rotate(m_model, self.world_yaw, glm.vec3(0, 1, 0))
        m_model = glm.scale(m_model, glm.vec3(self.world_scale))
        m_model = glm.translate(m_model, -center)
        return m_model

    def update(self):
        if not self.new_world:
            inv_model = glm.inverse(self.m_model)
            local_player_pos = inv_model * glm.vec4(self.engine.player.position, 1.0)
            x = int(local_player_pos.x // CHUNK_SIZE)
            y = int(local_player_pos.y // CHUNK_SIZE)
            z = int(local_player_pos.z // CHUNK_SIZE)
            self.load_visible_chunks(x, y, z)
        self.voxel_handler.update()

    def load_visible_chunks(self, center_x, center_y, center_z):
        WORLD_W_local = WORLD_W
        WORLD_H_local = WORLD_H
        WORLD_D_local = WORLD_D
        WORLD_AREA_local = WORLD_AREA
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

    def build_chunks(self):
        # Reset arrays
        for i in range(WORLD_VOL):
            self.chunks[i] = None
        self.voxels.fill(0)

        positions = []
        indices = []
        print("INDICES MARKING STARTED")
        for x in range(WORLD_W):
            for y in range(WORLD_H):
                for z in range(WORLD_D):
                    idx = x + WORLD_W * z + WORLD_AREA * y
                    positions.append((x, y, z))
                    indices.append(idx)
        print("INDICES MARKING DONE")

        # Generate chunks in parallel and save each immediately
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for pos, idx in zip(positions, indices):
                future = executor.submit(self._generate_chunk_data, pos, idx)
                futures[future] = (idx, pos)

            for future in as_completed(futures):
                idx, pos = futures[future]
                try:
                    chunk, voxels = future.result()
                    self.chunks[idx] = chunk
                    self.voxels[idx] = voxels
                except Exception as e:
                    print(f"Error generating chunk at {pos}: {e}")

        # Build meshes after all chunks are generated
        for chunk in self.chunks:
            if chunk:
                chunk.build_mesh()

        # No monolithic save – each chunk was saved already

    def _generate_chunk_data(self, pos, idx):
        x, y, z = pos
        chunk = Chunk(self, position=(x, y, z), generator=self.generator)
        voxels = chunk.build_voxels()
        chunk.voxels = voxels

        # Save this chunk to disk
        chunk_path = CHUNK_FILE_BASE_DIR / f"chunk_{idx}.npz"
        save_chunk(chunk_path, voxels)

        return chunk, voxels

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
                chunk.render()