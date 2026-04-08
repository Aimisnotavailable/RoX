from settings import *
from world_objects.chunk import Chunk
from world_handler.voxel_handler import VoxelHandler
from world_handler.world_data_handler import save_world, load_chunk_by_index
from concurrent.futures import ThreadPoolExecutor, as_completed


class World:
    def __init__(self, engine, new_world=False):
        self.engine = engine
        self.chunks = [None for _ in range(WORLD_VOL)]
        self.voxels = np.empty([WORLD_VOL, CHUNK_VOL], dtype='uint8')
        self.new_world = new_world
        if new_world:
            self.build_chunks()
            self.build_chunk_mesh()
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
        
        # Apply the scale right after rotation!
        m_model = glm.scale(m_model, glm.vec3(self.world_scale))
        
        m_model = glm.translate(m_model, -center)
        return m_model

    def update(self):
        if not self.new_world:
            # --- THE INVERSE MATRIX TRICK ---
            # Transform the camera's global position into the chunk grid's local space
            inv_model = glm.inverse(self.m_model)
            local_player_pos = inv_model * glm.vec4(self.engine.player.position, 1.0)
            
            # Now calculate the chunk coordinates based on this local position
            x = int(local_player_pos.x // CHUNK_SIZE)
            y = int(local_player_pos.y // CHUNK_SIZE)
            z = int(local_player_pos.z // CHUNK_SIZE)
            
            self.load_visible_chunks(x, y, z)
            
        self.voxel_handler.update()
    
    def load_visible_chunks(self, center_x, center_y, center_z):
        # localize constants for speed
        WORLD_W_local = WORLD_W
        WORLD_H_local = WORLD_H
        WORLD_D_local = WORLD_D
        WORLD_AREA_local = WORLD_AREA
        CHUNK_VOL_local = CHUNK_VOL
        R = RENDER_DISTANCE // 2

        x0 = max(0, center_x - R)
        x1 = min(WORLD_W_local, center_x + R)
        z0 = max(0, center_z - R)
        z1 = min(WORLD_D_local, center_z + R)
        y0 = 0
        y1 = WORLD_H_local

        mesh_to_build = []
        load_tasks = []
        path = CHUNK_FILE_BASE_DIR / "world.dat"

        # small thread pool for I/O; tune max_workers to your disk and CPU
        with ThreadPoolExecutor(max_workers=4) as ex:
            for xi in range(x0, x1):
                for yi in range(y0, y1):
                    for zi in range(z0, z1):
                        idx = xi + WORLD_W_local * zi + WORLD_AREA_local * yi

                        # skip if chunk already exists (loaded)
                        if self.chunks[idx] is not None:
                            continue

                        # schedule load; do not create Chunk yet
                        future = ex.submit(load_chunk_by_index, path, idx)
                        load_tasks.append((future, idx, (xi, yi, zi)))

            # collect results and create chunks
            for future, idx, pos in ((f, i, p) for f, i, p in load_tasks):
                try:
                    vox = future.result()
                except Exception:
                    # on error, skip or create empty chunk
                    vox = np.zeros(CHUNK_VOL_local, dtype=np.uint8)

                # validate and normalize voxel array length
                if vox is None:
                    vox = np.zeros(CHUNK_VOL_local, dtype=np.uint8)
                elif vox.size != CHUNK_VOL_local:
                    if vox.size < CHUNK_VOL_local:
                        padded = np.zeros(CHUNK_VOL_local, dtype=np.uint8)
                        padded[:vox.size] = vox
                        vox = padded
                    else:
                        vox = vox[:CHUNK_VOL_local]

                # create chunk and assign backing array
                chunk = Chunk(self, position=pos)
                self.chunks[idx] = chunk
                self.voxels[idx] = vox
                chunk.voxels = self.voxels[idx]
                chunk.is_empty = not np.any(vox)
                mesh_to_build.append(chunk)

        # build meshes in batch (your existing method)
        self.rebuild_chunk_mesh(mesh_to_build)



    # def build_chunk_by_position(self, pos=(0,0,0)):
    #     x, y, z = pos
        
    #     if not (0 <= x < WORLD_W and 0 <= z < WORLD_D):
    #         return
        
    #     mesh_to_build = []
    #     for x in range(x - RENDER_DISTANCE // 2, x + RENDER_DISTANCE // 2):
    #         if x < 0 or x >= WORLD_W:
    #             continue
    #         for y in range(WORLD_H):
    #             if y < 0 or y >= WORLD_H:
    #                 continue
    #             for z in range(z - RENDER_DISTANCE // 2, z + RENDER_DISTANCE // 2):
    #                 if z < 0 or z >= WORLD_D:
    #                     continue
    #                 chunk_index = x + WORLD_W * z + WORLD_AREA * y

    #                 if self.chunks[chunk_index]:
    #                     continue
    #                 chunk = Chunk(self, position=(x, y, z))

    #                 vox = load_chunk_by_index(CHUNK_FILE_BASE_DIR / "world.dat", chunk_index)
                    
    #                 self.chunks[chunk_index] = chunk

    #                 # put the chunk voxels in a separate array
    #                 self.voxels[chunk_index] = vox
    #                 # get pointer to voxels
    #                 chunk.voxels = self.voxels[chunk_index]
    #                 self.chunks[chunk_index].is_empty = False
    #                 mesh_to_build.append(chunk)
    #     self.rebuild_chunk_mesh(mesh_to_build)

        # self.rebuild_chunk_mesh(pos)
        
    def build_chunks(self):
        for x in range(WORLD_W):
            for y in range(WORLD_H):
                for z in range(WORLD_D):
                    chunk = Chunk(self, position=(x, y, z))

                    chunk_index = x + WORLD_W * z + WORLD_AREA * y
                    self.chunks[chunk_index] = chunk

                    # put the chunk voxels in a separate array
                    self.voxels[chunk_index] = chunk.build_voxels()

                    # get pointer to voxels
                    chunk.voxels = self.voxels[chunk_index]
        save_world(CHUNK_FILE_BASE_DIR /  "world.dat", self.voxels)

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
