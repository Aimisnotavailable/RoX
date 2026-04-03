from settings import *
from world_objects.chunk import Chunk
from world_handler.voxel_handler import VoxelHandler
from world_handler.world_data_handler import save_chunk, load_chunk
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading

from world_handler.world_generators import (
    TerrainWorldGenerator,
    FunctionWorldGenerator,
    sphere_generator,
    torus_generator,
    cube_generator,
    cylinder_generator,
    sinewave_generator,
    wave_generator,
    hill_generator,
    pyramid_generator,
)


class World:
    def __init__(self, engine, new_world=False, generator_type='terrain', build_meshes=True, **gen_kwargs):
        self.engine = engine
        self.chunks = [None for _ in range(WORLD_VOL)]
        self.voxels = np.empty([WORLD_VOL, CHUNK_VOL], dtype='uint8')
        self.new_world = new_world

        self.generator_type = generator_type
        self.generator_params = gen_kwargs
        self.generator = self._create_generator(generator_type, **gen_kwargs)

        # Background saver for chunks (non‑blocking I/O)
        self.save_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chunk_saver")

        if new_world:
            get_logger_info("DEBUG", f"Time : {datetime.now()}")
            get_logger_info("DEBUG", f"WORLD BUILDING STARTED")
            self.build_chunks(build_meshes=build_meshes)
            if build_meshes:
                self.build_chunk_mesh()
            get_logger_info("DEBUG", f"WORLD BUILDING DONE")
            get_logger_info("DEBUG", f"Time : {datetime.now()}")
        self.voxel_handler = VoxelHandler(self)

        self.world_yaw = 0.0
        self.world_pitch = 0.0
        self.world_scale = 1.0
        self.world_swapping = False

    def _create_generator(self, generator_type, **kwargs):
        # Default center for shape generators if not provided
        if generator_type in list(WORLD_GEN_PARAMS) and not generator_type in ('sinewave', 'wave'):
            if 'center' not in kwargs:
                kwargs['center'] = (
                    WORLD_W * CHUNK_SIZE // 2,
                    WORLD_H * CHUNK_SIZE // 2,
                    WORLD_D * CHUNK_SIZE // 2
                )
                kwargs.update(WORLD_GEN_PARAMS[generator_type])

        if generator_type == 'terrain':
            return TerrainWorldGenerator()
        elif generator_type == 'sphere':
            return sphere_generator(**kwargs)
        elif generator_type == 'torus':
            return torus_generator(**kwargs)
        elif generator_type == 'cube':
            return cube_generator(**kwargs)
        elif generator_type == 'cylinder':
            return cylinder_generator(**kwargs)
        elif generator_type == 'sinewave':
            return sinewave_generator(**kwargs)
        elif generator_type == 'wave':
            return wave_generator(**kwargs)
        elif generator_type == 'hill':
            return hill_generator(**kwargs)
        elif generator_type == 'pyramid':
            return pyramid_generator(**kwargs)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

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

    def build_chunks(self, build_meshes=True):
        # Reset arrays
        for i in range(WORLD_VOL):
            self.chunks[i] = None
        self.voxels.fill(0)

        # Bounding box pruning
        bbox = self.generator.get_bounding_box() if hasattr(self.generator, 'get_bounding_box') else None
        positions = []
        indices = []

        if bbox is not None:
            min_cx = max(0, bbox[0] // CHUNK_SIZE)
            max_cx = min(WORLD_W - 1, bbox[1] // CHUNK_SIZE)
            min_cy = max(0, bbox[2] // CHUNK_SIZE)
            max_cy = min(WORLD_H - 1, bbox[3] // CHUNK_SIZE)
            min_cz = max(0, bbox[4] // CHUNK_SIZE)
            max_cz = min(WORLD_D - 1, bbox[5] // CHUNK_SIZE)
            get_logger_info("DEBUG", f"Bounding box pruning: chunk range x[{min_cx},{max_cx}] y[{min_cy},{max_cy}] z[{min_cz},{max_cz}]")
            for cx in range(min_cx, max_cx + 1):
                for cy in range(min_cy, max_cy + 1):
                    for cz in range(min_cz, max_cz + 1):
                        idx = cx + WORLD_W * cz + WORLD_AREA * cy
                        positions.append((cx, cy, cz))
                        indices.append(idx)
        else:
            get_logger_info("DEBUG", "Full world generation (no bounding box)")
            for x in range(WORLD_W):
                for y in range(WORLD_H):
                    for z in range(WORLD_D):
                        idx = x + WORLD_W * z + WORLD_AREA * y
                        positions.append((x, y, z))
                        indices.append(idx)

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
                    chunk.voxels = self.voxels[idx]
                    chunk.is_empty = not np.any(voxels)
                except Exception as e:
                    get_logger_info("ERROR", f"Error generating chunk at {pos}: {e}")

        if build_meshes:
            for chunk in self.chunks:
                if chunk:
                    chunk.build_mesh()

    def _generate_chunk_data(self, pos, idx):
        """Generate voxels and schedule background save (non‑blocking)."""
        chunk = Chunk(self, position=pos, generator=self.generator)
        voxels = chunk.build_voxels()
        chunk.voxels = voxels

        if np.any(voxels):
            # Save in background – do not wait
            self.save_executor.submit(self._save_chunk, idx, voxels)
            return chunk, voxels
        else:
            # Empty chunk – nothing to save
            return None, voxels

    def _save_chunk(self, idx, voxels):
        """Background saving task."""
        chunk_path = CHUNK_FILE_BASE_DIR / f"chunk_{idx}.npz"
        save_chunk(chunk_path, voxels)

    def regenerate_world(self, generator_type, **kwargs):
        get_logger_info("GAME", f"Regenerating world as {generator_type} with {kwargs}")

        self.engine.scene.hud.show_temp_message(f"Regenerating world as {generator_type}...", duration=10.0)
        self.world_swapping = True

        def _generate_data():
            try:
                # Shutdown the old world's saver to avoid race conditions
                self.save_executor.shutdown(wait=True)
                # Create new world with build_meshes=False
                new_world = World(self.engine, new_world=True,
                                  generator_type=generator_type,
                                  build_meshes=False,
                                  **kwargs)
                get_logger_info("GAME", "new world generation done")
                self.engine.scene._pending_world = new_world
            except Exception as e:
                get_logger_info("ERROR", f"World generation failed: {e}")
                self.world_swapping = False

        threading.Thread(target=_generate_data, daemon=True).start()

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

    def shutdown(self):
        """Call this when destroying the world to flush pending saves."""
        self.save_executor.shutdown(wait=True)