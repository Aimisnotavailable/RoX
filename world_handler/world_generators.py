# world_generators.py
import numpy as np
from settings import *
from numba import njit

# Import terrain generation helpers (they are njit‑compiled)
from utils.terrain_gen import get_height, set_voxel_id

class BaseWorldGenerator:
    """Abstract base class for all world generators."""
    def generate_chunk(self, chunk_voxels: np.ndarray, chunk_pos: tuple) -> None:
        """
        Fill the voxel array for one chunk.
        :param chunk_voxels: 1D array of length CHUNK_VOL, dtype uint8
        :param chunk_pos: tuple (cx, cy, cz) in chunk coordinates
        """
        raise NotImplementedError

class TerrainWorldGenerator(BaseWorldGenerator):
    """Default terrain generator using Perlin noise (same as original)."""
    @staticmethod
    @njit
    def _generate_terrain(voxels, cx, cy, cz):
        for x in range(CHUNK_SIZE):
            wx = x + cx
            for z in range(CHUNK_SIZE):
                wz = z + cz
                world_height = get_height(wx, wz)
                local_height = min(world_height - cy, CHUNK_SIZE)
                for y in range(local_height):
                    wy = y + cy
                    set_voxel_id(voxels, x, y, z, wx, wy, wz, world_height)

    def generate_chunk(self, chunk_voxels, chunk_pos):
        cx, cy, cz = chunk_pos
        # Convert chunk coordinates to world coordinates (voxel units)
        self._generate_terrain(chunk_voxels,
                               cx * CHUNK_SIZE,
                               cy * CHUNK_SIZE,
                               cz * CHUNK_SIZE)

class FunctionWorldGenerator:
    """Generator that fills voxels based on a vectorized boolean function."""
    def __init__(self, func, voxel_id, **func_kwargs):
        """
        :param func: callable f(x, y, z, **kwargs) -> bool array
        :param voxel_id: integer block type to set
        :param func_kwargs: extra arguments passed to func
        """
        self.func = func
        self.voxel_id = voxel_id
        self.kwargs = func_kwargs

    def generate_chunk(self, chunk_voxels, chunk_pos):
        cx, cy, cz = chunk_pos
        base_x = cx * CHUNK_SIZE
        base_y = cy * CHUNK_SIZE
        base_z = cz * CHUNK_SIZE

        # Create coordinate grids in (x, y, z) order
        x = np.arange(base_x, base_x + CHUNK_SIZE)[:, None, None]
        y = np.arange(base_y, base_y + CHUNK_SIZE)[None, :, None]
        z = np.arange(base_z, base_z + CHUNK_SIZE)[None, None, :]

        # Evaluate the shape function (returns boolean array of shape (48,48,48))
        mask = self.func(x, y, z, **self.kwargs)

        # Reorder axes to match chunk's internal layout: (y, z, x)
        mask = mask.transpose(1, 2, 0)

        # Flatten and assign
        chunk_voxels[mask.ravel()] = self.voxel_id