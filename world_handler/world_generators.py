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

        x = np.arange(base_x, base_x + CHUNK_SIZE)[:, None, None]
        y = np.arange(base_y, base_y + CHUNK_SIZE)[None, :, None]
        z = np.arange(base_z, base_z + CHUNK_SIZE)[None, None, :]

        mask = self.func(x, y, z, **self.kwargs)
        
        # Ensure mask has full 3D shape (in case the function doesn't use all axes)
        if mask.shape != (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE):
            mask = np.broadcast_to(mask, (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE))

        # Reorder to match chunk's internal layout: (y, z, x)
        mask = mask.transpose(1, 2, 0)
        chunk_voxels[mask.ravel()] = self.voxel_id

# ---- Predefined shape factories (return a FunctionWorldGenerator) ----
def sphere_generator(center, radius, voxel_id=STONE):
    cx, cy, cz = center
    def sphere_func(x, y, z, cx, cy, cz, r):
        return (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= r**2
    return FunctionWorldGenerator(sphere_func, voxel_id,
                                  cx=cx, cy=cy, cz=cz, r=radius)

def torus_generator(center, R, r, voxel_id=STONE):
    cx, cy, cz = center
    def torus_func(x, y, z, cx, cy, cz, R, r):
        dx = x - cx
        dy = y - cy
        dz = z - cz
        return (np.sqrt(dx*dx + dz*dz) - R)**2 + dy*dy <= r*r
    return FunctionWorldGenerator(torus_func, voxel_id,
                                  cx=cx, cy=cy, cz=cz, R=R, r=r)

def cube_generator(center, half_size, voxel_id=STONE):
    cx, cy, cz = center
    hs = half_size
    def cube_func(x, y, z, cx, cy, cz, hs):
        return (np.abs(x - cx) <= hs) & (np.abs(y - cy) <= hs) & (np.abs(z - cz) <= hs)
    return FunctionWorldGenerator(cube_func, voxel_id,
                                  cx=cx, cy=cy, cz=cz, hs=hs)

def cylinder_generator(center, radius, height, voxel_id=STONE, axis='y'):
    cx, cy, cz = center
    half_h = height / 2
    if axis == 'y':
        def cylinder_func(x, y, z, cx, cy, cz, r, hh):
            dx = x - cx
            dz = z - cz
            return (dx*dx + dz*dz <= r*r) & (np.abs(y - cy) <= hh)
    elif axis == 'x':
        def cylinder_func(x, y, z, cx, cy, cz, r, hh):
            dy = y - cy
            dz = z - cz
            return (dy*dy + dz*dz <= r*r) & (np.abs(x - cx) <= hh)
    elif axis == 'z':
        def cylinder_func(x, y, z, cx, cy, cz, r, hh):
            dx = x - cx
            dy = y - cy
            return (dx*dx + dy*dy <= r*r) & (np.abs(z - cz) <= hh)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return FunctionWorldGenerator(cylinder_func, voxel_id,
                                  cx=cx, cy=cy, cz=cz, r=radius, hh=half_h)

def sinewave_generator(amplitude=10, wavelength=20, voxel_id=STONE):
    def sine_func(x, y, z, amp, wl):
        # y = amp * sin(2π x / wl) – fill below the surface
        surface = amp * np.sin(2 * np.pi * x / wl)
        return y <= surface
    return FunctionWorldGenerator(sine_func, voxel_id, amp=amplitude, wl=wavelength)

def wave_generator(amplitude=10, wavelength_x=20, wavelength_z=20, voxel_id=STONE):
    """3D wave surface: y = A * sin(2πx/λx) * cos(2πz/λz)"""
    def wave_func(x, y, z, amp, wx, wz):
        surface = amp * np.sin(2 * np.pi * x / wx) * np.cos(2 * np.pi * z / wz)
        return y <= surface
    return FunctionWorldGenerator(wave_func, voxel_id, amp=amplitude, wx=wavelength_x, wz=wavelength_z)

def hill_generator(radius=80, height=40, center=None, voxel_id=STONE):
    """Conical hill with given radius and height."""
    if center is None:
        center = (WORLD_W * CHUNK_SIZE // 2, 0, WORLD_D * CHUNK_SIZE // 2)
    cx, cy, cz = center
    def hill_func(x, y, z, cx, cy, cz, r, h):
        dx = x - cx
        dz = z - cz
        dist = np.sqrt(dx*dx + dz*dz)
        surface = cy + h * (1 - dist / r)
        # Only fill where dist <= r
        return (y <= surface) & (dist <= r)
    return FunctionWorldGenerator(hill_func, voxel_id, cx=cx, cy=cy, cz=cz, r=radius, h=height)

def pyramid_generator(center, half_base, height, voxel_id=STONE):
    """Square pyramid."""
    cx, cy, cz = center
    hh = half_base
    def pyramid_func(x, y, z, cx, cy, cz, hh, ht):
        dx = np.abs(x - cx)
        dz = np.abs(z - cz)
        # Linear interpolation: at the center (dx=0, dz=0) max height, at edges height=0
        max_d = max(dx, dz)
        surface = cy + ht * (1 - max_d / hh) * (max_d <= hh)
        return y <= surface
    return FunctionWorldGenerator(pyramid_func, voxel_id, cx=cx, cy=cy, cz=cz, hh=half_base, ht=height)