# world_generators.py
import numpy as np
from functools import partial
from settings import *
from numba import njit

# Import terrain generation helpers (they are njit‑compiled)
from utils.terrain_gen import get_height, set_voxel_id


class BaseWorldGenerator:
    """Abstract base class for all world generators."""

    def generate_chunk(self, chunk_voxels: np.ndarray, chunk_pos: tuple) -> None:
        raise NotImplementedError

    def get_bounding_box(self):
        """Return (min_x, max_x, min_y, max_y, min_z, max_z) in world voxel units.
        Return None if the shape occupies the entire world."""
        return None


class TerrainWorldGenerator(BaseWorldGenerator):
    """Default terrain generator using Perlin noise with heightmap caching."""

    _height_cache = None
    _cache_initialized = False

    @classmethod
    def _init_height_cache(cls):
        if cls._cache_initialized:
            return
        world_w = WORLD_W * CHUNK_SIZE
        world_d = WORLD_D * CHUNK_SIZE
        cls._height_cache = np.empty((world_w, world_d), dtype=np.int32)
        for wx in range(world_w):
            for wz in range(world_d):
                cls._height_cache[wx, wz] = get_height(wx, wz)
        cls._cache_initialized = True

    @staticmethod
    @njit
    def _generate_terrain(voxels, cx, cy, cz, height_cache):
        for x in range(CHUNK_SIZE):
            wx = x + cx
            for z in range(CHUNK_SIZE):
                wz = z + cz
                world_height = height_cache[wx, wz]
                local_height = min(world_height - cy, CHUNK_SIZE)
                for y in range(local_height):
                    wy = y + cy
                    set_voxel_id(voxels, x, y, z, wx, wy, wz, world_height)

    def generate_chunk(self, chunk_voxels, chunk_pos):
        cx, cy, cz = chunk_pos
        self._init_height_cache()
        self._generate_terrain(
            chunk_voxels,
            cx * CHUNK_SIZE,
            cy * CHUNK_SIZE,
            cz * CHUNK_SIZE,
            self._height_cache,
        )

    def get_bounding_box(self):
        return None


class FunctionWorldGenerator(BaseWorldGenerator):
    """Generator that fills voxels based on a vectorized boolean function."""

    def __init__(self, func, voxel_id, **func_kwargs):
        self.func = func
        self.voxel_id = voxel_id
        self.kwargs = func_kwargs
        self.bbox = None

    def get_bounding_box(self):
        return self.bbox

    def generate_chunk(self, chunk_voxels, chunk_pos):
        cx, cy, cz = chunk_pos
        base_x = cx * CHUNK_SIZE
        base_y = cy * CHUNK_SIZE
        base_z = cz * CHUNK_SIZE

        x = np.arange(base_x, base_x + CHUNK_SIZE)[:, None, None]
        y = np.arange(base_y, base_y + CHUNK_SIZE)[None, :, None]
        z = np.arange(base_z, base_z + CHUNK_SIZE)[None, None, :]

        mask = self.func(x, y, z, **self.kwargs)

        if mask.shape != (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE):
            mask = np.broadcast_to(mask, (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE))

        # Reorder to match chunk's internal layout: (y, z, x)
        mask = mask.transpose(1, 2, 0)
        chunk_voxels[mask.ravel()] = self.voxel_id


# ----------------------------------------------------------------------
# Top‑level shape functions (picklable)
# ----------------------------------------------------------------------
def _sphere_func(x, y, z, cx, cy, cz, r):
    return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r**2


def _torus_func(x, y, z, cx, cy, cz, R, r):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    return (np.sqrt(dx * dx + dz * dz) - R) ** 2 + dy * dy <= r * r


def _cube_func(x, y, z, cx, cy, cz, hs):
    return (np.abs(x - cx) <= hs) & (np.abs(y - cy) <= hs) & (np.abs(z - cz) <= hs)


def _cylinder_func_y(x, y, z, cx, cy, cz, r, hh):
    dx = x - cx
    dz = z - cz
    return (dx * dx + dz * dz <= r * r) & (np.abs(y - cy) <= hh)


def _cylinder_func_x(x, y, z, cx, cy, cz, r, hh):
    dy = y - cy
    dz = z - cz
    return (dy * dy + dz * dz <= r * r) & (np.abs(x - cx) <= hh)


def _cylinder_func_z(x, y, z, cx, cy, cz, r, hh):
    dx = x - cx
    dy = y - cy
    return (dx * dx + dy * dy <= r * r) & (np.abs(z - cz) <= hh)


def _sine_func(x, y, z, amp, wl, thick):
    surface = amp * np.sin(2 * np.pi * x / wl)
    return (y >= surface - thick) & (y <= surface + thick)


def _wave_func(x, y, z, amp, wx, wz, thick):
    surface = amp * np.sin(2 * np.pi * x / wx) * np.cos(2 * np.pi * z / wz)
    return (y >= surface - thick) & (y <= surface + thick)


def _hill_func(x, y, z, cx, cy, cz, r, h):
    dx = x - cx
    dz = z - cz
    dist = np.sqrt(dx * dx + dz * dz)
    surface = cy + h * (1 - dist / r)
    return (y <= surface) & (dist <= r)


def _pyramid_func(x, y, z, cx, cy, cz, hh, ht):
    dx = np.abs(x - cx)
    dz = np.abs(z - cz)
    max_d = np.maximum(dx, dz)
    surface = cy + ht * (1 - max_d / hh) * (max_d <= hh)
    return y <= surface


# ----------------------------------------------------------------------
# Factory functions (return a configured FunctionWorldGenerator)
# ----------------------------------------------------------------------
def sphere_generator(center, radius, voxel_id=STONE):
    cx, cy, cz = center
    r = radius
    bbox = (cx - r, cx + r, cy - r, cy + r, cz - r, cz + r)
    gen = FunctionWorldGenerator(_sphere_func, voxel_id, cx=cx, cy=cy, cz=cz, r=r)
    gen.bbox = bbox
    return gen


def torus_generator(center, R, r, voxel_id=STONE):
    cx, cy, cz = center
    bbox = (
        cx - R - r - 1,
        cx + R + r + 1,
        cy - r - 1,
        cy + r + 1,
        cz - R - r - 1,
        cz + R + r + 1,
    )
    gen = FunctionWorldGenerator(_torus_func, voxel_id, cx=cx, cy=cy, cz=cz, R=R, r=r)
    gen.bbox = bbox
    return gen


def cube_generator(center, half_size, voxel_id=STONE):
    cx, cy, cz = center
    hs = half_size
    bbox = (cx - hs, cx + hs, cy - hs, cy + hs, cz - hs, cz + hs)
    gen = FunctionWorldGenerator(_cube_func, voxel_id, cx=cx, cy=cy, cz=cz, hs=hs)
    gen.bbox = bbox
    return gen


def cylinder_generator(center, radius, height, voxel_id=STONE, axis="y"):
    cx, cy, cz = center
    half_h = height / 2

    if axis == "y":
        bbox = (
            cx - radius,
            cx + radius,
            cy - half_h,
            cy + half_h,
            cz - radius,
            cz + radius,
        )
        func = _cylinder_func_y
    elif axis == "x":
        bbox = (
            cx - half_h,
            cx + half_h,
            cy - radius,
            cy + radius,
            cz - radius,
            cz + radius,
        )
        func = _cylinder_func_x
    elif axis == "z":
        bbox = (
            cx - radius,
            cx + radius,
            cy - radius,
            cy + radius,
            cz - half_h,
            cz + half_h,
        )
        func = _cylinder_func_z
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    gen = FunctionWorldGenerator(func, voxel_id, cx=cx, cy=cy, cz=cz, r=radius, hh=half_h)
    gen.bbox = bbox
    return gen


def sinewave_generator(amplitude=10, wavelength=20, thickness=3, voxel_id=STONE):
    world_w = WORLD_W * CHUNK_SIZE
    world_d = WORLD_D * CHUNK_SIZE
    bbox = (0, world_w - 1, -amplitude - thickness, amplitude + thickness, 0, world_d - 1)
    gen = FunctionWorldGenerator(
        _sine_func, voxel_id, amp=amplitude, wl=wavelength, thick=thickness
    )
    gen.bbox = bbox
    return gen


def wave_generator(
    amplitude=10, wavelength_x=20, wavelength_z=20, thickness=3, voxel_id=STONE
):
    world_w = WORLD_W * CHUNK_SIZE
    world_d = WORLD_D * CHUNK_SIZE
    bbox = (0, world_w - 1, -amplitude - thickness, amplitude + thickness, 0, world_d - 1)
    gen = FunctionWorldGenerator(
        _wave_func, voxel_id, amp=amplitude, wx=wavelength_x, wz=wavelength_z, thick=thickness
    )
    gen.bbox = bbox
    return gen


def hill_generator(radius=80, height=40, center=None, voxel_id=STONE):
    if center is None:
        center = (
            WORLD_W * CHUNK_SIZE // 2,
            0,
            WORLD_D * CHUNK_SIZE // 2,
        )
    cx, cy, cz = center
    bbox = (cx - radius, cx + radius, cy, cy + height, cz - radius, cz + radius)
    gen = FunctionWorldGenerator(_hill_func, voxel_id, cx=cx, cy=cy, cz=cz, r=radius, h=height)
    gen.bbox = bbox
    return gen


def pyramid_generator(center, half_base, height, voxel_id=STONE):
    cx, cy, cz = center
    hh = half_base
    bbox = (cx - hh, cx + hh, cy, cy + height, cz - hh, cz + hh)
    gen = FunctionWorldGenerator(_pyramid_func, voxel_id, cx=cx, cy=cy, cz=cz, hh=half_base, ht=height)
    gen.bbox = bbox
    return gen