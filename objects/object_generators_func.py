# object_generators.py
import numpy as np
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
# Top‑level shape functions (all are njit‑compiled for speed)
# ----------------------------------------------------------------------
@njit(fastmath=True)
def _sphere_func(x, y, z, cx, cy, cz, r):
    return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r**2

@njit(fastmath=True)
def _torus_func(x, y, z, cx, cy, cz, R, r):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    return (np.sqrt(dx * dx + dz * dz) - R) ** 2 + dy * dy <= r * r

@njit(fastmath=True)
def _cube_func(x, y, z, cx, cy, cz, hs):
    return (np.abs(x - cx) <= hs) & (np.abs(y - cy) <= hs) & (np.abs(z - cz) <= hs)

@njit(fastmath=True)
def _cylinder_func_y(x, y, z, cx, cy, cz, r, hh):
    dx = x - cx
    dz = z - cz
    return (dx * dx + dz * dz <= r * r) & (np.abs(y - cy) <= hh)

@njit(fastmath=True)
def _cylinder_func_x(x, y, z, cx, cy, cz, r, hh):
    dy = y - cy
    dz = z - cz
    return (dy * dy + dz * dz <= r * r) & (np.abs(x - cx) <= hh)

@njit(fastmath=True)
def _cylinder_func_z(x, y, z, cx, cy, cz, r, hh):
    dx = x - cx
    dy = y - cy
    return (dx * dx + dy * dy <= r * r) & (np.abs(z - cz) <= hh)

@njit(fastmath=True)
def _sine_func(x, y, z, amp, wl, thick):
    surface = amp * np.sin(2 * np.pi * x / wl)
    return (y >= surface - thick) & (y <= surface + thick)

@njit(fastmath=True)
def _wave_func(x, y, z, amp, wx, wz, thick):
    surface = amp * np.sin(2 * np.pi * x / wx) * np.cos(2 * np.pi * z / wz)
    return (y >= surface - thick) & (y <= surface + thick)

@njit(fastmath=True)
def _hill_func(x, y, z, cx, cy, cz, r, h):
    dx = x - cx
    dz = z - cz
    dist = np.sqrt(dx * dx + dz * dz)
    surface = cy + h * (1 - dist / r)
    return (y <= surface) & (dist <= r)

@njit(fastmath=True)
def _pyramid_func(x, y, z, cx, cy, cz, hh, ht):
    dx = np.abs(x - cx)
    dz = np.abs(z - cz)
    max_d = np.maximum(dx, dz)
    inside_base = max_d <= hh
    surface = cy + ht * (1 - max_d / hh)
    return (y <= surface) & inside_base

@njit(fastmath=True)
def _goursat_func(x, y, z, cx, cy, cz, scale):
    dx = (x - cx) / scale
    dy = (y - cy) / scale
    dz = (z - cz) / scale
    value = 0.5 * (dx**4 + dy**4 + dz**4) - 8 * (dx**2 + dy**2 + dz**2) + 60
    return value <= 0

@njit(fastmath=True)
def _steinmetz_func(x, y, z, cx, cy, cz, r):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    return (dx*dx + dy*dy <= r*r) & (dx*dx + dz*dz <= r*r)

@njit(fastmath=True)
def _heart_func(x, y, z, cx, cy, cz, scale):
    dx = (x - cx) / scale
    dy = (y - cy) / scale
    dz = (z - cz) / scale
    term1 = dx*dx + 2.25*dy*dy + dz*dz - 1.0
    val = term1**3 - dx*dx * dz**3 - 0.1125 * dy*dy * dz**3
    return val <= 0

@njit(fastmath=True)
def _spiked_sphere_func(x, y, z, cx, cy, cz, radius, spikes, amplitude):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    # Polar angle from y-axis, azimuthal from x-axis
    theta = np.arctan2(np.sqrt(dx*dx + dz*dz), dy)
    phi = np.arctan2(dz, dx)
    # Spiky modulation
    spike = 1.0 + amplitude * (np.cos(spikes * phi) * np.sin(spikes * theta / 2))
    return r <= radius * spike

@njit(fastmath=True)
def _rounded_octahedron_func(x, y, z, cx, cy, cz, size, exponent):
    dx = np.abs(x - cx)
    dy = np.abs(y - cy)
    dz = np.abs(z - cz)
    # Superquadric: |x|^e + |y|^e + |z|^e <= size^e
    return dx**exponent + dy**exponent + dz**exponent <= size**exponent

@njit(fastmath=True)
def _mobius_func(x, y, z, cx, cy, cz, R, width, thickness):
    dx = x - cx
    dy = y - cy
    dz = z - cz
    rho = np.sqrt(dx*dx + dz*dz)
    theta = np.arctan2(dz, dx)
    cos_half = np.cos(theta / 2)
    # Vectorized conditional: where cos_half is not near zero, use dy/cos_half, else dy*2
    u = np.where(np.abs(cos_half) > 0.1, dy / cos_half, dy * 2.0)
    rho_target = R + u * np.sin(theta / 2)
    dist_rho = np.abs(rho - rho_target)
    dist_y = np.abs(dy - u * cos_half)
    return (dist_rho**2 + dist_y**2) <= thickness**2


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
    bbox = (cx - R - r - 1, cx + R + r + 1,
            cy - r - 1, cy + r + 1,
            cz - R - r - 1, cz + R + r + 1)
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
        bbox = (cx - radius, cx + radius, cy - half_h, cy + half_h, cz - radius, cz + radius)
        func = _cylinder_func_y
    elif axis == "x":
        bbox = (cx - half_h, cx + half_h, cy - radius, cy + radius, cz - radius, cz + radius)
        func = _cylinder_func_x
    elif axis == "z":
        bbox = (cx - radius, cx + radius, cy - radius, cy + radius, cz - half_h, cz + half_h)
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
    gen = FunctionWorldGenerator(_sine_func, voxel_id, amp=amplitude, wl=wavelength, thick=thickness)
    gen.bbox = bbox
    return gen

def wave_generator(amplitude=10, wavelength_x=20, wavelength_z=20, thickness=3, voxel_id=STONE):
    world_w = WORLD_W * CHUNK_SIZE
    world_d = WORLD_D * CHUNK_SIZE
    bbox = (0, world_w - 1, -amplitude - thickness, amplitude + thickness, 0, world_d - 1)
    gen = FunctionWorldGenerator(_wave_func, voxel_id, amp=amplitude, wx=wavelength_x, wz=wavelength_z, thick=thickness)
    gen.bbox = bbox
    return gen

def hill_generator(radius=80, height=40, center=None, voxel_id=STONE):
    if center is None:
        center = (WORLD_W * CHUNK_SIZE // 2, 0, WORLD_D * CHUNK_SIZE // 2)
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

def goursat_generator(center=None, scale=1.0, voxel_id=STONE):
    if center is None:
        center = (WORLD_W * CHUNK_SIZE // 2, WORLD_H * CHUNK_SIZE // 2, WORLD_D * CHUNK_SIZE // 2)
    cx, cy, cz = center
    max_extent = 4.2 * scale
    bbox = (cx - max_extent, cx + max_extent, cy - max_extent, cy + max_extent, cz - max_extent, cz + max_extent)
    gen = FunctionWorldGenerator(_goursat_func, voxel_id, cx=cx, cy=cy, cz=cz, scale=scale)
    gen.bbox = bbox
    return gen

def steinmetz_generator(center=None, radius=40, voxel_id=STONE):
    if center is None:
        center = (WORLD_W * CHUNK_SIZE // 2, WORLD_H * CHUNK_SIZE // 2, WORLD_D * CHUNK_SIZE // 2)
    cx, cy, cz = center
    bbox = (cx - radius, cx + radius, cy - radius, cy + radius, cz - radius, cz + radius)
    gen = FunctionWorldGenerator(_steinmetz_func, voxel_id, cx=cx, cy=cy, cz=cz, r=radius)
    gen.bbox = bbox
    return gen

def heart_generator(center=None, scale=30, voxel_id=STONE):
    if center is None:
        center = (WORLD_W * CHUNK_SIZE // 2, WORLD_H * CHUNK_SIZE // 2, WORLD_D * CHUNK_SIZE // 2)
    cx, cy, cz = center
    max_extent = 1.5 * scale
    bbox = (cx - max_extent, cx + max_extent, cy - max_extent, cy + max_extent, cz - max_extent, cz + max_extent)
    gen = FunctionWorldGenerator(_heart_func, voxel_id, cx=cx, cy=cy, cz=cz, scale=scale)
    gen.bbox = bbox
    return gen

def spiked_sphere_generator(center=None, radius=40, spikes=8, amplitude=0.3, voxel_id=STONE):
    if center is None:
        center = (WORLD_W * CHUNK_SIZE // 2, WORLD_H * CHUNK_SIZE // 2, WORLD_D * CHUNK_SIZE // 2)
    cx, cy, cz = center
    max_extent = radius * (1 + amplitude) + 1
    bbox = (cx - max_extent, cx + max_extent, cy - max_extent, cy + max_extent, cz - max_extent, cz + max_extent)
    gen = FunctionWorldGenerator(_spiked_sphere_func, voxel_id, cx=cx, cy=cy, cz=cz, radius=radius, spikes=spikes, amplitude=amplitude)
    gen.bbox = bbox
    return gen

def rounded_octahedron_generator(center=None, size=40, exponent=4, voxel_id=STONE):
    if center is None:
        center = (WORLD_W * CHUNK_SIZE // 2, WORLD_H * CHUNK_SIZE // 2, WORLD_D * CHUNK_SIZE // 2)
    cx, cy, cz = center
    bbox = (cx - size, cx + size, cy - size, cy + size, cz - size, cz + size)
    gen = FunctionWorldGenerator(_rounded_octahedron_func, voxel_id, cx=cx, cy=cy, cz=cz, size=size, exponent=exponent)
    gen.bbox = bbox
    return gen

def mobius_generator(center=None, R=40, width=20, thickness=3, voxel_id=STONE):
    if center is None:
        center = (WORLD_W * CHUNK_SIZE // 2, WORLD_H * CHUNK_SIZE // 2, WORLD_D * CHUNK_SIZE // 2)
    cx, cy, cz = center
    max_extent = R + width/2 + thickness + 5
    bbox = (cx - max_extent, cx + max_extent, cy - max_extent, cy + max_extent, cz - max_extent, cz + max_extent)
    gen = FunctionWorldGenerator(_mobius_func, voxel_id, cx=cx, cy=cy, cz=cz, R=R, width=width, thickness=thickness)
    gen.bbox = bbox
    return gen