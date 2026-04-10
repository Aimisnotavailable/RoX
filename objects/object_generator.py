from settings import *

from objects.object_generators_func import (
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
    goursat_generator,
    steinmetz_generator,
    heart_generator,
    spiked_sphere_generator,
    rounded_octahedron_generator,
    mobius_generator,
)

def create_generator(generator_type, **kwargs):
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
        elif generator_type == 'goursat':
            return goursat_generator(**kwargs)
        elif generator_type == 'steinmetz':
            return steinmetz_generator(**kwargs)
        elif generator_type == 'heart':
            return heart_generator(**kwargs)
        elif generator_type == 'spiked_sphere':
            return spiked_sphere_generator(**kwargs)
        elif generator_type == 'rounded_octahedron':
            return rounded_octahedron_generator(**kwargs)
        elif generator_type == 'mobius':
            return mobius_generator(**kwargs)
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")