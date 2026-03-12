import numpy as np
from meshes.base_mesh import BaseMesh
import math

class SphereMesh(BaseMesh):
    def __init__(self, engine, radius=0.1, sectors=16, stacks=16):
        super().__init__()
        self.engine = engine
        self.ctx = engine.ctx
        self.program = engine.shader_program.hand
        self.radius = radius
        self.sectors = sectors
        self.stacks = stacks
        self.vbo_format = '3f 3f'  # position and normal
        self.attrs = ('in_position', 'in_normal')
        self.vao = self.get_vao()

    def get_vertex_data(self):
        vertices = []
        normals = []
        indices = []

        # Generate vertices and normals
        for i in range(self.stacks + 1):
            stack_angle = math.pi / 2 - i * math.pi / self.stacks
            xy = self.radius * math.cos(stack_angle)
            z = self.radius * math.sin(stack_angle)

            for j in range(self.sectors + 1):
                sector_angle = j * 2 * math.pi / self.sectors
                x = xy * math.cos(sector_angle)
                y = xy * math.sin(sector_angle)
                # Normal is just normalized position (for a sphere)
                nx = x / self.radius
                ny = y / self.radius
                nz = z / self.radius
                vertices.append([x, y, z])
                normals.append([nx, ny, nz])

        # Generate indices for triangle strips (converted to triangles)
        for i in range(self.stacks):
            k1 = i * (self.sectors + 1)
            k2 = k1 + self.sectors + 1
            for j in range(self.sectors):
                if i != 0:
                    indices.append([k1 + j, k2 + j, k1 + j + 1])
                if i != self.stacks - 1:
                    indices.append([k1 + j + 1, k2 + j, k2 + j + 1])

        # Build vertex array (each index becomes a vertex with position and normal)
        vertex_data = []
        for tri in indices:
            for idx in tri:
                vertex_data.append(vertices[idx] + normals[idx])
        return np.array(vertex_data, dtype='f4')