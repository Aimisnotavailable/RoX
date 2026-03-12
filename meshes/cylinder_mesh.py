import numpy as np
from meshes.base_mesh import BaseMesh
import math

class CylinderMesh(BaseMesh):
    def __init__(self, engine, radius=0.05, height=1.0, sectors=12):
        super().__init__()
        self.engine = engine
        self.ctx = engine.ctx
        self.program = engine.shader_program.hand
        self.radius = radius
        self.height = height
        self.sectors = sectors
        self.vbo_format = '3f 3f'
        self.attrs = ('in_position', 'in_normal')
        self.vao = self.get_vao()

    def get_vertex_data(self):
        vertices = []
        normals = []

        # Generate side vertices (two rings) with normals pointing radially
        for ring in [0, 1]:
            y = -self.height/2 if ring == 0 else self.height/2
            for s in range(self.sectors):
                angle = 2 * math.pi * s / self.sectors
                x = self.radius * math.cos(angle)
                z = self.radius * math.sin(angle)
                vertices.append([x, y, z])
                # normal points radially outward
                nx = math.cos(angle)
                ny = 0
                nz = math.sin(angle)
                normals.append([nx, ny, nz])

        # End caps (center points) with normals pointing along Y
        bottom_center_idx = len(vertices)
        vertices.append([0, -self.height/2, 0])
        normals.append([0, -1, 0])  # down
        top_center_idx = len(vertices)
        vertices.append([0,  self.height/2, 0])
        normals.append([0, 1, 0])   # up

        # Now build triangles
        vertex_data = []
        # Side quads as two triangles
        for s in range(self.sectors):
            next_s = (s + 1) % self.sectors
            a = s
            b = next_s
            c = s + self.sectors
            d = next_s + self.sectors
            # Triangle 1: a, b, c
            for idx in [a, b, c]:
                vertex_data.append(vertices[idx] + normals[idx])
            # Triangle 2: b, d, c
            for idx in [b, d, c]:
                vertex_data.append(vertices[idx] + normals[idx])

        # Bottom cap triangles
        for s in range(self.sectors):
            next_s = (s + 1) % self.sectors
            for idx in [bottom_center_idx, next_s, s]:
                vertex_data.append(vertices[idx] + normals[idx])

        # Top cap triangles
        for s in range(self.sectors):
            next_s = (s + 1) % self.sectors
            for idx in [top_center_idx, s + self.sectors, next_s + self.sectors]:
                vertex_data.append(vertices[idx] + normals[idx])

        return np.array(vertex_data, dtype='f4')