# world_objects/hand_shadow.py
import glm
import numpy as np
import pygame as pg
import moderngl as mgl
from meshes.base_mesh import BaseMesh

class ShadowQuadMesh(BaseMesh):
    def __init__(self, engine, size=0.8):
        super().__init__()
        self.engine = engine
        self.ctx = engine.ctx
        self.program = engine.shader_program.shadow
        self.size = size

        # Create a radial gradient texture
        self.texture = self._create_gradient_texture()

        self.vbo_format = '3f 2f'  # position and uv
        self.attrs = ('in_position', 'in_uv')
        self.vao = self.get_vao()

    def _create_gradient_texture(self):
        # Generate a 64x64 radial gradient (white center, transparent edges)
        size = 64
        data = np.zeros((size, size, 4), dtype=np.uint8)
        center = size // 2
        max_dist = center
        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                dist = np.sqrt(dx*dx + dy*dy) / max_dist
                if dist < 1.0:
                    alpha = int(255 * (1.0 - dist))
                else:
                    alpha = 0
                data[y, x] = [0, 0, 0, alpha]  # black with alpha
        texture = self.ctx.texture((size, size), 4, data.tobytes())
        texture.filter = (mgl.LINEAR, mgl.LINEAR)
        return texture

    def get_vertex_data(self):
        s = self.size * 0.5
        vertices = [
            [-s, 0.0, -s, 0.0, 0.0],
            [ s, 0.0, -s, 1.0, 0.0],
            [ s, 0.0,  s, 1.0, 1.0],
            [-s, 0.0,  s, 0.0, 1.0],
            [-s, 0.0, -s, 0.0, 0.0],
            [ s, 0.0,  s, 1.0, 1.0],
        ]
        return np.array(vertices, dtype='f4')

class HandShadow:
    def __init__(self, engine):
        self.engine = engine
        self.mesh = ShadowQuadMesh(engine)
        self.visible = True
        self.position = glm.vec3(0)

    def update(self, hand_center):
        # Project onto ground (y=0)
        self.position = glm.vec3(hand_center.x, 0.0, hand_center.z)

    def render(self):
        if not self.visible:
            return
        prog = self.mesh.program
        prog['m_proj'].write(self.engine.player.m_proj)
        prog['m_view'].write(self.engine.player.m_view)
        prog['m_model'].write(glm.translate(glm.mat4(1.0), self.position))
        self.mesh.texture.use(location=0)
        prog['u_texture'] = 0
        self.mesh.vao.render()