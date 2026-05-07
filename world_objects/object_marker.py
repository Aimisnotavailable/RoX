# renderer/object_marker.py
import glm
from settings import *
from meshes.cube_mesh import CubeMesh

class ObjectMarker:
    def __init__(self, engine):
        self.engine = engine
        self.mesh = CubeMesh(engine)
        self.visible = False
        self.position = glm.vec3(0.0)
        self.size = glm.vec3(1.0)      # half extents? We'll use full size, scaling cube accordingly.

    def set_bounds(self, min_point, max_point):
        self.position = (min_point + max_point) * 0.5
        self.size = max_point - min_point
        self.visible = True

    def clear(self):
        self.visible = False

    def render(self):
        if not self.visible:
            return
        # model: translate to center, scale by size (CubeMesh is 1x1x1)
        mat = glm.translate(glm.mat4(1.0), self.position)
        mat = glm.scale(mat, self.size)
        self.mesh.program['m_proj'].write(self.engine.player.m_proj)
        self.mesh.program['m_view'].write(self.engine.player.m_view)
        self.mesh.program['m_model'].write(mat)
        try:
            self.mesh.program['mode_id'] = 3   # object selection color (yellow/orange)
        except:
            pass
        self.engine.ctx.wireframe = True
        self.mesh.vao.render()
        self.engine.ctx.wireframe = False