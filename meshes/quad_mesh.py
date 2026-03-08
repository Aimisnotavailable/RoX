import numpy as np
from meshes.base_mesh import BaseMesh

class QuadMesh(BaseMesh):
    """
    Fullscreen quad mesh with interleaved texcoord (2f) and position (3f).
    Positions are in normalized device coordinates (NDC) so the shader can
    render a fullscreen quad without extra transforms.
    """
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.ctx = self.engine.ctx
        # float format: 2 floats for texcoord, 3 floats for position
        self.vbo_format = '2f 3f'
        self.attrs = ('in_tex_coord', 'in_position')
        self.program = self.engine.shader_program.hud
        self.vao = self.get_vao()

    def get_vertex_data(self):
        # NDC positions (x, y, z) for two triangles covering the screen
        # and matching tex coords (s, t)
        # Triangle order: (0,0)-(1,1)-(1,0) and (0,0)-(0,1)-(1,1)
        vertices = np.array([
            # texcoord   position (NDC)
            (0.0, 0.0, -1.0, -1.0, 0.0),
            (1.0, 1.0,  1.0,  1.0, 0.0),
            (1.0, 0.0,  1.0, -1.0, 0.0),

            (0.0, 0.0, -1.0, -1.0, 0.0),
            (0.0, 1.0, -1.0,  1.0, 0.0),
            (1.0, 1.0,  1.0,  1.0, 0.0),
        ], dtype='f4')

        return vertices
