# world_objects/ghost_region.py
import glm
from meshes.cube_mesh import CubeMesh

class GhostRegion:
    def __init__(self, engine):
        self.engine = engine
        self.mesh = CubeMesh(engine)   # uses voxel_marker shader
        self.position = glm.vec3(0)
        self.size = 1
        self.visible = False

    def render(self):
        if not self.visible:
            return
        prog = self.mesh.program
        prog['m_proj'].write(self.engine.player.m_proj)
        prog['m_view'].write(self.engine.player.m_view)

        # mode_id 0 = red (for remove), 1 = blue (for add). We'll use red for ghost.
        prog['mode_id'] = 0

        # Scale cube to region size
        scale = glm.scale(glm.mat4(1.0), glm.vec3(self.size))
        trans = glm.translate(glm.mat4(1.0), self.position)
        model = trans * scale
        prog['m_model'].write(model)

        # Enable wireframe for ghost
        self.engine.ctx.wireframe = True
        self.mesh.vao.render()
        self.engine.ctx.wireframe = False