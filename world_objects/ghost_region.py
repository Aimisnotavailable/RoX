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

        # World model matrix (applies rotation and scale to the entire world)
        world_model = self.engine.scene.world.m_model

        # Build local model: first center the unit cube at origin, then scale, then translate
        center_offset = glm.translate(glm.mat4(1.0), glm.vec3(-0.5, -0.5, -0.5))
        scale = glm.scale(glm.mat4(1.0), glm.vec3(self.size))
        trans = glm.translate(glm.mat4(1.0), self.position)
        local_model = trans * scale * center_offset

        # Combine with world transform so the ghost moves with the rotated/scaled world
        final_model = world_model * local_model
        prog['m_model'].write(final_model)

        # Use green (mode_id = 2) – fallback to blue if shader not updated
        try:
            prog['mode_id'] = 2
        except:
            prog['mode_id'] = 1

        self.mesh.vao.render()