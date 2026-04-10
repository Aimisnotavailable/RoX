# voxel_marker.py
from settings import *
from meshes.cube_mesh import CubeMesh
import glm

class VoxelMarker:
    def __init__(self, voxel_handler):
        self.engine = voxel_handler.engine
        self.handler = voxel_handler
        self.position = glm.vec3(0)
        self.mesh = CubeMesh(self.engine)

    def update(self):
        # 1. If we are actively dragging a line of blocks, follow the drag cursor!
        if self.handler.is_dragging and self.handler.place_pos is not None:
            self.position = self.handler.place_pos
            
        # 2. Otherwise, if we are just hovering, follow the normal raycast
        elif self.handler.voxel_id:
            if self.handler.interaction_mode == 1: # ADD mode (hovering next to block)
                self.position = self.handler.voxel_world_pos + self.handler.voxel_normal
            else: # REMOVE mode (hovering inside block)
                self.position = self.handler.voxel_world_pos

    def set_uniform(self):
        self.mesh.program['mode_id'] = self.handler.interaction_mode
        
        # Get the active local world (you might need to handle multiple worlds later)
        lw = self.handler.local_world
        
        # Convert world position to local space relative to this world's origin
        local_pos = glm.ivec3(self.position) - lw.position
        
        # Model matrix: translate to local position, then apply world's rotation/scale
        local_model = glm.translate(glm.mat4(), glm.vec3(local_pos))
        final_model = lw.m_model * local_model
        
        self.mesh.program['m_model'].write(final_model.to_bytes())

    def get_model_matrix(self):
        # Fallback method in case anything else calls it
        m_model = glm.translate(glm.mat4(), glm.vec3(self.position))
        return m_model

    def render(self):
        if self.handler.voxel_id or self.handler.is_dragging:
            self.set_uniform()
            # --- FORCE WIREFRAME MODE ON ---
            self.engine.ctx.wireframe = True
            self.mesh.render()
            # --- TURN WIREFRAME OFF SO IT DOESN'T AFFECT THE HUD ---
            self.engine.ctx.wireframe = False