from settings import *
from meshes.cube_mesh import CubeMesh


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
        
        # Calculate local marker position
        local_model = glm.translate(glm.mat4(), glm.vec3(self.position))
        
        # Multiply by the spinning world matrix 
        final_model = self.engine.scene.world.m_model * local_model
        
        # --- FIX: Convert PyGLM matrix to bytes ---
        self.mesh.program['m_model'].write(final_model.to_bytes())

    def get_model_matrix(self):
        # Fallback method in case anything else calls it
        m_model = glm.translate(glm.mat4(), glm.vec3(self.position))
        return m_model

    def render(self):
        # Render if we are hovering over a block OR actively dragging
        if self.handler.voxel_id or self.handler.is_dragging:
            self.set_uniform()
            self.mesh.render()