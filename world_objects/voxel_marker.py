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
        self.object_selected = False

    def update(self):
        # 1. If we are actively dragging a line of blocks, follow the drag cursor!
        if self.handler.is_dragging and self.handler.place_pos is not None:
            self.position = self.handler.place_pos
            self.object_selected = False
            
        # 2. Otherwise, if we are just hovering, follow the normal raycast
        elif self.handler.voxel_id:
            if self.handler.interaction_mode == 1: # ADD mode
                self.position = self.handler.voxel_world_pos + self.handler.voxel_normal
            else: # REMOVE mode
                self.position = self.handler.voxel_world_pos
            self.object_selected = False
        else:
            self.object_selected = False

    def render(self):
        if self.handler.voxel_id or self.handler.is_dragging:
            self.set_uniform()
            self.engine.ctx.wireframe = True
            self.mesh.render()
            self.engine.ctx.wireframe = False
#TO-DO
# FIX voxel marker scaling  for the current world
# still broken XD

    def set_uniform(self):
        self.mesh.program['mode_id'] = self.handler.interaction_mode
        
        lw = self.handler.local_world
        if lw is None:
            lw = self.engine.scene.world_container.selected_object
        
        local_model = glm.translate(glm.mat4(), glm.vec3(self.position))
        # final_model = lw.m_model * local_model
        selected_object = self.engine.scene.world_container.selected_object
        if selected_object:
            final_model = glm.scale(local_model, selected_object.scale)
        self.mesh.program['m_model'].write(final_model.to_bytes())