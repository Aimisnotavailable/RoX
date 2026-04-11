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

        # 3. If an object is selected in OBJECT mode, we'll draw its bounding box
        selected = self.engine.scene.world_container.selected_object
        if selected and self.handler.interaction_mode == 4:
            self.object_selected = True
            self.position = selected.position

    def render(self):
        if self.object_selected:
            obj = self.engine.scene.world_container.selected_object
            if obj:
                prog = self.mesh.program
                prog['m_proj'].write(self.engine.player.m_proj)
                prog['m_view'].write(self.engine.player.m_view)
                local_min, local_max = obj.get_local_aabb()
                size = local_max - local_min
                center = (local_min + local_max) * 0.5
                model = obj.model_matrix * glm.translate(glm.mat4(1.0), center) * glm.scale(glm.mat4(1.0), size)
                prog['m_model'].write(model)
                try:
                    prog['mode_id'] = 3  # object selection color (e.g., yellow)
                except:
                    pass
                self.engine.ctx.wireframe = True
                self.mesh.vao.render()
                self.engine.ctx.wireframe = False

        if self.handler.voxel_id or self.handler.is_dragging:
            self.set_uniform()
            self.engine.ctx.wireframe = True
            self.mesh.render()
            self.engine.ctx.wireframe = False

    def set_uniform(self):
        self.mesh.program['mode_id'] = self.handler.interaction_mode
        
        lw = self.handler.local_world
        if lw is None:
            lw = self.engine.scene.world_container.local_worlds[0]
        
        local_model = glm.translate(glm.mat4(), glm.vec3(self.position))
        final_model = lw.m_model * local_model
        self.mesh.program['m_model'].write(final_model.to_bytes())