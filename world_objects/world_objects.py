import glm
import numpy as np
from settings import *
from meshes.chunk_mesh_builder import build_chunk_mesh
from meshes.chunk_mesh import ChunkMesh

class WorldObject:
    def __init__(self, engine, local_voxels: np.ndarray, position=glm.vec3(0)):
        """
        local_voxels: 3D numpy array of shape (SIZE, SIZE, SIZE) with voxel IDs.
                      Must be uint8. Size is fixed for all objects (e.g., 32).
        """
        self.engine = engine
        self.local_voxels = local_voxels   # shape (SIZE, SIZE, SIZE)
        self.size = local_voxels.shape[0]
        self.position = glm.vec3(position)
        self.rotation = glm.quat()         # identity quaternion
        self.scale = 1.0
        
        # Build mesh once (like a ChunkMesh but using a dummy "chunk" interface)
        self.mesh = self._build_mesh()
        
    def _build_mesh(self):
        """Convert local_voxels into vertex data and create a ChunkMesh."""
        # We need to adapt build_chunk_mesh to work on a 3D array directly.
        # See implementation suggestion below.
        vertex_data = build_object_mesh(self.local_voxels, self.size)
        # Create a simple mesh object (could reuse ChunkMesh with a custom builder)
        mesh = ObjectMesh(self.engine, vertex_data)
        return mesh
    
    @property
    def model_matrix(self):
        m = glm.mat4(1.0)
        m = glm.translate(m, self.position)
        m = m * glm.mat4(self.rotation)
        m = glm.scale(m, glm.vec3(self.scale))
        # If your world has a global transform (world.m_model), apply it here:
        # m = self.engine.scene.world.m_model * m
        return m
    
    def render(self):
        self.mesh.render(self.model_matrix)
    
    def raycast(self, ray_origin_world, ray_direction_world):
        """Transform ray to object local space and check against local_voxels."""
        inv_model = glm.inverse(self.model_matrix)
        local_origin = glm.vec3(inv_model * glm.vec4(ray_origin_world, 1.0))
        local_dir = glm.normalize(glm.vec3(inv_model * glm.vec4(ray_direction_world, 0.0)))
        
        # Use a simple voxel traversal (similar to your raycast_generic but on a bounded grid)
        hit, hit_pos_local, hit_normal_local = self._local_raycast(local_origin, local_dir)
        if hit:
            hit_world = glm.vec3(self.model_matrix * glm.vec4(hit_pos_local, 1.0))
            normal_world = glm.normalize(glm.vec3(self.model_matrix * glm.vec4(hit_normal_local, 0.0)))
            return hit_world, normal_world
        return None, None
    
    def _local_raycast(self, origin, direction):
        """A simplified DDA that respects object bounds [0, size]."""
        # ... implement similar to raycast_generic but with bounds checking.
        # Return (hit, hit_local_pos, hit_normal_local)