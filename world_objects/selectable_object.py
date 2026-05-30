import glm
import numpy as np
from settings import CHUNK_SIZE
from abc import ABC, abstractmethod

class SelectableObject(ABC):
    """Base class for any object that can be selected and transformed in the 3D world."""
    
    def __init__(self):
        # Transform components
        self.position = glm.vec3(0.0)
        self.rotation = glm.quat()          # identity quaternion
        self.scale = glm.vec3(1.0)
        
    @property
    def model_matrix(self) -> glm.mat4:
        """Compose model matrix from position, rotation, scale."""
        m_model = glm.mat4(1.0)
        center = glm.vec3(self.dimensions[0] * CHUNK_SIZE / 2, self.dimensions[1] * CHUNK_SIZE / 2, self.dimensions[2] * CHUNK_SIZE / 2)
        m_model = glm.translate(m_model, center)
        m_model = glm.rotate(m_model, self.world_pitch, glm.vec3(1, 0, 0))
        m_model = glm.rotate(m_model, self.world_yaw, glm.vec3(0, 1, 0))
        m_model = glm.scale(m_model, glm.vec3(self.world_scale))
        m_model = glm.translate(m_model, -center)
        return m_model
    
    @abstractmethod
    def get_local_aabb(self) -> tuple[glm.vec3, glm.vec3]:
        """
        Return axis‑aligned bounding box in local space: (min_point, max_point).
        """
        pass
    
    def get_global_aabb(self) -> tuple[glm.vec3, glm.vec3]:
        """Transform local AABB to global space (conservative)."""
        local_min, local_max = self.get_local_aabb()
        corners = [
            glm.vec3(local_min.x, local_min.y, local_min.z),
            glm.vec3(local_min.x, local_min.y, local_max.z),
            glm.vec3(local_min.x, local_max.y, local_min.z),
            glm.vec3(local_min.x, local_max.y, local_max.z),
            glm.vec3(local_max.x, local_min.y, local_min.z),
            glm.vec3(local_max.x, local_min.y, local_max.z),
            glm.vec3(local_max.x, local_max.y, local_min.z),
            glm.vec3(local_max.x, local_max.y, local_max.z),
        ]
        global_corners = [glm.vec3(self.model_matrix * glm.vec4(c, 1.0)) for c in corners]
        xs = [c.x for c in global_corners]
        ys = [c.y for c in global_corners]
        zs = [c.z for c in global_corners]
        return glm.vec3(min(xs), min(ys), min(zs)), glm.vec3(max(xs), max(ys), max(zs))
    
    def ray_intersect(self, ray_origin: glm.vec3, ray_dir: glm.vec3) -> float | None:
        """
        Test intersection with the object's bounding box.
        Returns distance along ray if hit, else None.
        """
        inv_model = glm.inverse(self.model_matrix)
        local_origin = glm.vec3(inv_model * glm.vec4(ray_origin, 1.0))
        local_dir = glm.vec3(inv_model * glm.vec4(ray_dir, 0.0))
        
        local_min, local_max = self.get_local_aabb()
        t_min = (local_min.x - local_origin.x) / local_dir.x if local_dir.x != 0 else -float('inf')
        t_max = (local_max.x - local_origin.x) / local_dir.x if local_dir.x != 0 else float('inf')
        if local_dir.x < 0:
            t_min, t_max = t_max, t_min
        
        ty_min = (local_min.y - local_origin.y) / local_dir.y if local_dir.y != 0 else -float('inf')
        ty_max = (local_max.y - local_origin.y) / local_dir.y if local_dir.y != 0 else float('inf')
        if local_dir.y < 0:
            ty_min, ty_max = ty_max, ty_min
        
        if t_min > ty_max or ty_min > t_max:
            return None
        t_min = max(t_min, ty_min)
        t_max = min(t_max, ty_max)
        
        tz_min = (local_min.z - local_origin.z) / local_dir.z if local_dir.z != 0 else -float('inf')
        tz_max = (local_max.z - local_origin.z) / local_dir.z if local_dir.z != 0 else float('inf')
        if local_dir.z < 0:
            tz_min, tz_max = tz_max, tz_min
        
        if t_min > tz_max or tz_min > t_max:
            return None
        t_min = max(t_min, tz_min)
        t_max = min(t_max, tz_max)
        
        if t_max < 0:
            return None
        return max(0.0, t_min)