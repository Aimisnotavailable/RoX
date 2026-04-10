from settings import *
from world_handler.local_world_container import WorldContainer

class ObjectHandler:

    def __init__(self, world_container : WorldContainer):
        self.container = world_container
    
    # def raycast_generic(self, origin, direction, is_rts=False):
    #     inv_model = glm.inverse(self.world.m_model)
    #     local_origin = glm.vec3(inv_model * glm.vec4(origin, 1.0))
    #     local_dir = glm.normalize(glm.vec3(inv_model * glm.vec4(direction, 0.0)))

    #     max_dist = 60.0 if is_rts else 60.0
    #     x1, y1, z1 = local_origin
    #     x2, y2, z2 = local_origin + local_dir * max_dist

    #     current_voxel_pos = glm.ivec3(x1, y1, z1)
    #     self.voxel_id = 0
    #     self.voxel_normal = glm.ivec3(0)
    #     step_dir = -1

    #     dx = glm.sign(x2 - x1)
    #     delta_x = min(dx / (x2 - x1), 10000000.0) if dx != 0 else 10000000.0
    #     max_x = delta_x * (1.0 - glm.fract(x1)) if dx > 0 else delta_x * glm.fract(x1)

    #     dy = glm.sign(y2 - y1)
    #     delta_y = min(dy / (y2 - y1), 10000000.0) if dy != 0 else 10000000.0
    #     max_y = delta_y * (1.0 - glm.fract(y1)) if dy > 0 else delta_y * glm.fract(y1)

    #     dz = glm.sign(z2 - z1)
    #     delta_z = min(dz / (z2 - z1), 10000000.0) if dz != 0 else 10000000.0
    #     max_z = delta_z * (1.0 - glm.fract(z1)) if dz > 0 else delta_z * glm.fract(z1)

    #     while not (max_x > 1.0 and max_y > 1.0 and max_z > 1.0):
    #         result = self.get_voxel_id(voxel_world_pos=current_voxel_pos)
    #         if result[0]:
    #             self.voxel_id, self.voxel_index, self.voxel_local_pos, self.chunk = result
    #             self.voxel_world_pos = current_voxel_pos

    #             if step_dir == 0: self.voxel_normal.x = -dx
    #             elif step_dir == 1: self.voxel_normal.y = -dy
    #             else: self.voxel_normal.z = -dz
    #             return True

    #         if max_x < max_y:
    #             if max_x < max_z:
    #                 current_voxel_pos.x += dx
    #                 max_x += delta_x
    #                 step_dir = 0
    #             else:
    #                 current_voxel_pos.z += dz
    #                 max_z += delta_z
    #                 step_dir = 2
    #         else:
    #             if max_y < max_z:
    #                 current_voxel_pos.y += dy
    #                 max_y += delta_y
    #                 step_dir = 1
    #             else:
    #                 current_voxel_pos.z += dz
    #                 max_z += delta_z
    #                 step_dir = 2
    #     self.voxel_world_pos = None
    #     return False