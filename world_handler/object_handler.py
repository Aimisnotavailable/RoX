from settings import *
from world_handler.local_world_container import WorldContainer

class ObjectHandler:

    def __init__(self, world_container : WorldContainer):
        self.container = world_container
        self.voxel_handler = world_container.voxel_handler
    
    def update(self):
        self.current_local_world = self.voxel_handler.local_world
    
    def object_ray_cast(self):
        pass
    
    def scale_object(self, scale):
        pass

    def rotate_object(self, rotate):
        pass

    def move_object(self, movement : glm.ivec3):
        pos = self.current_local_world.position
        self.current_local_world.position = pos + movement
    