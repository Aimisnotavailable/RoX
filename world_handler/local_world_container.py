from settings import *
from world_objects.chunk import Chunk
from world_handler.voxel_handler import VoxelHandler
from world_handler.world_data_handler import save_chunk, load_chunk
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from world_handler.local_world import LocalWorld

import threading

class WorldContainer:
    def __init__(self, engine):
        self.engine = engine
        self.local_worlds : list[LocalWorld] = [LocalWorld(self.engine)]
        self.voxel_handler = VoxelHandler(self)
    
    def get_local_world_at(self, world_pos: glm.vec3) -> LocalWorld | None:
        """Return the LocalWorld whose bounds contain world_pos."""
        for lw in self.local_worlds:
            if lw.contains(world_pos):
                return lw
        return None

    def get_voxel(self, world_pos: glm.vec3) -> tuple[int, glm.ivec3, Chunk] | None:
        """Return (voxel_id, local_position_in_chunk, chunk) or None if air/outside."""
        lw = self.get_local_world_at(world_pos)
        if lw is None:
            return None
        return lw.get_voxel(world_pos)

    def update(self):
        for local_world in self.local_worlds:
            local_world.update()
        self.voxel_handler.update()

    def render(self):
        for local_world in self.local_worlds:
            local_world.render()