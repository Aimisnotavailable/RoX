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
        # self.voxel_handler = VoxelHandler(world)
    
    def update(self):
        for local_world in self.local_worlds:
            local_world.update()

    def render(self):
        for local_world in self.local_worlds:
            local_world.render()