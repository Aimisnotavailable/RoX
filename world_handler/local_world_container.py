from settings import *
from world_objects.chunk import Chunk
from world_handler.voxel_handler import VoxelHandler
from world_handler.local_world import LocalWorld
from world_objects.selectable_object import SelectableObject
import threading

class WorldContainer:
    def __init__(self, engine):
        self.engine = engine
        self.local_worlds : list[LocalWorld] = [LocalWorld(self.engine)]
        self.voxel_handler = VoxelHandler(self)
        
        # Selection state
        self.selected_object: SelectableObject | None = None
        self.active_world: LocalWorld | None = self.local_worlds[0]   # world for voxel editing

    def get_local_world_at(self, world_pos: glm.vec3) -> LocalWorld | None:
        """Return the LocalWorld whose bounds contain world_pos."""
        for lw in self.local_worlds:
            if lw.contains_global(world_pos):
                return lw
        return None

    def get_voxel(self, world_pos: glm.vec3 | glm.ivec3) -> tuple[int, glm.ivec3, Chunk, int] | None:
        """
        Return (voxel_id, local_position_in_chunk, chunk, voxel_index)
        for the voxel at the given global position, or None if air/outside.
        Iterates over all worlds.
        """
        if isinstance(world_pos, glm.ivec3):
            world_pos = glm.vec3(world_pos)
        for lw in self.local_worlds:
            result = lw.get_voxel_global(world_pos)
            if result is not None:
                return result
        return None

    def raycast_object(self, ray_origin: glm.vec3, ray_dir: glm.vec3) -> tuple[SelectableObject, float] | None:
        """
        Find the closest selectable object hit by the ray.
        Returns (object, distance) or None.
        """
        best_obj = None
        best_dist = float('inf')
        for obj in self.local_worlds:
            dist = obj.ray_intersect(ray_origin, ray_dir)
            if dist is not None and dist < best_dist:
                best_dist = dist
                best_obj = obj
        return (best_obj, best_dist) if best_obj else None

    def set_active_world(self, world: LocalWorld):
        self.active_world = world
        self.voxel_handler.local_world = world

    def update(self):
        res = self.raycast_object(self.engine.player.position, self.engine.player.forward)

        if res:
            self.selected_object = res[0]
        for local_world in self.local_worlds:
            local_world.update()
        self.voxel_handler.update()

    def render(self):
        for local_world in self.local_worlds:
            local_world.render()