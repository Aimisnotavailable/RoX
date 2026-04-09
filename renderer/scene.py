# scene.py
from settings import *
import moderngl as mgl
from world_handler.local_world_container import WorldContainer
from world_objects.voxel_marker import VoxelMarker
from world_objects.water import Water
from world_objects.clouds import Clouds

from utils.hud import HUD
from world_objects.hand_renderer import HandRenderer
from world_objects.ghost_region import GhostRegion
from world_objects.raycast_ray import RayCastRay

class Scene:
    def __init__(self, engine):
        self.engine = engine
        self.worldcontainer = WorldContainer(self.engine)
        # self._pending_world = None

        self.voxel_marker = VoxelMarker(self.worldcontainer.local_worlds[0].voxel_handler)
        self.water = Water(engine)
        self.clouds = Clouds(engine)
        
        self.hand_left = HandRenderer(engine, 'LEFT')
        self.hand_right = HandRenderer(engine, 'RIGHT')

        self.ghost_region = GhostRegion(engine)

        # Ray beam
        self.ray_beam = RayCastRay(engine, hand_label='RIGHT')
        # ---> ADD THIS INITIALIZATION <---
        self.hud = HUD(engine)

    def update(self):
        # if self._pending_world is not None:
        #     new_world = self._pending_world
        #     self.world = new_world
        #     self.voxel_marker.handler = new_world.voxel_handler
        #     self._pending_world = None
        #     # Build meshes for the new world (in main thread)
        #     new_world.build_chunk_mesh()

        self.worldcontainer.update()
        self.voxel_marker.update()
        self.clouds.update()

        ar = self.engine.ar_controller
        if ar:
            self.hand_left.update(ar.smooth_left_landmarks)
            self.hand_right.update(ar.smooth_right_landmarks)
        
        self.ray_beam.update() 


    def render(self):
        # chunks rendering
        self.worldcontainer.render()

        # rendering without cull face
        self.engine.ctx.disable(mgl.CULL_FACE)
        self.clouds.render()
        self.water.render()
        self.engine.ctx.enable(mgl.CULL_FACE)

        self.hand_left.render()
        self.hand_right.render()

        # voxel selection
        self.voxel_marker.render()
        self.ghost_region.render()

        self.ray_beam.render()

        # ---> ADD THIS HUD RENDER PASS <---
        # Disable DEPTH_TEST so the 2D UI doesn't clip into 3D chunks
        self.engine.ctx.disable(mgl.DEPTH_TEST)
        self.engine.ctx.enable(mgl.BLEND)
        
        self.hud.render()
        
        # Re-enable DEPTH_TEST for the next frame
        self.engine.ctx.enable(mgl.DEPTH_TEST)