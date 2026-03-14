# scene.py
from settings import *
import moderngl as mgl
from world_handler.world import World
from world_objects.voxel_marker import VoxelMarker
from world_objects.water import Water
from world_objects.clouds import Clouds

from utils.hud import HUD
from world_objects.hand_renderer import HandRenderer
from world_objects.ghost_region import GhostRegion

class Scene:
    def __init__(self, engine):
        self.engine = engine
        self.world = World(self.engine)
        self.voxel_marker = VoxelMarker(self.world.voxel_handler)
        self.water = Water(engine)
        self.clouds = Clouds(engine)
        
        self.hand_left = HandRenderer(engine, 'LEFT')
        self.hand_right = HandRenderer(engine, 'RIGHT')

        self.ghost_region = GhostRegion(engine)
        # ---> ADD THIS INITIALIZATION <---
        self.hud = HUD(engine)

    def update(self):
        self.world.update()
        self.voxel_marker.update()
        self.clouds.update()

        ar = self.engine.ar_controller
        if ar:
            self.hand_left.update(ar.smooth_left_landmarks)
            self.hand_right.update(ar.smooth_right_landmarks)


    def render(self):
        # chunks rendering
        self.world.render()

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

        # ---> ADD THIS HUD RENDER PASS <---
        # Disable DEPTH_TEST so the 2D UI doesn't clip into 3D chunks
        self.engine.ctx.disable(mgl.DEPTH_TEST)
        self.engine.ctx.enable(mgl.BLEND)
        
        self.hud.render()
        
        # Re-enable DEPTH_TEST for the next frame
        self.engine.ctx.enable(mgl.DEPTH_TEST)