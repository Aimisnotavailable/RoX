# scene.py
from settings import *
import moderngl as mgl
from world import World
from world_objects.voxel_marker import VoxelMarker
from world_objects.water import Water
from world_objects.clouds import Clouds

# ---> ADD THIS IMPORT <---
from hud import HUD

class Scene:
    def __init__(self, engine):
        self.engine = engine
        self.world = World(self.engine)
        self.voxel_marker = VoxelMarker(self.world.voxel_handler)
        self.water = Water(engine)
        self.clouds = Clouds(engine)
        
        # ---> ADD THIS INITIALIZATION <---
        self.hud = HUD(engine)

    def update(self):
        self.world.update()
        self.voxel_marker.update()
        self.clouds.update()

    def render(self):
        # chunks rendering
        self.world.render()

        # rendering without cull face
        self.engine.ctx.disable(mgl.CULL_FACE)
        self.clouds.render()
        self.water.render()
        self.engine.ctx.enable(mgl.CULL_FACE)

        # voxel selection
        self.voxel_marker.render()

        # ---> ADD THIS HUD RENDER PASS <---
        # Disable DEPTH_TEST so the 2D UI doesn't clip into 3D chunks
        self.engine.ctx.disable(mgl.DEPTH_TEST)
        self.engine.ctx.enable(mgl.BLEND)
        
        self.hud.render()
        
        # Re-enable DEPTH_TEST for the next frame
        self.engine.ctx.enable(mgl.DEPTH_TEST)