from meshes.quad_mesh import QuadMesh
from settings import *


class Water:
    def __init__(self, engine):
        self.engine = engine
        self.mesh = QuadMesh(engine)

    def render(self):
        self.mesh.render()
