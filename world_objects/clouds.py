from settings import *
from meshes.cloud_mesh import CloudMesh


class Clouds:
    def __init__(self, engine):
        self.engine = engine
        self.mesh = CloudMesh(engine)

    def update(self):
        self.mesh.program['u_time'] = self.engine.time

    def render(self):
        self.mesh.render()
