from scripts.arconfig import WIDTH, HEIGHT
from scripts.ar import *
from src.engine2d import GraphicsEngine2D
from src.engine3d import GraphicsEngine3D


class RoX:

    def __init__(self, type = "2D", winsize=(WIDTH, HEIGHT)):
        self.type = type
        self.engine = GraphicsEngine2D(winsize) if self.type == "2D" else GraphicsEngine3D(winsize)

    def run(self):
        self.engine.run()
