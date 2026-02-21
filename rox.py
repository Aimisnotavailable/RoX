from scripts.config import *
from scripts.ar import *
from src.engine2d import GraphicsEngine2D
from src.engine3d import GraphicsEngine3D


class RoX:

    def __init__(self, type = "2D"):
        self.type = type
        self.engine = GraphicsEngine2D() if self.type == "2D" else GraphicsEngine3D()
        
