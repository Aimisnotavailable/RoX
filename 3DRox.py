from src.configs.engineconfig import *
from rox import RoX

class RoX3D(RoX):

    def __init__(self, winsize=(WIDTH, HEIGHT)):
        super().__init__("3D", winsize)

RoX3D().run()