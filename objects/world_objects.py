from settings import *
from objects.object_generator import create_generator

class WorldObjects:

    def __init__(self, dimensions, type='sphere', **gen_kwargs):
        self.generator = create_generator(dimensions, type, **gen_kwargs)