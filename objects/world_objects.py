from settings import *
from objects.object_generator import create_generator

class WorldObjects:

    def __init__(self, type='sphere', **gen_kwargs):
        self.generator = create_generator(type, **gen_kwargs)