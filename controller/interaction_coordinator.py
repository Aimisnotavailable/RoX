from enum import Enum, auto

class Mode(Enum):
    EDIT = auto()      # voxel editing inside active world
    OBJECT = auto()    # selecting/transforming whole objects

class InteractionCoordinator:
    def __init__(self, world_container):
        self.container = world_container
        # self.voxel_handler = world_container.voxel_handler
        # self.object_handler = world_container.object_handler
        self.mode = Mode.OBJECT   # start in object mode to see objects first

    def toggle_mode(self):
        self.mode = Mode.OBJECT if self.mode == Mode.EDIT else Mode.EDIT
        # # clear targets when switching
        # self.voxel_handler.reset()
        # self.object_handler.reset()

    def update(self):
        pass
        # if self.mode == Mode.EDIT:
        #     self.voxel_handler.update()
        # else:
        #     self.object_handler.update()

    def handle_event(self, event):
        """Keyboard/mouse events are forwarded to the active handler."""
        pass
        # if self.mode == Mode.EDIT:
        #     self.voxel_handler.handle_event(event)
        # else:
        #     self.object_handler.handle_event(event)