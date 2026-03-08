import math
import glm
from frustum import Frustum
from abc import ABC, abstractmethod

# Ensure you have your settings imported
from settings import *

class Camera2D:
    def __init__(self, pos=(0.0,0.0), zoom=1.0, angle=0.0):
        self.pos = [float(pos[0]), float(pos[1])]
        self.zoom = float(zoom)
        self.angle = float(angle)

    def screen_to_world(self, sx, sy, screen_w, screen_h):
        cx, cy = screen_w/2.0, screen_h/2.0
        dx = (sx - cx) / self.zoom
        dy = (sy - cy) / self.zoom
        cos_a = math.cos(-self.angle)
        sin_a = math.sin(-self.angle)
        wx = cos_a*dx - sin_a*dy + self.pos[0]
        wy = sin_a*dx + cos_a*dy + self.pos[1]
        return (wx, wy)

    def world_to_screen(self, wx, wy, screen_w, screen_h):
        cx, cy = screen_w/2.0, screen_h/2.0
        dx = wx - self.pos[0]
        dy = wy - self.pos[1]
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        sx = cos_a*dx - sin_a*dy
        sy = sin_a*dx + cos_a*dy
        return (int(sx*self.zoom + cx), int(sy*self.zoom + cy))

    def clamp_zoom(self, min_z=ZOOM_MIN, max_z=ZOOM_MAX):
        self.zoom = max(min_z, min(max_z, self.zoom))


class Camera3D(ABC):
    def __init__(self, type="FPS"):
        self.type = type
        self.frustum = Frustum(self)
        self.movement_rel = (0, 0)
        
        self.position = glm.vec3(0, 0, 0)
        self.forward = glm.vec3(0, 0, -1)
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        
        self.yaw = 0.0
        self.pitch = 0.0
        self.m_proj = glm.mat4()
        self.m_view = glm.mat4()

    @abstractmethod
    def update_vectors(self):
        raise NotImplementedError
        
    def update(self):
        self.update_vectors()
        self.m_view = glm.lookAt(self.position, self.position + self.forward, self.up)


class FPSCamera(Camera3D):
    def __init__(self, position=PLAYER_POS, yaw=-90, pitch=0):
        super().__init__("FPS")
        self.position = glm.vec3(position)
        self.yaw = glm.radians(yaw)
        self.pitch = glm.radians(pitch)

        self.m_proj = glm.perspective(V_FOV, ASPECT_RATIO, NEAR, FAR)

    def update_vectors(self):
        self.forward.x = glm.cos(self.yaw) * glm.cos(self.pitch)
        self.forward.y = glm.sin(self.pitch)
        self.forward.z = glm.sin(self.yaw) * glm.cos(self.pitch)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def rotate_pitch(self, delta_y):
        self.pitch -= delta_y
        self.pitch = glm.clamp(self.pitch, -PITCH_MAX, PITCH_MAX)

    def rotate_yaw(self, delta_x):
        self.yaw += delta_x

    def move_left(self, velocity):
        self.position -= self.right * velocity

    def move_right(self, velocity):
        self.position += self.right * velocity

    def move_up(self, velocity):
        self.position += self.up * velocity

    def move_down(self, velocity):
        self.position -= self.up * velocity

    def move_forward(self, velocity):
        self.position += self.forward * velocity

    def move_back(self, velocity):
        self.position -= self.forward * velocity


class RTSCamera(Camera3D):
    def __init__(self, position=PLAYER_POS, yaw=-90, pitch=-60):
        super().__init__("RTS")
        self.aspect_ratio = ASPECT_RATIO

        self.position = glm.vec3(position)
        self.yaw = glm.radians(yaw)
        self.pitch = glm.radians(pitch)

        self.m_proj = glm.perspective(glm.radians(45), self.aspect_ratio, 0.1, 100)
        self._height_offset = 12.0

    def update_vectors(self):
        self.forward.x = glm.cos(self.yaw) * glm.cos(self.pitch)
        self.forward.y = glm.sin(self.pitch)
        self.forward.z = glm.sin(self.yaw) * glm.cos(self.pitch)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def rotate_pitch(self, delta_y):
        self.pitch -= delta_y
        # Prevent the camera from flipping upside down
        self.pitch = glm.clamp(self.pitch, glm.radians(-89.0), glm.radians(89.0))

    def rotate_yaw(self, delta_x):
        self.yaw += delta_x

    # RTS uses flat panning (parallel to the ground)
    def move_forward(self, velocity):
        flat_forward = self._get_flat_forward()
        self.position += flat_forward * velocity

    def move_back(self, velocity):
        flat_forward = self._get_flat_forward()
        self.position -= flat_forward * velocity

    def move_left(self, velocity):
        self.position -= self.right * velocity

    def move_right(self, velocity):
        self.position += self.right * velocity

    def _get_flat_forward(self):
        flat_forward = glm.vec3(self.forward.x, 0, self.forward.z)
        if glm.length(flat_forward) < 1e-6:
            return glm.vec3(0, 0, -1)
        return glm.normalize(flat_forward)