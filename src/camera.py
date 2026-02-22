from scripts.config import *
from abc import ABC, abstractmethod

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

    @abstractmethod
    def update_camera_vectors(self):
        raise NotImplementedError
    
    @abstractmethod
    def move(self):
        raise NotImplementedError
    
    @abstractmethod
    def on_resize(self, width, height):
        raise NotImplementedError

class FPSCamera(Camera3D):
    def __init__(self, app, position=(0, 0, 4), yaw=-90, pitch=0):
        super().__init__("FPS")
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]
        self.position = glm.vec3(position)
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, -1)
        self.yaw = yaw
        self.pitch = pitch

        # Projection and view matrices
        self.m_proj = glm.perspective(glm.radians(90), self.aspect_ratio, 0.1, 100)
        self.m_view = glm.mat4()

        # Movement and mouse sensitivity
        self.speed = 0.04
        self.sensitivity = 0.1

        # Saved state used when switching to/from RTS mode
        self._saved_state = None

    # -------------------------
    # State save / restore API
    # -------------------------
    def save_state(self):
        """
        Save the current FPS camera state so it can be restored later.
        Called before switching into RTS mode.
        """
        self._saved_state = {
            'position': glm.vec3(self.position),
            'yaw': float(self.yaw),
            'pitch': float(self.pitch)
        }

    def restore_state(self):
        """
        Restore the previously saved FPS camera state.
        If no saved state exists, this is a no-op.
        """
        if self._saved_state is None:
            return
        self.position = glm.vec3(self._saved_state['position'])
        self.yaw = self._saved_state['yaw']
        self.pitch = self._saved_state['pitch']
        self._saved_state = None

    def get_state(self):
        """
        Return a tuple (position, yaw, pitch) representing the current camera state.
        Useful for other cameras to base themselves on the FPS camera.
        """
        return glm.vec3(self.position), float(self.yaw), float(self.pitch)

    def set_state(self, position, yaw, pitch):
        """
        Set the camera state from external values.
        """
        self.position = glm.vec3(position)
        self.yaw = float(yaw)
        self.pitch = float(pitch)

    # -------------------------
    # Rotation / vector updates
    # -------------------------
    def rotate(self):
        """
        Update yaw and pitch from mouse relative movement.
        Ensure pitch is clamped to avoid gimbal lock.
        """
        rel_x, rel_y = pygame.mouse.get_rel()
        self.yaw += rel_x * self.sensitivity
        self.pitch -= rel_y * self.sensitivity
        self.pitch = max(-89, min(89, self.pitch))

    def update_camera_vectors(self):
        """
        Recalculate forward, right and up vectors from yaw and pitch.
        """
        yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)

        self.forward.x = glm.cos(yaw) * glm.cos(pitch)
        self.forward.y = glm.sin(pitch)
        self.forward.z = glm.sin(yaw) * glm.cos(pitch)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def update(self):
        """
        Call each frame to rotate and update view matrix.
        Note: movement is separate; call move() when you want to process keyboard movement.
        """
        self.rotate()
        self.update_camera_vectors()
        self.m_view = glm.lookAt(self.position, self.position + self.forward, self.up)

    # -------------------------
    # Movement
    # -------------------------
    def move(self):
        """
        Keyboard movement in FPS style: WASD for planar movement, Q/E for up/down.
        """
        velocity = self.speed * self.app.delta_time
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.position += self.forward * velocity
        if keys[pygame.K_s]:
            self.position -= self.forward * velocity
        if keys[pygame.K_a]:
            self.position -= self.right * velocity
        if keys[pygame.K_d]:
            self.position += self.right * velocity
        if keys[pygame.K_q]:
            self.position += self.up * velocity
        if keys[pygame.K_e]:
            self.position -= self.up * velocity

    def on_resize(self, width, height):
        """
        Update projection matrix when the window size changes.
        """
        if height == 0:
            height = 1
        self.aspect_ratio = width / height
        self.m_proj = glm.perspective(glm.radians(45), self.aspect_ratio, 0.1, 100.0)


class RTSCamera(Camera3D):
    def __init__(self, app, position=(0, 20, 15), yaw=-90, pitch=-60):
        super().__init__("RTS")
        self.app = app
        self.aspect_ratio = app.WIN_SIZE[0] / app.WIN_SIZE[1]

        self.position = glm.vec3(position)
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, -1)

        self.yaw = yaw
        self.pitch = pitch

        # Projection matrix (perspective)
        self.m_proj = glm.perspective(glm.radians(45), self.aspect_ratio, 0.1, 100)
        self.m_view = glm.mat4()

        # Movement / zoom parameters
        self.speed = 0.04
        self.zoom_speed = 0.5  # Smooth zoom speed for keys
        self.min_height = 5.0
        self.max_height = 50.0

        # When basing RTS on FPS, this offset controls how high above the FPS eye we place the RTS camera
        self._height_offset = 12.0

    # -------------------------
    # Helper to base RTS on FPS
    # -------------------------
    def from_fps(self, fps_camera, height_offset=None):
        """
        Configure the RTS camera so it is centered above the FPS camera position.
        This helps the user not get lost when switching modes.

        - fps_camera: instance of FPSCamera
        - height_offset: optional override for vertical offset above the FPS eye
        """
        if height_offset is not None:
            self._height_offset = float(height_offset)

        # Get FPS state
        fps_pos, fps_yaw, fps_pitch = fps_camera.get_state()

        # Place RTS camera above the FPS x,z coordinates
        target_xz = glm.vec3(fps_pos.x, 0.0, fps_pos.z)

        # Compute desired RTS height clamped to min/max
        desired_y = fps_pos.y + self._height_offset
        desired_y = max(self.min_height, min(self.max_height, desired_y))

        # Move the RTS camera to the same X,Z as the FPS camera and the computed Y
        self.position = glm.vec3(target_xz.x, desired_y, target_xz.z)

        # Align yaw so the view direction feels continuous
        # Keep RTS pitch (top-down) but copy yaw so rotation matches player's facing direction
        self.yaw = fps_yaw
        # Optionally set pitch to a default RTS pitch if you want a consistent top-down angle
        # self.pitch = -60

        # Recompute vectors immediately so the view matrix is valid next frame
        self.update_camera_vectors()
        self.m_view = glm.lookAt(self.position, self.position + self.forward, self.up)

    # -------------------------
    # State API (optional)
    # -------------------------
    def get_state(self):
        return glm.vec3(self.position), float(self.yaw), float(self.pitch)

    def set_state(self, position, yaw, pitch):
        self.position = glm.vec3(position)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.update_camera_vectors()
        self.m_view = glm.lookAt(self.position, self.position + self.forward, self.up)

    # -------------------------
    # Camera math
    # -------------------------
    def update_camera_vectors(self):
        """
        Recalculate the camera's forward, right and up vectors from the current yaw and pitch.
        """
        yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)
        self.forward.x = glm.cos(yaw) * glm.cos(pitch)
        self.forward.y = glm.sin(pitch)
        self.forward.z = glm.sin(yaw) * glm.cos(pitch)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))
    
    def update(self):
        """
        Update camera orientation and position each frame.
        Note: update_camera_vectors is called before move so movement uses current orientation.
        """
        self.update_camera_vectors()
        self.move()
        self.m_view = glm.lookAt(self.position, self.position + self.forward, self.up)

    def move(self):
        """
        Handle keyboard-based panning (WASD) and zooming (Z/X).
        Movement uses a flattened forward vector so panning stays parallel to the ground.
        """
        velocity = self.speed * self.app.delta_time
        keys = pygame.key.get_pressed()

        # PANNING (WASD)
        flat_forward = glm.vec3(self.forward.x, 0, self.forward.z)
        # Guard against degenerate flat_forward when pitch is near +/-90 degrees
        if glm.length(flat_forward) < 1e-6:
            flat_forward = glm.vec3(0, 0, -1)
        else:
            flat_forward = glm.normalize(flat_forward)

        if keys[pygame.K_w]:
            self.position += flat_forward * velocity
        if keys[pygame.K_s]:
            self.position -= flat_forward * velocity
        if keys[pygame.K_a]:
            self.position -= self.right * velocity
        if keys[pygame.K_d]:
            self.position += self.right * velocity

        # # ZOOMING (Z / X) - Replaces Scroll Wheel
        # zoom_val = 0
        # if keys[pygame.K_z]:  # Zoom In
        #     zoom_val = 1
        # if keys[pygame.K_x]:  # Zoom Out
        #     zoom_val = -1

        # if zoom_val != 0:
        #     zoom_amount = zoom_val * self.zoom_speed * self.app.delta_time
        #     new_pos = self.position + (self.forward * zoom_amount)
        #     # Constrain height
        #     if self.min_height < new_pos.y < self.max_height:
        #         self.position = new_pos

    # We removed handle_event since we don't use scroll wheel anymore

    def on_resize(self, width, height):
        """
        Call this when the window is resized to update the projection matrix.
        """
        if height == 0:
            height = 1
        self.aspect_ratio = width / height
        self.m_proj = glm.perspective(glm.radians(45), self.aspect_ratio, 0.1, 100.0)
