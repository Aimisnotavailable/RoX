# fpscamera.py
import glm
import pygame

class FPSCamera:
    def __init__(self, app, position=(0, 0, 4), yaw=-90, pitch=0):
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
