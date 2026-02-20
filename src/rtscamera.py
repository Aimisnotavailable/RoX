# rtscamera.py
import glm
import pygame

class RTSCamera:
    def __init__(self, app, position=(0, 20, 15), yaw=-90, pitch=-60):
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
