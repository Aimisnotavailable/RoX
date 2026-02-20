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
        
        self.m_proj = glm.perspective(glm.radians(45), self.aspect_ratio, 0.1, 100)
        self.m_view = glm.mat4()
        
        self.speed = 0.005
        self.sensitivity = 0.1

    def rotate(self):
        rel_x, rel_y = pygame.mouse.get_rel()
        self.yaw += rel_x * self.sensitivity
        self.pitch -= rel_y * self.sensitivity
        self.pitch = max(-89, min(89, self.pitch))

    def update_camera_vectors(self):
        yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)
        
        self.forward.x = glm.cos(yaw) * glm.cos(pitch)
        self.forward.y = glm.sin(pitch)
        self.forward.z = glm.sin(yaw) * glm.cos(pitch)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def update(self):
        self.rotate()
        self.update_camera_vectors()
        self.m_view = glm.lookAt(self.position, self.position + self.forward, self.up)

    def move(self):
        velocity = self.speed * self.app.delta_time
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: self.position += self.forward * velocity
        if keys[pygame.K_s]: self.position -= self.forward * velocity
        if keys[pygame.K_a]: self.position -= self.right * velocity
        if keys[pygame.K_d]: self.position += self.right * velocity
        if keys[pygame.K_q]: self.position += self.up * velocity
        if keys[pygame.K_e]: self.position -= self.up * velocity