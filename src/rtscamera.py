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
        
        self.m_proj = glm.perspective(glm.radians(45), self.aspect_ratio, 0.1, 100)
        self.m_view = glm.mat4()
        
        self.speed = 0.02
        self.zoom_speed = 0.5 # Smooth zoom speed for keys
        self.min_height = 5.0
        self.max_height = 50.0

    def update_camera_vectors(self):
        yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)
        self.forward.x = glm.cos(yaw) * glm.cos(pitch)
        self.forward.y = glm.sin(pitch)
        self.forward.z = glm.sin(yaw) * glm.cos(pitch)

        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def update(self):
        self.move() 
        self.update_camera_vectors()
        self.m_view = glm.lookAt(self.position, self.position + self.forward, self.up)

    def move(self):
        velocity = self.speed * self.app.delta_time
        keys = pygame.key.get_pressed()
        
        # PANNING (WASD)
        flat_forward = glm.normalize(glm.vec3(self.forward.x, 0, self.forward.z))
        
        if keys[pygame.K_w]: self.position += flat_forward * velocity
        if keys[pygame.K_s]: self.position -= flat_forward * velocity
        if keys[pygame.K_a]: self.position -= self.right * velocity
        if keys[pygame.K_d]: self.position += self.right * velocity

        # ZOOMING (Z / X) - Replaces Scroll Wheel
        zoom_val = 0
        if keys[pygame.K_z]: # Zoom In
            zoom_val = 1
        if keys[pygame.K_x]: # Zoom Out
            zoom_val = -1
            
        if zoom_val != 0:
            zoom_amount = zoom_val * self.zoom_speed
            new_pos = self.position + (self.forward * zoom_amount)
            # Constrain height
            if self.min_height < new_pos.y < self.max_height:
                self.position = new_pos
    
    # We removed handle_event since we don't use scroll wheel anymore