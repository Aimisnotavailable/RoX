from settings import *
import glm

class Frustum:
    def __init__(self, camera):
        self.cam = camera

        self.factor_y = 1.0 / math.cos(half_y := V_FOV * 0.5)
        self.tan_y = math.tan(half_y)

        self.factor_x = 1.0 / math.cos(half_x := H_FOV * 0.5)
        self.tan_x = math.tan(half_x)

        # New variables to hold local-space camera data
        self.local_pos = glm.vec3()
        self.local_forward = glm.vec3()
        self.local_up = glm.vec3()
        self.local_right = glm.vec3()
        self.local_near = NEAR
        self.local_far = FAR

    def update(self, inv_model, world_scale = 1):
        # 1. Transform Camera Position to Local Space (w=1.0 for points)
        self.local_pos = glm.vec3(inv_model * glm.vec4(self.cam.position, 1.0))

        # 2. Transform Camera Vectors to Local Space (w=0.0 for directions)
        self.local_forward = glm.normalize(glm.vec3(inv_model * glm.vec4(self.cam.forward, 0.0)))
        self.local_up = glm.normalize(glm.vec3(inv_model * glm.vec4(self.cam.up, 0.0)))
        self.local_right = glm.normalize(glm.vec3(inv_model * glm.vec4(self.cam.right, 0.0)))

        # 3. Adjust Near/Far planes based on zoom scale!
        self.local_near = NEAR / world_scale
        self.local_far = FAR / world_scale
        
    def is_on_frustum(self, chunk):
        # We now use the LOCAL camera position and vectors!
        sphere_vec = chunk.center - self.local_pos

        # outside the NEAR and FAR planes?
        sz = glm.dot(sphere_vec, self.local_forward)
        if not (self.local_near - CHUNK_SPHERE_RADIUS <= sz <= self.local_far + CHUNK_SPHERE_RADIUS):
            return False

        # outside the TOP and BOTTOM planes?
        sy = glm.dot(sphere_vec, self.local_up)
        dist = self.factor_y * CHUNK_SPHERE_RADIUS + sz * self.tan_y
        if not (-dist <= sy <= dist):
            return False

        # outside the LEFT and RIGHT planes?
        sx = glm.dot(sphere_vec, self.local_right)
        dist = self.factor_x * CHUNK_SPHERE_RADIUS + sz * self.tan_x
        if not (-dist <= sx <= dist):
            return False

        return True