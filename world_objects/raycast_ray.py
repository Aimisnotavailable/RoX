# world_objects/raycast_ray.py
import glm
import math
from meshes.cylinder_mesh import CylinderMesh

class RayCastRay:
    """Visual beam from index fingertip to the currently targeted block or object."""
    def __init__(self, engine, hand_label='RIGHT'):
        self.engine = engine
        self.hand_label = hand_label
        self.visible = False
        self.color = glm.vec3(0.0, 1.0, 1.0)  # cyan for voxels
        self.radius = 0.03

        self.mesh = CylinderMesh(engine, radius=self.radius, height=1.0)
        self.start_pos = glm.vec3(0.0)
        self.end_pos = glm.vec3(0.0)

    def update(self):
        ar = self.engine.ar_controller
        if not ar:
            self.visible = False
            return

        if self.hand_label == 'RIGHT':
            landmarks = ar.smooth_right_landmarks
        else:
            landmarks = ar.smooth_left_landmarks

        if not landmarks or len(landmarks) < 9:
            self.visible = False
            return

        tip_norm = landmarks[8]
        cam = self.engine.player
        inv_proj = glm.inverse(cam.m_proj)
        inv_view = glm.inverse(cam.m_view)

        clip = glm.vec4(
            tip_norm.x * 2.0 - 1.0,
            1.0 - tip_norm.y * 2.0,
            -1.0,
            1.0
        )
        eye = inv_proj * clip
        eye_dir = glm.vec3(eye.x, eye.y, -1.0)
        world_dir = glm.normalize(glm.vec3(inv_view * glm.vec4(eye_dir, 0.0)))

        base_distance = 2.0
        depth_range = 4.0
        distance = base_distance + tip_norm.z * depth_range
        self.start_pos = cam.position + world_dir * distance

        vh = self.engine.scene.world_container.voxel_handler
        if ar.is_grabbing:
            self.visible = False
            return

        if vh.object_hit is not None:
            # Object hit
            hit_point = cam.position + world_dir * vh.object_hit_dist
            self.end_pos = hit_point
            self.color = glm.vec3(1.0, 0.5, 0.0)  # orange
            self.visible = True
        elif vh.voxel_world_pos is not None:
            pos = vh.voxel_world_pos
            if vh.is_dragging:
                pos = vh.place_pos
            local_center = glm.vec3(pos) + glm.vec3(0.5)
            self.end_pos = local_center
            self.color = glm.vec3(0.0, 1.0, 1.0)  # cyan
            self.visible = True
        else:
            self.visible = False

    def render(self):
        if not self.visible:
            return

        direction = self.end_pos - self.start_pos
        length = glm.length(direction)
        if length < 1e-6:
            return

        direction = glm.normalize(direction)

        y_axis = glm.vec3(0.0, 1.0, 0.0)
        if abs(glm.dot(direction, y_axis)) > 0.9999:
            rot_mat = glm.mat3(1.0)
        else:
            axis = glm.normalize(glm.cross(y_axis, direction))
            angle = math.acos(glm.dot(y_axis, direction))
            rot_mat = glm.mat3(glm.rotate(angle, axis))

        scale = glm.scale(glm.mat4(1.0), glm.vec3(1.0, length, 1.0))
        mid = (self.start_pos + self.end_pos) * 0.5
        model = glm.translate(glm.mat4(1.0), mid) * glm.mat4(rot_mat) * scale

        prog = self.mesh.program
        prog['m_proj'].write(self.engine.player.m_proj)
        prog['m_view'].write(self.engine.player.m_view)
        prog['view_pos'].write(self.engine.player.position)
        prog['color'].write(self.color)
        prog['m_model'].write(model)
        self.mesh.vao.render()