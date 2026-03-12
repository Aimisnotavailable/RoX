# world_objects/hand_renderer.py
import glm
import moderngl as mgl
from settings import *
from meshes.sphere_mesh import SphereMesh
from meshes.cylinder_mesh import CylinderMesh
from world_objects.hand_shadow import HandShadow   # we'll create this

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm
]

class HandRenderer:
    def __init__(self, engine, hand_label='LEFT'):
        self.engine = engine
        self.hand_label = hand_label
        self.visible = True

        # Meshes with normals for lighting
        self.joint_mesh = SphereMesh(engine, radius=0.03)
        self.bone_mesh = CylinderMesh(engine, radius=0.015, height=1.0)

        # Shadow
        self.shadow = HandShadow(engine)

        self.joint_transforms = []
        self.bone_transforms = []
        self.joint_color = glm.vec3(1.0, 0.8, 0.6)   # skin tone
        self.bone_color = glm.vec3(0.9, 0.7, 0.5)

    def update(self, landmarks_norm):
        """
        landmarks_norm: list of 21 glm.vec3 in normalized camera coordinates (0..1)
        """
        if not landmarks_norm or len(landmarks_norm) < 21:
            self.visible = False
            return
        self.visible = True

        cam = self.engine.player
        inv_proj = glm.inverse(cam.m_proj)
        inv_view = glm.inverse(cam.m_view)

        world_positions = []
        for lm in landmarks_norm:
            # Convert normalized screen coordinates to clip space
            clip = glm.vec4(
                lm.x * 2.0 - 1.0,
                1.0 - lm.y * 2.0,
                -1.0,
                1.0
            )
            eye = inv_proj * clip
            eye_dir = glm.vec3(eye.x, eye.y, -1.0)
            world_dir = glm.normalize(glm.vec3(inv_view * glm.vec4(eye_dir, 0.0)))

            # Map depth to distance
            base_distance = 2.0
            depth_range = 4.0
            distance = base_distance + lm.z * depth_range

            world_pos = cam.position + world_dir * distance
            world_positions.append(world_pos)

        # Update joint transforms
        self.joint_transforms = [glm.translate(glm.mat4(1.0), p) for p in world_positions]

        # Update bone transforms
        self.bone_transforms = []
        for a, b in HAND_CONNECTIONS:
            p1 = world_positions[a]
            p2 = world_positions[b]
            if glm.length(p2 - p1) < 0.001:
                continue
            mid = (p1 + p2) * 0.5
            direction = glm.normalize(p2 - p1)
            length = glm.length(p2 - p1)

            # Build rotation matrix aligning Y with direction
            if abs(direction.y) < 0.999:
                x_axis = glm.normalize(glm.cross(direction, glm.vec3(0, 1, 0)))
            else:
                x_axis = glm.normalize(glm.cross(direction, glm.vec3(0, 0, 1)))
            y_axis = direction
            z_axis = glm.cross(x_axis, y_axis)
            x_axis = glm.cross(y_axis, z_axis)  # re-orthogonalize
            rot_matrix = glm.mat3(x_axis, y_axis, z_axis)

            scale_mat = glm.scale(glm.mat4(1.0), glm.vec3(1.0, length, 1.0))
            model = glm.translate(glm.mat4(1.0), mid) * glm.mat4(rot_matrix) * scale_mat
            self.bone_transforms.append(model)

        # Update shadow position (center of hand)
        hand_center = sum(world_positions, glm.vec3(0)) / len(world_positions)
        self.shadow.update(hand_center)

    def render(self):
        if not self.visible:
            return

        prog = self.joint_mesh.program
        prog['m_proj'].write(self.engine.player.m_proj)
        prog['m_view'].write(self.engine.player.m_view)
        prog['view_pos'].write(self.engine.player.position)

        # Draw joints
        prog['color'].write(self.joint_color)
        for mat in self.joint_transforms:
            prog['m_model'].write(mat)
            self.joint_mesh.vao.render()

        # Draw bones
        prog['color'].write(self.bone_color)
        for mat in self.bone_transforms:
            prog['m_model'].write(mat)
            self.bone_mesh.vao.render()

        # Draw shadow (with blending)
        self.engine.ctx.enable(mgl.BLEND)
        self.shadow.render()
        self.engine.ctx.disable(mgl.BLEND)