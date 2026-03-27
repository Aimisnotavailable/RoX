# world_objects/raycast_ray.py
import glm
import math
from meshes.cylinder_mesh import CylinderMesh

# Fix visual beam lock
# Fix raycast method
# reuse raycast method from voxel handler
class RayCastRay:
    """Visual beam from index fingertip to the currently targeted block,
       accounting for world rotation and scaling."""
    def __init__(self, engine, hand_label='RIGHT'):
        self.engine = engine
        self.hand_label = hand_label
        self.visible = False
        self.color = glm.vec3(0.0, 1.0, 1.0)  # cyan
        self.radius = 0.03

        # Create a cylinder mesh (height 1, will be scaled)
        self.mesh = CylinderMesh(engine, radius=self.radius, height=1.0)
        self.start_pos = glm.vec3(0.0)
        self.end_pos = glm.vec3(0.0)

    def update(self):
        """Compute world positions of fingertip and hit block, applying world transform to the block."""
        ar = self.engine.ar_controller
        if not ar:
            self.visible = False
            return

        # Get fingertip normalized coordinates (index tip = landmark 8)
        if self.hand_label == 'RIGHT':
            landmarks = ar.smooth_right_landmarks
        else:
            landmarks = ar.smooth_left_landmarks

        if not landmarks or len(landmarks) < 9:
            self.visible = False
            return

        tip_norm = landmarks[8]   # glm.vec3 with x,y,z in [0,1]

        # Convert to world position using the same method as HandRenderer
        cam = self.engine.player
        inv_proj = glm.inverse(cam.m_proj)
        inv_view = glm.inverse(cam.m_view)

        # Clip coordinates
        clip = glm.vec4(
            tip_norm.x * 2.0 - 1.0,
            1.0 - tip_norm.y * 2.0,
            -1.0,
            1.0
        )
        eye = inv_proj * clip
        eye_dir = glm.vec3(eye.x, eye.y, -1.0)
        world_dir = glm.normalize(glm.vec3(inv_view * glm.vec4(eye_dir, 0.0)))

        # Map depth to distance (same as HandRenderer)
        base_distance = 2.0
        depth_range = 4.0
        distance = base_distance + tip_norm.z * depth_range

        self.start_pos = cam.position + world_dir * distance

        # Get the targeted block world position from the voxel handler
        vh = self.engine.scene.world.voxel_handler
        if vh.voxel_world_pos is not None:
            # Block's local lower corner, convert to center in local coordinates
            local_center = glm.vec3(vh.voxel_world_pos) + glm.vec3(0.5)
            # Apply world transform (rotation + scale) to get global position
            world = self.engine.scene.world
            global_center = world.m_model * glm.vec4(local_center, 1.0)
            self.end_pos = glm.vec3(global_center)
            self.visible = True
        else:
            self.visible = False

    def render(self):
        if not self.visible:
            return

        # Compute direction and length
        direction = self.end_pos - self.start_pos
        length = glm.length(direction)
        if length < 1e-6:
            return

        direction = glm.normalize(direction)

        # Build model matrix for the cylinder (aligned with Y axis)
        # CylinderMesh expects its own Y axis to be the length direction,
        # so we rotate the standard Y‑aligned cylinder to point along 'direction'.
        y_axis = glm.vec3(0.0, 1.0, 0.0)
        if abs(glm.dot(direction, y_axis)) > 0.9999:
            # nearly parallel, use identity rotation
            rot_mat = glm.mat3(1.0)
        else:
            axis = glm.normalize(glm.cross(y_axis, direction))
            angle = math.acos(glm.dot(y_axis, direction))
            rot_mat = glm.mat3(glm.rotate(angle, axis))

        # Scale: cylinder height = length, radius already set
        scale = glm.scale(glm.mat4(1.0), glm.vec3(1.0, length, 1.0))

        # Translate to midpoint
        mid = (self.start_pos + self.end_pos) * 0.5
        model = glm.translate(glm.mat4(1.0), mid) * glm.mat4(rot_mat) * scale

        prog = self.mesh.program
        prog['m_proj'].write(self.engine.player.m_proj)
        prog['m_view'].write(self.engine.player.m_view)
        prog['view_pos'].write(self.engine.player.position)
        prog['color'].write(self.color)

        prog['m_model'].write(model)
        self.mesh.vao.render()