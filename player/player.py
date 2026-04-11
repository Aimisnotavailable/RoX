import pygame as pg
import glm
from camera.camera import FPSCamera, RTSCamera
from settings import *

class Player:
    def __init__(self, engine, position=PLAYER_POS, yaw=-90, pitch=0):
        self.engine = engine
        self.p_speed = PLAYER_SPEED
        
        # Initialize both cameras
        self.fps_camera = FPSCamera()
        self.rts_camera = RTSCamera()
        
        # Determine active mode ("FPS" or "RTS")
        self.mode = "FPS"
        self.active_camera = self.fps_camera

        self.gen = 0

# --- MATRICES ---
    @property
    def m_view(self):
        return self.active_camera.m_view
        
    @property
    def m_proj(self):
        return self.active_camera.m_proj

    # --- POSITION ---
    @property
    def position(self):
        return self.active_camera.position

    @position.setter
    def position(self, value):
        self.active_camera.position = value

    # --- VECTORS ---
    @property
    def forward(self):
        return self.active_camera.forward
    
    @property
    def right(self):
        return self.active_camera.right

    @property
    def up(self):
        return self.active_camera.up

    # --- ROTATION ---
    @property
    def yaw(self):
        return self.active_camera.yaw

    @yaw.setter
    def yaw(self, value):
        self.active_camera.yaw = value

    @property
    def pitch(self):
        return self.active_camera.pitch

    @pitch.setter
    def pitch(self, value):
        self.active_camera.pitch = value

    # --- UTILITY ---
    @property
    def frustum(self):
        return self.active_camera.frustum

    @property
    def type(self):
        return self.active_camera.type

        
    def update(self):
        self.keyboard_control()
        if self.mode == "FPS":
            self.mouse_control()

        # Update whichever camera is active
        self.active_camera.update()
        
        # --- NEW: UPDATE FRUSTUM ---
        # Get inverse matrix and scale, then pass them to the active frustum
        inv_model = glm.inverse(self.engine.scene.world_container.local_worlds[0].m_model)
        self.active_camera.frustum.update(inv_model)

    def handle_event(self, event):
        # Toggle camera mode
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_TAB:
                self.switch_camera_mode()
            elif event.key == pygame.K_r:
                self.gen = (self.gen + 1) % len(WORLD_GEN_PARAMS)
                gtype = list(WORLD_GEN_PARAMS)[self.gen]
                self.engine.scene.world.regenerate_world(gtype, **WORLD_GEN_PARAMS[gtype])
            elif event.key == pygame.K_g:
                self.engine.scene.world_container.local_worlds[0].position[0] += 1
                
        # Voxel handling (Clicks)
        if event.type == pg.MOUSEBUTTONDOWN:
            voxel_handler = self.engine.scene.world_container.voxel_handler
            if event.button == 1:
                voxel_handler.set_voxel()
            if event.button == 3:
                voxel_handler.switch_mode()
        

    def switch_camera_mode(self):
        if self.mode == "FPS":
            self.mode = "RTS"
            self.active_camera = self.rts_camera
            
            # Snap RTS camera to "ghost overview" above the old FPS position
            new_rts_pos = glm.vec3(self.fps_camera.position)
            new_rts_pos.y += self.rts_camera._height_offset
            
            self.rts_camera.position = new_rts_pos
            self.rts_camera.yaw = self.fps_camera.yaw
            self.rts_camera.pitch = glm.radians(-60) # Look downward at the world
            
            # Free the mouse for RTS clicking if needed
            pg.event.set_grab(False)
            pg.mouse.set_visible(True)
            
        else:
            self.mode = "FPS"
            self.active_camera = self.fps_camera
            
            # Lock the mouse back to the center for FPS
            pg.event.set_grab(True)
            pg.mouse.set_visible(False)

    def mouse_control(self):
        mouse_dx, mouse_dy = pg.mouse.get_rel()
        if mouse_dx:
            self.fps_camera.rotate_yaw(delta_x=mouse_dx * MOUSE_SENSITIVITY)
        if mouse_dy:
            self.fps_camera.rotate_pitch(delta_y=mouse_dy * MOUSE_SENSITIVITY)

    def keyboard_control(self):
        key_state = pg.key.get_pressed()
        self.p_speed = min(MAX_PLAYER_SPEED, self.p_speed + 0.001)
        vel = self.p_speed * self.engine.delta_time

        # Movement affects the active camera
        if key_state[pg.K_w]:
            self.active_camera.move_forward(vel)
        if key_state[pg.K_s]:
            self.active_camera.move_back(vel)
        if key_state[pg.K_d]:
            self.active_camera.move_right(vel)
        if key_state[pg.K_a]:
            self.active_camera.move_left(vel)
        if self.mode == "FPS":
            if key_state[pg.K_q]:
                self.active_camera.move_up(vel)
            if key_state[pg.K_e]:
                self.active_camera.move_down(vel)
        
        # Transform selected object with keyboard (if any)
        selected = self.engine.scene.world_container.selected_object
        if selected is not None:
            scale_speed = 2.0 * self.engine.delta_time * 0.001
            if key_state[pg.K_x]: 
                selected.scale += glm.vec3(scale_speed)
            if key_state[pg.K_z]: 
                selected.scale -= glm.vec3(scale_speed)
            selected.scale = glm.clamp(selected.scale, 0.1, 5.0)

            rotation_speed = 2.0 * self.engine.delta_time * 0.001
            euler = glm.eulerAngles(selected.rotation)
            if key_state[pg.K_LEFT]: 
                euler.y -= rotation_speed
            if key_state[pg.K_RIGHT]: 
                euler.y += rotation_speed
            if key_state[pg.K_UP]: 
                euler.x -= rotation_speed
            if key_state[pg.K_DOWN]: 
                euler.x += rotation_speed
            selected.rotation = glm.quat(euler)
        else:
            # Fallback to first world (legacy)
            world = self.engine.scene.world_container.local_worlds[0]
            scale_speed = 2.0 * self.engine.delta_time * 0.001
            if key_state[pg.K_x]: 
                world.world_scale += scale_speed
            if key_state[pg.K_z]: 
                world.world_scale -= scale_speed
            world.world_scale = glm.clamp(world.world_scale, 0.1, 5.0)
            rotation_speed = 2.0 * self.engine.delta_time * 0.001
            if key_state[pg.K_LEFT]: 
                world.world_yaw -= rotation_speed
            if key_state[pg.K_RIGHT]: 
                world.world_yaw += rotation_speed
            if key_state[pg.K_UP]: 
                world.world_pitch -= rotation_speed
            if key_state[pg.K_DOWN]: 
                world.world_pitch += rotation_speed

        if not (key_state[pg.K_w] or key_state[pg.K_s] or
                key_state[pg.K_a] or key_state[pg.K_d] or
                key_state[pg.K_q] or key_state[pg.K_e]):
            self.p_speed = PLAYER_SPEED