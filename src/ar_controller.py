class ARController:
    def __init__(self, engine):
        self.engine = engine

    def update_trails(self, l_pos, r_pos):
        if l_pos:
            self.engine.trails['left'].append(l_pos)
            if len(self.engine.trails['left']) > 10: self.engine.trails['left'].pop(0)
        else:
            self.engine.trails['left'].clear()
        
        if r_pos:
            self.engine.trails['right'].append(r_pos)
            if len(self.engine.trails['right']) > 10: self.engine.trails['right'].pop(0)
        else:
            self.engine.trails['right'].clear()

    def process_hand_inputs(self):
        # We access the input_handler (which is ARInputHandler in the engine)
        l_state = self.engine.input_handler.left_state
        r_state = self.engine.input_handler.right_state
        l_pos = self.engine.input_handler.left_finger_ema
        r_pos = self.engine.input_handler.right_finger_ema
        
        self.update_trails(l_pos, r_pos)
        self.engine.ar_cursor_pos = r_pos if r_pos else None
        self.engine.current_action_label = "" 
        
        # Reset visual flags
        self.engine.zoom_line_visible = False

        # ZOOM
        if l_state.active and r_state.active and l_pos and r_pos:
            self.engine.current_action_label = "ZOOMING"
            dx = l_pos[0] - r_pos[0]
            dy = l_pos[1] - r_pos[1]
            current_dist = (dx**2 + dy**2)**0.5
            
            # Store coords for rendering in render_hud()
            self.engine.zoom_line_visible = True
            self.engine.zoom_line_coords = (l_pos, r_pos)
            self.engine.zoom_pct_display = int(current_dist / 5)

            if self.engine.last_pinch_dist is not None:
                delta = current_dist - self.engine.last_pinch_dist
                self.engine.camera.position += self.engine.camera.forward * (delta * 0.05)
                
            self.engine.last_pinch_dist = current_dist
            self.engine.last_rotate_pos = None
            self.engine.clicking = False
            self.engine.builder.stop_raycast = False
            return
        else:
            self.engine.last_pinch_dist = None

        # ROTATE
        if l_state.active and l_pos:
            self.engine.current_action_label = "ROTATING"
            if self.engine.last_rotate_pos is None:
                self.engine.last_rotate_pos = l_pos
            
            dx = l_pos[0] - self.engine.last_rotate_pos[0]
            dy = l_pos[1] - self.engine.last_rotate_pos[1]
            
            rot_speed = 0.005
            self.engine.builder.rotation.y += dx * rot_speed
            self.engine.builder.rotation.x += dy * rot_speed
            
            self.engine.last_rotate_pos = l_pos
        else:
            self.engine.last_rotate_pos = None

        # BUILD
        if r_state.active and r_pos:
            self.engine.current_action_label = "BUILDING"
            self.engine.clicking = True
            self.engine.builder.stop_raycast = True 
            
            if self.engine.last_build_hand_pos is None:
                self.engine.last_build_hand_pos = r_pos
                rel_x, rel_y = 0, 0
            else:
                rel_x = r_pos[0] - self.engine.last_build_hand_pos[0]
                rel_y = r_pos[1] - self.engine.last_build_hand_pos[1]
            
            self.engine.camera.movement_rel = (rel_x, rel_y)
            self.engine.last_build_hand_pos = r_pos
        else:
            self.engine.clicking = False
            self.engine.builder.stop_raycast = False
            self.engine.last_build_hand_pos = None