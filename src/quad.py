import numpy as np
import moderngl

class ScreenQuad:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        
        # 4 vertices (x, y) covering the screen + (u, v) texture coords
        # (-1, -1) is bottom-left, (1, 1) is top-right
        vertices = [
            # x, y,   u, v
            -1.0,  1.0, 0.0, 1.0, # Top-left
            -1.0, -1.0, 0.0, 0.0, # Bottom-left
             1.0, -1.0, 1.0, 0.0, # Bottom-right
            
            -1.0,  1.0, 0.0, 1.0, # Top-left
             1.0, -1.0, 1.0, 0.0, # Bottom-right
             1.0,  1.0, 1.0, 1.0, # Top-right
        ]
        
        vertices = np.array(vertices, dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
        
        self.program = self.app.load_program('shaders/quad')
        
        # Only position (2f) and texcoord (2f)
        content = [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')]
        self.vao = self.ctx.vertex_array(self.program, content)

    def render(self, texture, image_size=None, keep_aspect=False):
        texture.use(location=0)
        
        # 1. Send Screen Resolution (Required for aspect math)
        self.program['u_resolution'].value = self.app.WIN_SIZE
        
        # 2. Send Image Resolution
        if image_size:
            self.program['u_image_res'].value = image_size
        else:
            # If no size given, assume full screen (1:1 with window)
            self.program['u_image_res'].value = self.app.WIN_SIZE

        # 3. THE MISSING PIECE: Send the Toggle State!
        # We convert the boolean (True/False) to an Integer (1/0)
        if 'u_keep_aspect' in self.program:
            self.program['u_keep_aspect'].value = 1 if keep_aspect else 0
            
        self.vao.render()