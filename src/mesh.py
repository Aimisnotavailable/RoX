import numpy as np
import moderngl

class CubeMesh:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx
        self.program = app.prog
        
        # 1. Get Data
        vertices, indices, lines = self.get_data()
        
        # 2. Create Buffers
        self.vbo = self.ctx.buffer(vertices.astype('f4').tobytes())
        self.ibo = self.ctx.buffer(indices.astype('i4').tobytes())
        self.line_ibo = self.ctx.buffer(lines.astype('i4').tobytes()) # <--- NEW: Line Index Buffer
        
        # 3. VAO Content
        # We don't need barycentric anymore, just pos, uv, norm
        content = [(self.vbo, '3f 2f 3f', 'in_position', 'in_texcoord', 'in_normal')]
        
        # 4. Create Two VAOs: One for Solid blocks, one for Lines
        self.vao = self.ctx.vertex_array(self.program, content, index_buffer=self.ibo)
        self.vao_lines = self.ctx.vertex_array(self.program, content, index_buffer=self.line_ibo) # <--- NEW

    def get_data(self):
        # Format: x, y, z,   u, v,   nx, ny, nz
        vertices = [
            # Front Face (z=0.5)
            -0.5, -0.5,  0.5,   0, 0,   0, 0, 1,
             0.5, -0.5,  0.5,   1, 0,   0, 0, 1,
             0.5,  0.5,  0.5,   1, 1,   0, 0, 1,
            -0.5,  0.5,  0.5,   0, 1,   0, 0, 1,

            # Back Face (z=-0.5)
             0.5, -0.5, -0.5,   0, 0,   0, 0,-1,
            -0.5, -0.5, -0.5,   1, 0,   0, 0,-1,
            -0.5,  0.5, -0.5,   1, 1,   0, 0,-1,
             0.5,  0.5, -0.5,   0, 1,   0, 0,-1,

            # Left Face (x=-0.5)
            -0.5, -0.5, -0.5,   0, 0,  -1, 0, 0,
            -0.5, -0.5,  0.5,   1, 0,  -1, 0, 0,
            -0.5,  0.5,  0.5,   1, 1,  -1, 0, 0,
            -0.5,  0.5, -0.5,   0, 1,  -1, 0, 0,

            # Right Face (x=0.5)
             0.5, -0.5,  0.5,   0, 0,   1, 0, 0,
             0.5, -0.5, -0.5,   1, 0,   1, 0, 0,
             0.5,  0.5, -0.5,   1, 1,   1, 0, 0,
             0.5,  0.5,  0.5,   0, 1,   1, 0, 0,

            # Top Face (y=0.5)
            -0.5,  0.5,  0.5,   0, 0,   0, 1, 0,
             0.5,  0.5,  0.5,   1, 0,   0, 1, 0,
             0.5,  0.5, -0.5,   1, 1,   0, 1, 0,
            -0.5,  0.5, -0.5,   0, 1,   0, 1, 0,

            # Bottom Face (y=-0.5)
            -0.5, -0.5, -0.5,   0, 0,   0,-1, 0,
             0.5, -0.5, -0.5,   1, 0,   0,-1, 0,
             0.5, -0.5,  0.5,   1, 1,   0,-1, 0,
            -0.5, -0.5,  0.5,   0, 1,   0,-1, 0,
        ]
        
        # Correct Counter-Clockwise Indices
        indices = [
             0, 1, 2,  2, 3, 0,  # Front
             4, 5, 6,  6, 7, 4,  # Back
             8, 9,10, 10,11, 8,  # Left
            12,13,14, 14,15,12,  # Right
            16,17,18, 18,19,16,  # Top
            20,21,22, 22,23,20   # Bottom
        ]
        
        # Simple Box Outline (Wireframe)
        lines = [
             0, 1,  1, 2,  2, 3,  3, 0, # Front
             4, 5,  5, 6,  6, 7,  7, 4, # Back
             0, 11, 1, 12, 2, 19, 3, 8  # Connecting (Indices mapped to corners roughly)
             # Note: For perfect wireframe, using the corner indices (0,1,2,3 for front) is enough
        ]

        return np.array(vertices, dtype='f4'), np.array(indices, dtype='i4'), np.array(lines, dtype='i4')

    def render(self):
        # Render Triangles (Solid)
        self.vao.render()
        
    def render_lines(self):
        # Render Lines (Wireframe) - mode=moderngl.LINES
        self.vao_lines.render(mode=moderngl.LINES)