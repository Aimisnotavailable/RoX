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
        # ... (Vertices same as before) ...
        vertices = [
            -0.5, -0.5, 0.5, 0, 0, 0, 0, 1,
             0.5, -0.5, 0.5, 1, 0, 0, 0, 1,
             0.5,  0.5, 0.5, 1, 1, 0, 0, 1,
            -0.5,  0.5, 0.5, 0, 1, 0, 0, 1,
            -0.5, -0.5, -0.5, 1, 0, 0, 0, -1,
             0.5, -0.5, -0.5, 0, 0, 0, 0, -1,
             0.5,  0.5, -0.5, 0, 1, 0, 0, -1,
            -0.5,  0.5, -0.5, 1, 1, 0, 0, -1,
             0.5, -0.5, 0.5, 0, 0, 1, 0, 0,
             0.5, -0.5, -0.5, 1, 0, 1, 0, 0,
             0.5,  0.5, -0.5, 1, 1, 1, 0, 0,
             0.5,  0.5, 0.5, 0, 1, 1, 0, 0,
            -0.5, -0.5, 0.5, 1, 0, -1, 0, 0,
            -0.5, -0.5, -0.5, 0, 0, -1, 0, 0,
            -0.5,  0.5, -0.5, 0, 1, -1, 0, 0,
            -0.5,  0.5, 0.5, 1, 1, -1, 0, 0,
            -0.5,  0.5, 0.5, 0, 0, 0, 1, 0,
             0.5,  0.5, 0.5, 1, 0, 0, 1, 0,
             0.5,  0.5, -0.5, 1, 1, 0, 1, 0,
            -0.5,  0.5, -0.5, 0, 1, 0, 1, 0,
            -0.5, -0.5, 0.5, 0, 1, 0, -1, 0,
             0.5, -0.5, 0.5, 1, 1, 0, -1, 0,
             0.5, -0.5, -0.5, 1, 0, 0, -1, 0,
            -0.5, -0.5, -0.5, 0, 0, 0, -1, 0,
        ]
        
        # Triangle Indices (For solid blocks)
        indices = [
            0, 1, 2, 2, 3, 0, 5, 4, 7, 7, 6, 5, 8, 9, 10, 10, 11, 8,
            13, 12, 15, 15, 14, 13, 16, 17, 18, 18, 19, 16, 21, 20, 23, 23, 22, 21
        ]

        # Line Indices (For the wireframe outline)
        # We manually define the 12 edges of a cube.
        # We use the Front Face (0,1,2,3) and Back Face (4,5,6,7) vertices.
        lines = [
            0, 1, 1, 2, 2, 3, 3, 0, # Front Face Loop
            4, 5, 5, 6, 6, 7, 7, 4, # Back Face Loop
            0, 3, 1, 2, # Connect front to back (Top/Bottom edges)... wait, the indices above are messy.
            # Let's just hardcode the connections based on visual positions.
            # Bottom Square: 0-1, 1-5, 5-4, 4-0 (Using raw positions might be safer but let's try indices)
            
            # Since our vertex array repeats vertices for normals, we just need to pick *any* vertex
            # at the corners.
            # Corners: 
            # Front-Bottom-Left: 0, Front-Bottom-Right: 1, Front-Top-Right: 2, Front-Top-Left: 3
            # Back-Bottom-Left: 4, Back-Bottom-Right: 5, Back-Top-Right: 6, Back-Top-Left: 7 
            # (Note: index 4,5,6,7 mapping depends on previous list, let's just trace the unique corners)
            
            0, 1,  1, 2,  2, 3,  3, 0, # Front Square
            4, 5,  5, 6,  6, 7,  7, 4, # Back Square
            0, 4,  1, 5,  2, 6,  3, 7  # Connecting Edges
        ]
        
        # Note: Because our vertex list has 24 vertices (duplicates for normals), 
        # indices 0-7 above only cover the "Front" and "Back" faces defined in the array.
        # This is actually fine! Those 8 vertices cover all 8 corners of the cube spatially.
        
        return np.array(vertices, dtype='f4'), np.array(indices, dtype='i4'), np.array(lines, dtype='i4')

    def render(self):
        # Render Triangles (Solid)
        self.vao.render()
        
    def render_lines(self):
        # Render Lines (Wireframe) - mode=moderngl.LINES
        self.vao_lines.render(mode=moderngl.LINES)