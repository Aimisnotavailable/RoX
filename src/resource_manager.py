import moderngl
from PIL import Image

class ResourceManager:
    def __init__(self, ctx):
        self.ctx = ctx

    def load_program(self, path):
        with open(f'{path}.vert') as f: vertex_src = f.read()
        with open(f'{path}.frag') as f: fragment_src = f.read()
        return self.ctx.program(vertex_shader=vertex_src, fragment_shader=fragment_src)

    def load_texture(self, path):
        try:
            img = Image.open(path).convert('RGBA')
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            texture = self.ctx.texture(img.size, 4, img.tobytes())
            texture.build_mipmaps()
            return texture
        except FileNotFoundError:
            print(f"Error: {path} not found.")
            # Return a tiny magenta texture as a fallback so the game doesn't crash
            return self.ctx.texture((2, 2), 4, b'\xff\x00\xff\xff' * 4)
    
    def load_texture_array(self, path):
        try:
            img = Image.open(path).convert('RGBA')
            tile_size = 512 
            cols = img.width // tile_size
            rows = img.height // tile_size
            tile_data = bytearray()
            
            for y in range(rows):
                for x in range(cols):
                    left = x * tile_size
                    upper = y * tile_size
                    right = left + tile_size
                    lower = upper + tile_size
                    tile = img.crop((left, upper, right, lower))
                    tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
                    tile_data.extend(tile.tobytes())
                    
            num_layers = rows * cols
            texture_array = self.ctx.texture_array((tile_size, tile_size, num_layers), 4, tile_data)
            texture_array.build_mipmaps()
            texture_array.filter = (moderngl.NEAREST_MIPMAP_LINEAR, moderngl.NEAREST)
            return texture_array
        except Exception as e:
            print(f"Error loading texture array: {e}")
            return None