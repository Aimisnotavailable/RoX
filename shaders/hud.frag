#version 330 core

in vec2 uv;
out vec4 out_color;

uniform sampler2D u_texture_0;

void main() {
    vec4 color = texture(u_texture_0, uv);
    
    // Discard fully transparent pixels to optimize rendering 
    // and prevent depth/blend interference
    if (color.a < 0.05) {
        discard;
    }
    
    out_color = color;
}