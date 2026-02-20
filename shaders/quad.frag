#version 330 core
out vec4 fragColor;
in vec2 v_texcoord;

uniform sampler2D u_texture;
uniform vec2 u_resolution;
uniform vec2 u_image_res; 
uniform int u_keep_aspect; // <--- MUST BE HERE

void main() {
    vec2 uv = v_texcoord;

    // IF TOGGLE IS ON (1)
    if (u_keep_aspect == 1) {
        float screenAspect = u_resolution.x / u_resolution.y;
        float imageAspect = (u_image_res.x > 0.0) ? (u_image_res.x / u_image_res.y) : 1.0;

        vec2 scale = vec2(1.0);
        if (screenAspect > imageAspect) {
            scale.y = imageAspect / screenAspect;
        } else {
            scale.x = screenAspect / imageAspect;
        }
        
        // Center the scaling
        uv = (v_texcoord - 0.5) * scale + 0.5;
    }

    fragColor = texture(u_texture, uv);
}