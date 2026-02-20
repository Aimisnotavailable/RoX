#version 330 core

out vec4 fragColor;

in vec2 v_texcoord;
in vec3 v_normal;       // Lighting Normal
in vec3 v_local_normal; // Texture Selection Normal <--- NEW
in vec3 v_frag_pos;

uniform sampler2DArray u_texture_array;

uniform int u_layer_side;
uniform int u_layer_top;
uniform int u_layer_bottom;

uniform int is_wireframe;
uniform int is_ghost;
uniform vec3 objectColor;
uniform vec3 light_pos;

void main() {
    if (is_wireframe == 1) {
        fragColor = vec4(objectColor, 1.0);
        return;
    }

    // --- FACE DETECTION LOGIC ---
    int layer = u_layer_side; // Default to Side
    
    // Check LOCAL Y-Normal (Ignores block rotation)
    if (v_local_normal.y > 0.5) {
        layer = u_layer_top;
    } 
    else if (v_local_normal.y < -0.5) {
        layer = u_layer_bottom;
    }

    // --- TEXTURE LOOKUP ---
    // (Keep your existing coordinate logic)
    vec3 uv_layer = vec3(v_texcoord.x, v_texcoord.y, float(layer));
    
    vec4 texColor = texture(u_texture_array, uv_layer);

    // --- GHOST & LIGHTING ---
    if (is_ghost == 1) {
        fragColor = vec4(texColor.rgb, 0.5);
        return;
    }

    // Lighting uses v_normal (World Space) so shadows look correct
    float ambient = 0.5;
    vec3 norm = normalize(v_normal); 
    vec3 lightDir = normalize(light_pos - v_frag_pos);
    float diff = max(dot(norm, lightDir), 0.0);
    
    vec3 result = (ambient + diff) * objectColor * texColor.rgb;
    fragColor = vec4(result, 1.0);
}