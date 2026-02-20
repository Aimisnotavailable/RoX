#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_texcoord;
layout (location = 2) in vec3 in_normal;

out vec2 v_texcoord;
out vec3 v_normal;       // World Normal (For Lighting)
out vec3 v_local_normal; // Local Normal (For Texture Selection) <--- NEW

out vec3 v_frag_pos;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

void main() {
    v_texcoord = in_texcoord;
    
    // Pass the raw normal without rotating it
    v_local_normal = in_normal; 
    
    // Rotate normal for lighting only
    v_normal = mat3(transpose(inverse(m_model))) * in_normal;
    
    v_frag_pos = vec3(m_model * vec4(in_position, 1.0));
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
}