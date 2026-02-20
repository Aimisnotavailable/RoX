#version 330 core

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_texcoord;
layout (location = 2) in vec3 in_normal;
layout (location = 3) in vec3 in_barycentric; // NEW INPUT

out vec2 v_texcoord;
out vec3 v_normal;
out vec3 v_frag_pos;
out vec3 v_bary; // PASS TO FRAGMENT

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

void main() {
    v_texcoord = in_texcoord;
    v_normal = mat3(transpose(inverse(m_model))) * in_normal;
    v_frag_pos = vec3(m_model * vec4(in_position, 1.0));
    v_bary = in_barycentric; // Pass it through
    
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
}