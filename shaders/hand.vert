#version 330

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;

out vec3 v_normal;
out vec3 v_frag_pos;

void main() {
    vec4 world_pos = m_model * vec4(in_position, 1.0);
    gl_Position = m_proj * m_view * world_pos;
    v_frag_pos = vec3(world_pos);
    v_normal = mat3(transpose(inverse(m_model))) * in_normal;
}