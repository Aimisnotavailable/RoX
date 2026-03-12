#version 330
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec2 in_uv;
out vec2 v_uv;
uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model;
void main() {
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
    v_uv = in_uv;
}