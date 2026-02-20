#version 330 core

out vec4 fragColor;

in vec2 v_texcoord;
in vec3 v_normal;
in vec3 v_frag_pos;

uniform sampler2D u_texture_0;
uniform vec3 light_pos;
uniform vec3 cam_pos;
uniform vec3 objectColor;
uniform int is_wireframe; 

void main() {
    // If it's a wireframe, just draw the solid color (Green/Red) and exit
    if (is_wireframe == 1) {
        fragColor = vec4(objectColor, 1.0);
        return;
    }

    // Standard Lighting for Solid Blocks
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    float ambientStrength = 0.4;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(v_normal);
    vec3 lightDir = normalize(light_pos - v_frag_pos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 viewDir = normalize(cam_pos - v_frag_pos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = 0.5 * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * objectColor; 
    fragColor = texture(u_texture_0, v_texcoord) * vec4(result, 1.0);
}