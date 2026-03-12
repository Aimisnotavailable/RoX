#version 330

in vec3 v_normal;
in vec3 v_frag_pos;

uniform vec3 color;
uniform vec3 light_dir;          // main light direction (e.g., from above)
uniform vec3 view_pos;            // camera position
uniform vec3 sky_color = vec3(0.7, 0.8, 1.0);
uniform vec3 ground_color = vec3(0.2, 0.15, 0.1);

out vec4 fragColor;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(light_dir);
    vec3 view_dir = normalize(view_pos - v_frag_pos);

    // Hemisphere ambient
    float hemi_factor = normal.y * 0.5 + 0.5; // map -1..1 to 0..1
    vec3 ambient = mix(ground_color, sky_color, hemi_factor) * 0.5;

    // Diffuse (soft)
    float diff = max(dot(normal, light), 0.0);
    diff = diff * 0.8 + 0.2; // keep some brightness even in shadow

    // Specular (very subtle)
    vec3 reflect_dir = reflect(-light, normal);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 16.0) * 0.2;

    // Rim light (edges glow slightly)
    float rim = 1.0 - max(dot(view_dir, normal), 0.0);
    rim = pow(rim, 3.0) * 0.4;

    vec3 lit = color * (ambient + diff + spec + rim);
    fragColor = vec4(lit, 1.0);
}