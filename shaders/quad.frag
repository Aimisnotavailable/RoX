#version 330 core
out vec4 fragColor;
in vec2 v_texcoord;

uniform sampler2D u_texture;
uniform vec2 u_resolution; // The window size (e.g., 1280x720)
uniform vec2 u_image_res;  // The texture size (e.g., 1920x1080)

void main() {
    // 1. Calculate Aspect Ratios
    float screenAspect = u_resolution.x / u_resolution.y;
    // If image res is 0 (safety check), default to 1
    float imageAspect = (u_image_res.x > 0.0) ? (u_image_res.x / u_image_res.y) : 1.0;

    // 2. Calculate "Cover" Scale
    // We want the image to cover the screen entirely without stretching.
    vec2 scale = vec2(1.0);
    
    if (screenAspect > imageAspect) {
        // Screen is wider than image: Stretch width to fit, crop top/bottom
        scale.y = imageAspect / screenAspect;
    } else {
        // Screen is taller than image: Stretch height to fit, crop sides
        scale.x = screenAspect / imageAspect;
    }

    // 3. Center and Apply Scale
    // (v_texcoord - 0.5) moves origin to center
    // * scale applies the correction
    // + 0.5 moves origin back to corner
    vec2 uv = (v_texcoord - 0.5) * scale + 0.5;

    // 4. Render
    // Optional: If you zoom out too far, you might see clamp edges. 
    // "Cover" logic usually prevents this, but let's just sample.
    fragColor = texture(u_texture, uv);
}