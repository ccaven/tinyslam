@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<storage, read> latest_corners: array<vec2u>;

@group(0) @binding(2)
var<storage, read> latest_corners_counter: u32;

@group(0) @binding(3)
var<storage, read> previous_corners: array<vec2u>;

@group(0) @binding(4)
var<storage, read> previous_corners_counter: u32;

@group(0) @binding(5)
var<storage, read> feature_matches: array<vec2u>;

@group(0) @binding(6)
var<storage, read> feature_matches_counter: u32;

@compute
@workgroup_size(64, 1, 1)
fn matches_visualization(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if global_id.x >= feature_matches_counter {
        return;
    }

    let feature_match = feature_matches[global_id.x];
    let corner_a = latest_corners[feature_match.x];
    let corner_b = previous_corners[feature_match.y];

    // Rasterize line between a and b
    for (var l = 0u; l < 50u; l ++) {
        let px = corner_a.x + (corner_b.x * l - corner_a.x * l) / 50u;
        let py = corner_a.y + (corner_b.y * l - corner_a.y * l) / 50u;

        textureStore(texture, vec2u(px, textureDimensions(texture).y - 1u - py), vec4f(0.0, 0.0, 1.0, 1.0));
    }
}