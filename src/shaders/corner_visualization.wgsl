@group(0) @binding(0)
var texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<storage, read> corners: array<vec2u>;

@group(0) @binding(2)
var<storage, read> corners_counter: u32;

@group(0) @binding(3)
var gaussian_blur: texture_2d<f32>;

var<private> CORNERS_16: array<vec2i, 16> = array(
    vec2i(-3, 0),
    vec2i(-3, -1),
    vec2i(-2, -2),
    vec2i(-1, -3),
    vec2i(0, -3),
    vec2i(1, -3),
    vec2i(2, -2),
    vec2i(3, -1),
    vec2i(3, 0),
    vec2i(3, 1),
    vec2i(2, 2),
    vec2i(1, 3),
    vec2i(0, 3),
    vec2i(-1, 3),
    vec2i(-2, 2),
    vec2i(-3, 1)
);

@compute
@workgroup_size(64, 1, 1)
fn corner_visualization(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if global_id.x >= corners_counter {
        return;
    }

    let corner = corners[global_id.x];
    let pos = vec2i(corner);

    let gaussian_center = textureLoad(gaussian_blur, pos, 0).x;
    let dx = textureLoad(gaussian_blur, pos + vec2i(0, 2), 0).x - gaussian_center;
    let dy = textureLoad(gaussian_blur, pos + vec2i(2, 0), 0).x - gaussian_center;
    let normed = normalize(vec2f(dx, dy));

    for (var i = 0; i < 16; i ++) {
        let point = pos + CORNERS_16[i];
        let x = point.x;
        let y = i32(textureDimensions(texture).y - 1u) - point.y;

        let dp = dot(normalize(vec2f(CORNERS_16[i])), normed);

        textureStore(texture, vec2i(x, y), vec4f(max(dp, 0.0), 0.0, 0.0, 1.0));
    }
}