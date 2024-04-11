
struct FeatureData {
    x: u32,
    y: u32
}

struct BriefDescriptor {
    data: array<u32, 8>
}

@group(0) @binding(0)
var<storage, read_write> v_index: atomic<u32>;

@group(0) @binding(1)
var<storage, read_write> v_features: array<FeatureData>;

@group(0) @binding(2)
var<uniform> v_threshold: u32;

@group(1) @binding(0)
var<storage, read> v_image: array<u32>;

@group(1) @binding(1)
var<uniform> image_size: vec2u;

@group(2) @binding(0)
var<storage, read_write> v_descriptors: array<BriefDescriptor>;

fn read_image_intensity(x: i32, y: i32) -> u32 {
    if x < 0 || y < 0 || x >= i32(image_size.x) || y >= i32(image_size.y) {
        return 0u;
    }

    let image_index = u32(y * i32(image_size.x) + x);
    let real_index = image_index >> 2;
    let leftover = image_index & 3;

    let val = (v_image[real_index] >> (8 * leftover)) & 255;
    
    return val;
}

var<private> CORNERS_16: array<vec2i, 16> = array(
    vec2i(-3, 1),
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
    vec2i(-2, 2)
);

@compute
@workgroup_size(16, 16, 1)
fn compute_orb(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    
    if global_id.x < 3 || global_id.y < 3 || global_id.x > image_size.x - 4 || global_id.y > image_size.y - 4 {
        return;
    }    

    let center_value = read_image_intensity(i32(global_id.x), i32(global_id.y));

    var centroid = vec2f(0, 0);
    var is_over: u32 = 0u;
    var is_under: u32 = 0u;

    for (var i = 0u; i < 16u; i ++) {
        let corner_value = read_image_intensity(
            i32(global_id.x) + CORNERS_16[i].x,
            i32(global_id.y) + CORNERS_16[i].y
        );

        let diff = i32(corner_value) - i32(center_value);

        if diff > i32(v_threshold) {
            is_over |= 1u << i;
        } else if diff < -i32(v_threshold) {
            is_under |= 1u << i;
        }
    }

    var over_streak = 0u;
    var max_over_streak = 0u;
    var under_streak = 0u;
    var max_under_streak = 0u;
    for (var i = 0u; i < 24u; i ++) {
        let j = i % 16u;
        if extractBits(is_over, j, 1u) == 1u {
            over_streak ++;
            if over_streak > max_over_streak {
                max_over_streak = over_streak;
            }
        } else {
            over_streak = 0u;
        }
        if extractBits(is_under, j, 1u) == 1u {
            under_streak ++;
            if under_streak > max_under_streak {
                max_under_streak = under_streak;
            }
        } else {
            under_streak = 0u;
        }
    }

    if max_over_streak > 11 || max_under_streak > 11 {
        let index = atomicAdd(&v_index, 1u);

        var data: FeatureData;

        data.x = global_id.x;
        data.y = global_id.y;

        v_features[index] = data;
    }
    
}

@compute
@workgroup_size(64, 1, 1)
fn compute_brief(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if global_id.x > atomicLoad(&v_index) {
        return;
    }

    let feature = v_features[global_id.x];

    let center_value = read_image_intensity(i32(feature.x), i32(feature.y));

    var centroid = vec2f(0, 0);

    for (var i = 0u; i < 16u; i ++) {
        centroid += vec2f(CORNERS_16[i]) * f32(read_image_intensity(
            i32(feature.x) + CORNERS_16[i].x,
            i32(feature.y) + CORNERS_16[i].y
        ));
    }

    var angle = atan2(centroid.y, centroid.x);

    // Now, rotate BRIEF descriptors by angle and run tests
    _ = angle;
    
    var descriptor: BriefDescriptor;

    descriptor.data = array(center_value, v_threshold, 0, 0, 0, 0, 0, 0);

    v_descriptors[global_id.x] = descriptor;
}