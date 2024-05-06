struct Feature {
    x: u32,
    y: u32,
    angle: f32,
    octave: u32
}

@group(0) @binding(0)
var texture: texture_2d<f32>;

@group(0) @binding(1)
var<storage, read_write> corners: array<Feature>;

@group(0) @binding(2)
var<storage, read_write> global_counter: atomic<u32>;

var<push_constant> octave: u32;

var<workgroup> counter: atomic<u32>;
var<workgroup> workgroup_global_index: u32;

var<private> CORNERS_4: array<vec2i, 4> = array(
    vec2i(3, 0),
    vec2i(-3, 0),
    vec2i(0, 3),
    vec2i(0, -3),
);

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

fn rotate_bits_16(num: u32, count: u32) -> u32 {
    let bitmask_16 = 0x0000ffffu;
    return (num >> count) | (num << (16u - count) & bitmask_16);
}

fn detect_streak_16(x: u32) -> u32 {
    let o_6 = x & rotate_bits_16(x, 6u);
    let o_3 = o_6 & rotate_bits_16(o_6, 3u);
    return o_3 & rotate_bits_16(o_3, 2u) & rotate_bits_16(o_3, 1u);
}

@compute
@workgroup_size(8, 8, 1)
fn compute_fast(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_index: u32
) {

    var is_corner = false;
    var workgroup_index: u32;
    var angle: f32;
    var descriptor: array<u32, 8>;

    // Any valid corner must be inside a certain distance from the edges
    // to properly calculate its BRIEF descriptor
    if (all(global_id.xy > vec2u(16, 16)) && all(global_id.xy < textureDimensions(texture))) {
        let center_value = textureLoad(texture, global_id.xy, i32(octave)).x;

        let threshold = 0.20;

        let id_i32 = vec2i(global_id.xy);

        var num_over = 0u;
        var num_under = 0u;
        
        for (var i = 0u; i < 4u; i ++) {
            let corner_value = textureLoad(texture, id_i32 + CORNERS_4[i], i32(octave)).x;
            let diff = corner_value - center_value;
            if diff > threshold {
                num_over ++;
            } else if diff < -threshold {
                num_under ++;
            }
        }

        if num_over >= 3 || num_under >= 3 {

            // Full FAST-16
            var is_over: u32 = 0u;
            var is_under: u32 = 0u;
            var centroid = vec2f(0, 0);

            for (var i = 0u; i < 16u; i ++) {
                let corner_value = textureLoad(texture, id_i32 + CORNERS_16[i], i32(octave)).x;
                let diff = corner_value - center_value;
                
                centroid += corner_value * vec2f(CORNERS_16[i]);

                if diff > threshold {
                    is_over |= 1u << i;
                } else if diff < -threshold {
                    is_under |= 1u << i;
                }
            }

            angle = atan2(centroid.y, centroid.x);

            // Detect streak of 12 bits
            let streak_a = detect_streak_16(is_over);
            let streak_b = detect_streak_16(is_under);

            let streak = (streak_a | streak_b) > 0u;

            if streak {
                workgroup_index = atomicAdd(&counter, 1u);
                is_corner = true;
            }

        }
    }

    workgroupBarrier();

    // No features? :(
    if counter == 0 {
        return;
    }

    // Stand in line to get our workgroup index
    if local_index == 0 {
        workgroup_global_index = atomicAdd(&global_counter, counter);
    }

    workgroupBarrier();

    // Add corner
    if is_corner {
        let global_index = workgroup_global_index + workgroup_index;

        var feature: Feature;

        feature.x = global_id.x;
        feature.y = global_id.y;
        feature.angle = angle;
        feature.octave = octave;

        corners[global_index] = feature;
    }

}