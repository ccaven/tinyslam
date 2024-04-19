@group(0) @binding(0)
var texture: texture_2d<f32>;

@group(0) @binding(1)
var<storage, read_write> latest_corners: array<vec2u>;

@group(0) @binding(2)
var<storage, read_write> latest_corners_counter: atomic<u32>;

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

@compute
@workgroup_size(8, 8, 1)
fn corner_detector(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_index: u32
) {

    var is_corner = false;
    var workgroup_index: u32;

    if (all(global_id.xy > vec2u(16, 16)) && all(global_id.xy < textureDimensions(texture))) {
        let center_value = textureLoad(texture, global_id.xy, 0).x;

        let threshold = 0.20;

        let id_i32 = vec2i(global_id.xy);

        var num_over = 0u;
        var num_under = 0u;
        
        for (var i = 0u; i < 4u; i ++) {
            let corner_value = textureLoad(texture, id_i32 + CORNERS_4[i], 0).x;
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
                let corner_value = textureLoad(texture, id_i32 + CORNERS_16[i], 0).x;
                let diff = corner_value - center_value;
                if diff > threshold {
                    is_over |= 1u << i;
                } else if diff < -threshold {
                    is_under |= 1u << i;
                }
            }

            // Detect streak of 12 bits
            let o_6 = is_over & rotate_bits_16(is_over, 6u);
            let o_3 = o_6 & rotate_bits_16(o_6, 3u);
            let streak_a = o_3 & rotate_bits_16(o_3, 2u) & rotate_bits_16(o_3, 1u);
            
            let u_6 = is_under & rotate_bits_16(is_under, 6u);
            let u_3 = u_6 & rotate_bits_16(u_6, 3u);
            let streak_b = u_3 & rotate_bits_16(u_3, 1u) & rotate_bits_16(u_3, 2u);

            let streak = (streak_a | streak_b) > 0u;

            if streak {
                workgroup_index = atomicAdd(&counter, 1u);
                is_corner = true;
            }

        }
    }

    workgroupBarrier();

    if counter == 0 {
        return;
    }

    if local_index == 0 {
        workgroup_global_index = atomicAdd(&latest_corners_counter, counter);
    }

    workgroupBarrier();

    if is_corner {
        let global_index = workgroup_global_index + workgroup_index;
        latest_corners[global_index] = global_id.xy;
    }

}