
@group(0) @binding(0)
var input_image: texture_2d<f32>;

@group(0) @binding(1)
var grayscale_image: texture_storage_2d<r32float, read_write>;

@group(0) @binding(2)
var blurred_x: texture_storage_2d<r32float, read_write>;

@group(0) @binding(3)
var blurred_xy: texture_storage_2d<r32float, read_write>;

@group(0) @binding(4)
var<storage, read_write> latest_corners: array<vec2u>;

@group(0) @binding(5)
var<storage, read_write> latest_corners_counter: atomic<u32>;

@compute
@workgroup_size(8, 8, 1)
fn compute_grayscale(
    @builtin(global_invocation_id) global_id: vec3u
) {

    if any(global_id.xy > textureDimensions(input_image)) {
        return;
    }

    let col = textureLoad(input_image, global_id.xy, 0);
    let gray = col.x * 0.229 + col.y * 0.587 + col.b * 0.114;

    textureStore(grayscale_image, global_id.xy, vec4f(gray, 0, 0, 0));
}

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

var<workgroup> workgroup_counter: atomic<u32>;
var<workgroup> workgroup_corners: array<vec2u, 32>;
var<workgroup> workgroup_global_index: atomic<u32>;
var<private> workgroup_index: u32;
var<private> is_corner: bool;

fn rotate_bits_16(num: u32, count: u32) -> u32 {
    return (num >> count) | (num & (1u << count - 1u) << (16u - count));
}

@compute
@workgroup_size(8, 8, 1)
fn compute_corners(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32
) {
    if !(all(global_id.xy > vec2u(16, 16)) && all(global_id.xy < textureDimensions(input_image))) {
        let center_value = textureLoad(grayscale_image, global_id.xy).x;

        let threshold = 0.15;

        let id_i32 = vec2i(global_id.xy);

        var num_over = 0u;
        var num_under = 0u;
        
        for (var i = 0u; i < 4u; i ++) {
            let corner_value = textureLoad(grayscale_image, id_i32 + CORNERS_4[i]).x;
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
                let test_pos = vec2u(id_i32 + CORNERS_16[i]);
                let tex_val = textureLoad(grayscale_image, test_pos);
                let corner_value = tex_val.x;
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
                workgroup_index = atomicAdd(&workgroup_counter, 1u);
                is_corner = true;
            }

        }
    }

    workgroupBarrier();

    if local_invocation_index == 0 {
        workgroup_global_index = atomicAdd(&latest_corners_counter, workgroup_counter);
    }

    workgroupBarrier();

    if is_corner {
        let global_index = workgroup_global_index + workgroup_index;
        latest_corners[global_index] = global_id.xy;
    }
}

var<private> gaussian_offsets: array<i32, 15> = array(
    -7i,
    -6i,
    -5i,
    -4i,
    -3i,
    -2i,
    -1i,
    0i,
    1i,
    2i,
    3i,
    4i,
    5i,
    6i,
    7i
);

var<private> gaussian_weights: array<f32, 15> = array(
    0.00048872837522002,
    0.0024031572869088716,
    0.009246250740395456,
    0.027839605612666265,
    0.06560233156931679,
    0.12099884565428047,
    0.17469734691589356,
    0.19744746769063704,
    0.17469734691589356,
    0.12099884565428047,
    0.06560233156931679,
    0.027839605612666265,
    0.009246250740395456,
    0.0024031572869088716,
    0.00048872837522002
);

@compute
@workgroup_size(8, 8, 1)
fn compute_gaussian_x(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if any(global_id.xy >= textureDimensions(input_image)) {
        return;
    }
    let id = vec2i(global_id.xy);
    var result = vec4f(0.0);
    for (var i = 0; i < 15; i ++) {
        let offset = vec2i(gaussian_offsets[i], 0);
        result.x += textureLoad(grayscale_image, vec2u(id + offset)).x * gaussian_weights[i];
    }
    
    textureStore(blurred_x, global_id.xy, result);
}

@compute
@workgroup_size(8, 8, 1)
fn compute_gaussian_y(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if any(global_id.xy >= textureDimensions(input_image)) {
        return;
    }
    let id = vec2i(global_id.xy);
    var result = vec4f(0.0);
    for (var i = 0; i < 15; i ++) {
        let offset = vec2i(0, gaussian_offsets[i]);
        result.x += textureLoad(blurred_x, vec2u(id + offset)).x * gaussian_weights[i];
    }
    textureStore(blurred_xy, global_id.xy, result);
}

// var<push_constant> stride: u32;

// @compute
// @workgroup_size(8, 8, 1)
// fn compute_integral_image_x(
//     @builtin(global_invocation_id) global_id: vec3u
// ) {
//     let dimensions = textureDimensions(input_image);

//     if global_id.y >= dimensions.y {
//         return;
//     }

//     let h = stride - 1u;
//     let h_bits = global_id.x >> h << (h + 1u);
//     let l_bits = global_id.x & ((1u << h) - 1u);
//     let x = h_bits | l_bits | (1u << h);

//     if x >= dimensions.x {
//         return;
//     }

//     let y = global_id.y;
//     let half_stride = 1u << (stride - 1u);
//     let bx = (x >> stride) << stride;
//     let nx = bx + half_stride - 1u;

//     var current = textureLoad(
//         integral_image,
//         vec2u(x, y)
//     );
    
//     current.x += textureLoad(
//         integral_image,
//         vec2u(nx, y)
//     ).x;

//     textureStore(
//         integral_image,
//         vec2u(x, y),
//         current
//     );
// }

// @compute
// @workgroup_size(8, 8, 1)
// fn compute_integral_image_y(
//     @builtin(global_invocation_id) global_id: vec3u
// ) {
//     let dimensions = textureDimensions(input_image);

//     if global_id.x >= dimensions.x {
//         return;
//     }

//     let h = stride - 1u;
//     let h_bits = global_id.y >> h << (h + 1u);
//     let l_bits = global_id.y & ((1u << h) - 1u);
//     let y = h_bits | l_bits | (1u << h);

//     if y >= dimensions.y {
//         return;
//     }

//     let x = global_id.x;
//     let half_stride = 1u << (stride - 1u);
//     let by = (y >> stride) << stride;
//     let ny = by + half_stride - 1u;

//     let current = textureLoad(
//         integral_image,
//         vec2u(x, y)
//     );
//     let addon = textureLoad(
//         integral_image,
//         vec2u(x, ny)
//     );
//     textureStore(
//         integral_image,
//         vec2u(x, y),
//         current + addon
//     );
// }

