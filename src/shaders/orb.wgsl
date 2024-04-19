
struct Corner {
    x: u32,
    y: u32,
    angle: f32
};

struct BriefDescriptor {
    bits: array<atomic<u32>, 8>
}

// Group 0 = Input Image Information
@group(0) @binding(0)
var<storage, read> input_image: array<u32>;

@group(0) @binding(1)
var<uniform> input_image_size: vec2u;

// Group 0b = FAST Corner Data
@group(0) @binding(2)
var<storage, read_write> latest_corners: array<Corner>;

// Group 0c = BRIEF Descriptor Data
@group(0) @binding(3)
var<storage, read_write> latest_descriptors: array<u32>;

@group(0) @binding(4)
var<storage, read_write> latest_corners_counter: atomic<u32>;

@group(0) @binding(5)
var<storage, read_write> integral_image_out: array<f32>;

@group(0) @binding(6)
var<storage, read_write> integral_image_stride: u32;

@group(0) @binding(7)
var<storage, read_write> integral_image_vis: array<atomic<u32>>;

@group(0) @binding(8)
var<storage, read> previous_corners: array<Corner>;

@group(0) @binding(9)
var<storage, read> previous_descriptors: array<u32>;

@group(0) @binding(10)
var<storage, read> previous_corners_counter: u32;

var<push_constant> integral_image_stride_pc: u32;

// Helper functions

fn read_image_intensity(x: i32, y: i32) -> vec4f {
    if x < 0 || y < 0 || x >= i32(input_image_size.x) || y >= i32(input_image_size.y) {
        return vec4f(0, 0, 0, 0);
    }
    let image_index = u32(y * i32(input_image_size.x) + x);
    let tex_val = input_image[image_index];
    return unpack4x8unorm(tex_val);
}

fn rgba_to_gray(col: vec4f) -> f32 {
    return col.x * 0.229 + col.y * 0.587 + 0.114 * col.b;
}

@compute
@workgroup_size(8, 8, 1)
fn compute_grayscale(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if global_id.x >= input_image_size.x || global_id.y >= input_image_size.y {
        return;
    }

    let gray = rgba_to_gray(read_image_intensity(i32(global_id.x), i32(global_id.y)));

    // Write to output texture
    let l = global_id.x + global_id.y * input_image_size.x;
    // integral_image_in[l] = gray;
    integral_image_out[l] = gray;
}

/*
COMPUTE KERNEL 1: INTEGRAL IMAGE CALCULATION

global_id.x is the x coordinate of the current pixel
global_id.y is the y coordinate of the current pixel

Compute integral image relative to (i - i % stride, j - j % stride)

Keep increasing stride until it is 2^ceil(log2(max(width, height)))
*/


fn read_integral_image_out(x: i32, y: i32) -> f32 {
    if x < 0 || y < 0 || x >= i32(input_image_size.x) || y > i32(input_image_size.y) {
        return 0.0;
    } else {
        return integral_image_out[u32(y) * input_image_size.x + u32(x)];
    }
}

fn compute_local_average(x: u32, y: u32, r: u32) -> f32 {
    let d = r * 2u + 1u;
    let top_left_index = (x - r) + (y - r) * input_image_size.x;
    let top_right_index = top_left_index + d;
    let bottom_left_index = top_left_index + d * input_image_size.x;
    let bottom_right_index = bottom_left_index + d;

    let s = integral_image_out[bottom_right_index] + 
            integral_image_out[top_left_index] - 
            integral_image_out[top_right_index] - 
            integral_image_out[bottom_left_index];
    
    let a = f32(d * d);
    
    return s / a;
}

@compute
@workgroup_size(16, 4, 1)
fn compute_integral_image_x(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if global_id.y >= input_image_size.y {
        return;
    }

    let stride = integral_image_stride_pc;

    // We launched half as many workgroups, so we need to calculate the real x
    let h = stride - 1u;
    let h_bits = global_id.x >> h << (h + 1u);
    let l_bits = global_id.x & ((1u << h) - 1u);
    let x = h_bits | l_bits | (1u << h);

    if x >= input_image_size.x {
        return;
    }

    let y = global_id.y;
    let half_stride = 1u << (stride - 1u);
    let bx = (x >> stride) << stride;
    let lx = x - bx;
    let nx = bx + half_stride - 1u;
    if lx >= half_stride {
        integral_image_out[y * input_image_size.x + x] += integral_image_out[y * input_image_size.x + nx];
    }
}

@compute
@workgroup_size(8, 8, 1)
fn compute_integral_image_y(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if global_id.x >= input_image_size.x {
        return;
    }

    let stride = integral_image_stride_pc;

    let h = stride - 1u;
    let h_bits = global_id.y >> h << (h + 1u);
    let l_bits = global_id.y & ((1u << h) - 1u);
    let y = h_bits | l_bits | (1u << h);

    if y >= input_image_size.y {
        return;
    }

    let x = global_id.x;
    let half_stride = 1u << (stride - 1u);
    let by = (y >> stride) << stride;
    let ly = y - by;
    let ny = by + half_stride - 1u;
    if ly >= half_stride {
        integral_image_out[y * input_image_size.x + x] += integral_image_out[ny * input_image_size.x + x];
    }
}

var<private> CORNERS_4: array<vec2i, 4> = array(
    vec2i(3, 0),
    vec2i(-3, 0),
    vec2i(0, 3),
    vec2i(0, -3),
);

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

var<workgroup> workgroup_counter: atomic<u32>;
var<workgroup> workgroup_corners: array<Corner, 64>;
var<workgroup> workgroup_global_index: atomic<u32>;

fn rotate_bits_16(num: u32, count: u32) -> u32 {
    return (num >> count) | (num & (1u << count - 1u) << (16u - count));
}

@compute
@workgroup_size(8, 8, 1)
fn compute_fast_corners(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(local_invocation_index) local_invocation_index: u32
) {
    if global_id.x > 16 || 
       global_id.y > 16 || 
       global_id.x <= input_image_size.x - 16 || 
       global_id.y <= input_image_size.y - 16
    {
        let center_value = rgba_to_gray(read_image_intensity(i32(global_id.x), i32(global_id.y)));

        // Shortcut to discard pixels before full FAST check
        let threshold = 0.15;

        var num_over = 0u;
        var num_under = 0u;
        
        for (var i = 0u; i < 4u; i ++) {
            let corner_value = rgba_to_gray(read_image_intensity(
                i32(global_id.x) + CORNERS_4[i].x,
                i32(global_id.y) + CORNERS_4[i].y
            ));

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
                let corner_value = rgba_to_gray(read_image_intensity(
                    i32(global_id.x) + CORNERS_16[i].x,
                    i32(global_id.y) + CORNERS_16[i].y
                ));

                centroid += vec2f(CORNERS_16[i]) * corner_value;

                let diff = corner_value - center_value;
                
                if diff > threshold {
                    is_over |= 1u << i;
                } else if diff < -threshold {
                    is_under |= 1u << i;
                }
            }

            // Detect streak of 12 bits
            
            // let streak_a = is_over &
            //     rotate_bits_16(is_over, 1u) &
            //     rotate_bits_16(is_over, 2u) &
            //     rotate_bits_16(is_over, 3u) &
            //     rotate_bits_16(is_over, 4u) &
            //     rotate_bits_16(is_over, 5u) &
            //     rotate_bits_16(is_over, 6u) &
            //     rotate_bits_16(is_over, 7u) &
            //     rotate_bits_16(is_over, 8u) &
            //     rotate_bits_16(is_over, 9u) &
            //     rotate_bits_16(is_over, 10u) &
            //     rotate_bits_16(is_over, 11u) &
            //     rotate_bits_16(is_over, 12u);

            let o_6 = is_over & rotate_bits_16(is_over, 6u);
            let o_3 = o_6 & rotate_bits_16(o_6, 3u);
            let streak_a = o_3 & rotate_bits_16(o_3, 2u) & rotate_bits_16(o_3, 1u);
            
            let u_6 = is_under & rotate_bits_16(is_under, 6u);
            let u_3 = u_6 & rotate_bits_16(u_6, 3u);
            let streak_b = u_3 & rotate_bits_16(u_3, 1u) & rotate_bits_16(u_3, 2u);
            
            // let streak_b = is_under &
            //     rotate_bits_16(is_under, 1u) &
            //     rotate_bits_16(is_under, 2u) &
            //     rotate_bits_16(is_under, 3u) &
            //     rotate_bits_16(is_under, 4u) &
            //     rotate_bits_16(is_under, 5u) &
            //     rotate_bits_16(is_under, 6u) &
            //     rotate_bits_16(is_under, 7u) &
            //     rotate_bits_16(is_under, 8u) &
            //     rotate_bits_16(is_under, 9u) &
            //     rotate_bits_16(is_under, 10u) &
            //     rotate_bits_16(is_under, 11u) &
            //     rotate_bits_16(is_under, 12u);

            let streak = (streak_a | streak_b) > 0u;

            

            var over_streak = 0u;
            var max_over_streak = 0u;
            var under_streak = 0u;
            var max_under_streak = 0u;
            for (var i = 0u; i < 24u; i ++) {
                let j = i % 16u;
                if extractBits(is_over, j, 1u) == 1u {
                    over_streak ++;
                    max_over_streak = max(over_streak, max_over_streak);
                } else {
                    over_streak = 0u;
                }
                if extractBits(is_under, j, 1u) == 1u {
                    under_streak ++;
                    max_under_streak = max(under_streak, max_under_streak);
                } else {
                    under_streak = 0u;
                }
            }

            let streak_c = max_over_streak > 11u || max_under_streak > 11u;

            //if max(max_over_streak, max_under_streak) > 11u {
            if streak_c {
                let workgroup_index = atomicAdd(&workgroup_counter, 1u);

                var feature: Corner;

                feature.x = global_id.x;
                feature.y = global_id.y;
                feature.angle = atan2(centroid.y, centroid.x);

                workgroup_corners[workgroup_index] = feature;
            }

        }
    }

    workgroupBarrier();

    if workgroup_counter == 0u {
        return;
    }

    if local_invocation_index == 0 {
        workgroup_global_index = atomicAdd(&latest_corners_counter, workgroup_counter);
    }

    workgroupBarrier();

    if local_invocation_index >= workgroup_counter {
        return;
    }

    let feature = workgroup_corners[local_invocation_index];
    let global_index = workgroup_global_index + local_invocation_index;
    latest_corners[global_index] = feature;  
}

@compute
@workgroup_size(8, 8, 1)
fn visualize_box_blur(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if global_id.x >= input_image_size.x || global_id.y >= input_image_size.y {
        return;
    }

    let r = 5u;
    let avg = compute_local_average(global_id.x, global_id.y, r);

    let vis_r = u32(avg * 255.0);
    let vis_g = u32(avg * 255.0);
    let vis_b = u32(avg * 255.0);

    let image_index = global_id.x + global_id.y * input_image_size.x;
    integral_image_vis[image_index] = (vis_b << 16) | (vis_g << 8) | vis_r;
}

@compute
@workgroup_size(64, 1, 1)
fn visualize_features(
    @builtin(global_invocation_id) global_id: vec3u
) {
    if global_id.x >= latest_corners_counter {
        return;
    }

    let feature = latest_corners[global_id.x];

    let vis_r = 255u;
    let vis_g = 0u;
    let vis_b = 0u;

    for (var i = 0u; i < 16u; i ++) {
        let pixel_x = u32(i32(feature.x) + CORNERS_16[i].x);
        let pixel_y = u32(i32(feature.y) + CORNERS_16[i].y);
        let image_index = pixel_x + pixel_y * input_image_size.x;
        
        atomicStore(&integral_image_vis[image_index], (vis_b << 16) | (vis_g << 8) | vis_r);
    }    
}


/*
COMPUTE KERNEL 3: BRIEF Feature Descriptors
*/
var<workgroup> current_descriptors: array<atomic<u32>, 2>;

var<private> brief_descriptors: array<vec4i, 256> = array(
    vec4i(8,-3, 9,5),
    vec4i(4,2, 7,-12),
    vec4i(-11,9, -8,2),
    vec4i(7,-12, 12,-13),
    vec4i(2,-13, 2,12),
    vec4i(1,-7, 1,6),
    vec4i(-2,-10, -2,-4),
    vec4i(-13,-13, -11,-8),
    vec4i(-13,-3, -12,-9),
    vec4i(10,4, 11,9),
    vec4i(-13,-8, -8,-9),
    vec4i(-11,7, -9,12),
    vec4i(7,7, 12,6),
    vec4i(-4,-5, -3,0),
    vec4i(-13,2, -12,-3),
    vec4i(-9,0, -7,5),
    vec4i(12,-6, 12,-1),
    vec4i(-3,6, -2,12),
    vec4i(-6,-13, -4,-8),
    vec4i(11,-13, 12,-8),
    vec4i(4,7, 5,1),
    vec4i(5,-3, 10,-3),
    vec4i(3,-7, 6,12),
    vec4i(-8,-7, -6,-2),
    vec4i(-2,11, -1,-10),
    vec4i(-13,12, -8,10),
    vec4i(-7,3, -5,-3),
    vec4i(-4,2, -3,7),
    vec4i(-10,-12, -6,11),
    vec4i(5,-12, 6,-7),
    vec4i(5,-6, 7,-1),
    vec4i(1,0, 4,-5),
    vec4i(9,11, 11,-13),
    vec4i(4,7, 4,12),
    vec4i(2,-1, 4,4),
    vec4i(-4,-12, -2,7),
    vec4i(-8,-5, -7,-10),
    vec4i(4,11, 9,12),
    vec4i(0,-8, 1,-13),
    vec4i(-13,-2, -8,2),
    vec4i(-3,-2, -2,3),
    vec4i(-6,9, -4,-9),
    vec4i(8,12, 10,7),
    vec4i(0,9, 1,3),
    vec4i(7,-5, 11,-10),
    vec4i(-13,-6, -11,0),
    vec4i(10,7, 12,1),
    vec4i(-6,-3, -6,12),
    vec4i(10,-9, 12,-4),
    vec4i(-13,8, -8,-12),
    vec4i(-13,0, -8,-4),
    vec4i(3,3, 7,8),
    vec4i(5,7, 10,-7),
    vec4i(-1,7, 1,-12),
    vec4i(3,-10, 5,6),
    vec4i(2,-4, 3,-10),
    vec4i(-13,0, -13,5),
    vec4i(-13,-7, -12,12),
    vec4i(-13,3, -11,8),
    vec4i(-7,12, -4,7),
    vec4i(6,-10, 12,8),
    vec4i(-9,-1, -7,-6),
    vec4i(-2,-5, 0,12),
    vec4i(-12,5, -7,5),
    vec4i(3,-10, 8,-13),
    vec4i(-7,-7, -4,5),
    vec4i(-3,-2, -1,-7),
    vec4i(2,9, 5,-11),
    vec4i(-11,-13, -5,-13),
    vec4i(-1,6, 0,-1),
    vec4i(5,-3, 5,2),
    vec4i(-4,-13, -4,12),
    vec4i(-9,-6, -9,6),
    vec4i(-12,-10, -8,-4),
    vec4i(10,2, 12,-3),
    vec4i(7,12, 12,12),
    vec4i(-7,-13, -6,5),
    vec4i(-4,9, -3,4),
    vec4i(7,-1, 12,2),
    vec4i(-7,6, -5,1),
    vec4i(-13,11, -12,5),
    vec4i(-3,7, -2,-6),
    vec4i(7,-8, 12,-7),
    vec4i(-13,-7, -11,-12),
    vec4i(1,-3, 12,12),
    vec4i(2,-6, 3,0),
    vec4i(-4,3, -2,-13),
    vec4i(-1,-13, 1,9),
    vec4i(7,1, 8,-6),
    vec4i(1,-1, 3,12),
    vec4i(9,1, 12,6),
    vec4i(-1,-9, -1,3),
    vec4i(-13,-13, -10,5),
    vec4i(7,7, 10,12),
    vec4i(12,-5, 12,9),
    vec4i(6,3, 7,11),
    vec4i(5,-13, 6,10),
    vec4i(2,-12, 2,3),
    vec4i(3,8, 4,-6),
    vec4i(2,6, 12,-13),
    vec4i(9,-12, 10,3),
    vec4i(-8,4, -7,9),
    vec4i(-11,12, -4,-6),
    vec4i(1,12, 2,-8),
    vec4i(6,-9, 7,-4),
    vec4i(2,3, 3,-2),
    vec4i(6,3, 11,0),
    vec4i(3,-3, 8,-8),
    vec4i(7,8, 9,3),
    vec4i(-11,-5, -6,-4),
    vec4i(-10,11, -5,10),
    vec4i(-5,-8, -3,12),
    vec4i(-10,5, -9,0),
    vec4i(8,-1, 12,-6),
    vec4i(4,-6, 6,-11),
    vec4i(-10,12, -8,7),
    vec4i(4,-2, 6,7),
    vec4i(-2,0, -2,12),
    vec4i(-5,-8, -5,2),
    vec4i(7,-6, 10,12),
    vec4i(-9,-13, -8,-8),
    vec4i(-5,-13, -5,-2),
    vec4i(8,-8, 9,-13),
    vec4i(-9,-11, -9,0),
    vec4i(1,-8, 1,-2),
    vec4i(7,-4, 9,1),
    vec4i(-2,1, -1,-4),
    vec4i(11,-6, 12,-11),
    vec4i(-12,-9, -6,4),
    vec4i(3,7, 7,12),
    vec4i(5,5, 10,8),
    vec4i(0,-4, 2,8),
    vec4i(-9,12, -5,-13),
    vec4i(0,7, 2,12),
    vec4i(-1,2, 1,7),
    vec4i(5,11, 7,-9),
    vec4i(3,5, 6,-8),
    vec4i(-13,-4, -8,9),
    vec4i(-5,9, -3,-3),
    vec4i(-4,-7, -3,-12),
    vec4i(6,5, 8,0),
    vec4i(-7,6, -6,12),
    vec4i(-13,6, -5,-2),
    vec4i(1,-10, 3,10),
    vec4i(4,1, 8,-4),
    vec4i(-2,-2, 2,-13),
    vec4i(2,-12, 12,12),
    vec4i(-2,-13, 0,-6),
    vec4i(4,1, 9,3),
    vec4i(-6,-10, -3,-5),
    vec4i(-3,-13, -1,1),
    vec4i(7,5, 12,-11),
    vec4i(4,-2, 5,-7),
    vec4i(-13,9, -9,-5),
    vec4i(7,1, 8,6),
    vec4i(7,-8, 7,6),
    vec4i(-7,-4, -7,1),
    vec4i(-8,11, -7,-8),
    vec4i(-13,6, -12,-8),
    vec4i(2,4, 3,9),
    vec4i(10,-5, 12,3),
    vec4i(-6,-5, -6,7),
    vec4i(8,-3, 9,-8),
    vec4i(2,-12, 2,8),
    vec4i(-11,-2, -10,3),
    vec4i(-12,-13, -7,-9),
    vec4i(-11,0, -10,-5),
    vec4i(5,-3, 11,8),
    vec4i(-2,-13, -1,12),
    vec4i(-1,-8, 0,9),
    vec4i(-13,-11, -12,-5),
    vec4i(-10,-2, -10,11),
    vec4i(-3,9, -2,-13),
    vec4i(2,-3, 3,2),
    vec4i(-9,-13, -4,0),
    vec4i(-4,6, -3,-10),
    vec4i(-4,12, -2,-7),
    vec4i(-6,-11, -4,9),
    vec4i(6,-3, 6,11),
    vec4i(-13,11, -5,5),
    vec4i(11,11, 12,6),
    vec4i(7,-5, 12,-2),
    vec4i(-1,12, 0,7),
    vec4i(-4,-8, -3,-2),
    vec4i(-7,1, -6,7),
    vec4i(-13,-12, -8,-13),
    vec4i(-7,-2, -6,-8),
    vec4i(-8,5, -6,-9),
    vec4i(-5,-1, -4,5),
    vec4i(-13,7, -8,10),
    vec4i(1,5, 5,-13),
    vec4i(1,0, 10,-13),
    vec4i(9,12, 10,-1),
    vec4i(5,-8, 10,-9),
    vec4i(-1,11, 1,-13),
    vec4i(-9,-3, -6,2),
    vec4i(-1,-10, 1,12),
    vec4i(-13,1, -8,-10),
    vec4i(8,-11, 10,-6),
    vec4i(2,-13, 3,-6),
    vec4i(7,-13, 12,-9),
    vec4i(-10,-10, -5,-7),
    vec4i(-10,-8, -8,-13),
    vec4i(4,-6, 8,5),
    vec4i(3,12, 8,-13),
    vec4i(-4,2, -3,-3),
    vec4i(5,-13, 10,-12),
    vec4i(4,-13, 5,-1),
    vec4i(-9,9, -4,3),
    vec4i(0,3, 3,-9),
    vec4i(-12,1, -6,1),
    vec4i(3,2, 4,-8),
    vec4i(-10,-10, -10,9),
    vec4i(8,-13, 12,12),
    vec4i(-8,-12, -6,-5),
    vec4i(2,2, 3,7),
    vec4i(10,6, 11,-8),
    vec4i(6,8, 8,-12),
    vec4i(-7,10, -6,5),
    vec4i(-3,-9, -3,9),
    vec4i(-1,-13, -1,5),
    vec4i(-3,-7, -3,4),
    vec4i(-8,-2, -8,3),
    vec4i(4,2, 12,12),
    vec4i(2,-5, 3,11),
    vec4i(6,-9, 11,-13),
    vec4i(3,-1, 7,12),
    vec4i(11,-1, 12,4),
    vec4i(-3,0, -3,6),
    vec4i(4,-11, 4,12),
    vec4i(2,-4, 2,1),
    vec4i(-10,-6, -8,1),
    vec4i(-13,7, -11,1),
    vec4i(-13,12, -11,-13),
    vec4i(6,0, 11,-13),
    vec4i(0,-1, 1,4),
    vec4i(-13,3, -9,-2),
    vec4i(-9,8, -6,-3),
    vec4i(-13,-6, -8,-2),
    vec4i(5,-9, 8,10),
    vec4i(2,7, 3,-9),
    vec4i(-1,-6, -1,-1),
    vec4i(9,5, 11,-2),
    vec4i(11,-3, 12,-8),
    vec4i(3,0, 3,5),
    vec4i(-1,4, 0,10),
    vec4i(3,-6, 4,5),
    vec4i(-13,0, -10,5),
    vec4i(5,8, 12,11),
    vec4i(8,9, 9,-6),
    vec4i(7,-4, 8,-12),
    vec4i(-10,4, -10,9),
    vec4i(7,3, 12,4),
    vec4i(9,-7, 10,-2),
    vec4i(7,0, 12,-2),
    vec4i(-1,-6, 0,-11)
);

@compute
@workgroup_size(32, 2, 1)
fn compute_brief_descriptors(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32
) {
    let feature_index = global_id.x;
    let descriptor_index = global_id.y;
    let descriptor_id = local_id.x;

    if feature_index >= latest_corners_counter {
        return;
    }

    let feature = latest_corners[feature_index];
    let test_point = brief_descriptors[descriptor_index];

    // TODO: Rotate test point based on feature angle
    let st = sin(feature.angle);
    let ct = cos(feature.angle);

    let m = mat2x2f(
        ct, -st,
        st, ct
    );

    let test_point_a = vec2i(round(m * vec2f(test_point.xy)));
    let test_point_b = vec2i(round(m * vec2f(test_point.zw)));

    let avg_1 = compute_local_average(
        u32(i32(feature.x) + test_point_a.x), 
        u32(i32(feature.y) + test_point_a.y), 
        2u
    );
    let avg_2 = compute_local_average(
        u32(i32(feature.x) + test_point_b.x), 
        u32(i32(feature.y) + test_point_b.y), 
        2u
    );

    if avg_1 > avg_2 {
        let descriptor_bit = local_id.y;
        atomicOr(&current_descriptors[descriptor_id], 1u << descriptor_bit);
    }

    workgroupBarrier();

    if local_index >= 2 {
        return;
    }

    let which_u32 = workgroup_id.y;
    latest_descriptors[feature_index * 8u + which_u32] = current_descriptors[descriptor_id];
}

@compute
@workgroup_size(8, 8, 1)
fn match_features(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let i = global_id.x;
    let j = global_id.y;

    if i >= latest_corners_counter || j >= previous_corners_counter {
        return;
    }

    var score = 0u;
    for (var k = 0u; k < 8u; k ++) {
        score += countOneBits(
            ~(latest_descriptors[i * 8 + k] ^
             previous_descriptors[j * 8 + k])
        );
    }

    if score == 256u {
        // It's a match!
        // Rasterize the line

        let feature_a = latest_corners[i];
        let feature_b = latest_corners[j];

        for (var l = 0u; l < 100u; l ++) {
            let px = feature_a.x + (feature_b.x * l - feature_a.x * l) / 100u;
            let py = feature_a.y + (feature_b.y * l - feature_a.y * l) / 100u;

            let image_index = px + py * input_image_size.x;

            let vis_r = 0u;
            let vis_g = 0u;
            let vis_b = 255u;

            atomicStore(&integral_image_vis[image_index], (vis_b << 16) | (vis_g << 8) | vis_r);
        }
    }
}