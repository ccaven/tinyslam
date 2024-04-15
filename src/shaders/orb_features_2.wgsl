
struct Corner {
    x: u32,
    y: u32
};

struct BriefDescriptor {
    bits: array<u32, 8>
}

// Group 0 = Input Image Information
@group(0) @binding(0)
var<storage, read> input_image: array<u32>;

@group(0) @binding(1)
var<uniform> input_image_size: vec2u;

// Group 0b = FAST Corner Data

@group(0) @binding(2)
var<storage, read_write> latest_corners: array<Corner>;

@group(0) @binding(3)
var<storage, read_write> chunk_corners: array<array<Corner, 64>>;

@group(0) @binding(4)
var<storage, read_write> chunk_counters: array<atomic<u32>>;

@group(0) @binding(5)
var<storage, read_write> chunk_counters_global: array<u32>;

@group(0) @binding(6)
var<storage, read_write> chunk_stride: u32;

// Group 0c = BRIEF Descriptor Data
@group(0) @binding(7)
var<storage, read_write> latest_descriptors: array<BriefDescriptor>;

// Group 1 = Integral Image (different bind group b/c ping pong)
@group(1) @binding(0)
var<storage, read_write> integral_image_in: array<f32>;

@group(1) @binding(1)
var<storage, read_write> integral_image_out: array<f32>;

@group(1) @binding(2)
var<storage, read_write> integral_image_stride: u32;

@group(1) @binding(3)
var<storage, read_write> integral_image_vis: array<atomic<u32>>;

// Helper functions

fn read_image_intensity(x: i32, y: i32) -> vec4f {
    if x < 0 || y < 0 || x >= i32(input_image_size.x) || y >= i32(input_image_size.y) {
        return vec4f(0, 0, 0, 0);
    }

    let image_index = u32(y * i32(input_image_size.x) + x);

    let tex_val = input_image[image_index];

    let tex_val_a = f32((tex_val >> 24) & 255) / 255.0;
    let tex_val_b = f32((tex_val >> 16) & 255) / 255.0;
    let tex_val_g = f32((tex_val >> 8) & 255) / 255.0;
    let tex_val_r = f32((tex_val) & 255) / 255.0;
    
    return vec4f(tex_val_r, tex_val_g, tex_val_b, tex_val_a);
}

fn rgba_to_gray(col: vec4f) -> f32 {
    return col.x * 0.229 + col.y * 0.587 + 0.114 * col.b;
}

@compute
@workgroup_size(8, 8, 1)
fn compute_grayscale(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    _ = input_image[0];
    _ = input_image_size.x;

    _ = integral_image_in[0];
    _ = integral_image_out[0];

    _ = integral_image_stride;

    if global_id.x >= input_image_size.x || global_id.y >= input_image_size.y {
        return;
    }

    let gray = rgba_to_gray(read_image_intensity(i32(global_id.x), i32(global_id.y)));

    // Write to output texture
    let l = global_id.x + global_id.y * input_image_size.x;
    integral_image_in[l] = gray;
}

/*
COMPUTE KERNEL 1: INTEGRAL IMAGE CALCULATION

global_id.x is the x coordinate of the current pixel
global_id.y is the y coordinate of the current pixel

Compute integral image relative to (i - i % stride, j - j % stride)

Keep increasing stride until it is 2^ceil(log2(max(width, height)))
*/

fn read_integral_image_in(x: u32, y: u32) -> f32 {
    return integral_image_in[y * input_image_size.x + x];
}

fn read_integral_image_out(x: i32, y: i32) -> f32 {
    if x < 0 || y < 0 || x >= i32(input_image_size.x) || y > i32(input_image_size.y) {
        return 0.0;
    } else {
        return integral_image_out[u32(y) * input_image_size.x + u32(x)];
    }
}

@compute
@workgroup_size(1, 1, 1)
fn zero_stride() {
    integral_image_stride = 0u;
}

@compute
@workgroup_size(1, 1, 1)
fn increment_stride() {
    integral_image_stride += 1u;
}

@compute
@workgroup_size(8, 8, 1)
fn compute_integral_image(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if global_id.x >= input_image_size.x || global_id.y >= input_image_size.y {
        return;
    }

    let x = global_id.x;
    let y = global_id.y;
    
    let image_index = y * input_image_size.x + x;

    var current_val = read_integral_image_in(x, y);

    let half_stride = 1u << (integral_image_stride - 1u);

    let bx = (x >> integral_image_stride) << integral_image_stride;
    let by = (y >> integral_image_stride) << integral_image_stride;

    let lx = x - bx;
    let ly = y - by;

    if lx >= half_stride {
        current_val += read_integral_image_in(
            bx + half_stride - 1u, 
            y
        );
    }

    if ly >= half_stride {
        current_val += read_integral_image_in(
            x, 
            by + half_stride - 1u
        );
    }

    if lx >= half_stride && ly >= half_stride {
        current_val += read_integral_image_in(
            bx + half_stride - 1u,
            by + half_stride - 1u
        );
    }

    integral_image_out[image_index] = current_val;    
}

@compute
@workgroup_size(8, 8, 1)
fn box_blur_visualization(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    if global_id.x >= input_image_size.x || global_id.y >= input_image_size.y {
        return;
    }

    let x = global_id.x;
    let y = global_id.y;
    
    let image_index = y * input_image_size.x + x;

    let o = 1;

    let area_a = read_integral_image_out(i32(x) + o, i32(y) + o);
    let area_b = read_integral_image_out(i32(x) - o, i32(y) + o);
    let area_c = read_integral_image_out(i32(x) + o, i32(y) - o);
    let area_d = read_integral_image_out(i32(x) - o, i32(y) - o);

    let average = (area_d + area_a - area_b - area_c) / ((f32(2 * o) + 1.0) * (f32(2 * o) + 1.0));
    
    let vis_r = u32(average * 255.0);
    let vis_g = u32(average * 255.0);
    let vis_b = u32(average * 255.0);

    atomicStore(&integral_image_vis[image_index], (vis_b << 16) | (vis_g << 8) | vis_r);
}

/* COMPUTE KERNEL 2: FAST Feature Extraction */

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

@compute
@workgroup_size(8, 8, 1)
fn compute_fast_corners(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u
) {
    if global_id.x < 16 || 
       global_id.y < 16 || 
       global_id.x >= input_image_size.x - 16 || 
       global_id.y >= input_image_size.y - 16
    {
        return;
    }

    let center_value = rgba_to_gray(read_image_intensity(i32(global_id.x), i32(global_id.y)));

    // Shortcut to discard pixels before full FAST check
    let threshold = 0.25;

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

    if num_over < 3 && num_under < 3 {

        // Full FAST-16
        var is_over: u32 = 0u;
        var is_under: u32 = 0u;

        for (var i = 0u; i < 16u; i ++) {
            let corner_value = rgba_to_gray(read_image_intensity(
                i32(global_id.x) + CORNERS_16[i].x,
                i32(global_id.y) + CORNERS_16[i].y
            ));

            let diff = corner_value - center_value;
            
            if diff > threshold {
                is_over |= 1u << i;
            } else if diff < -threshold {
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

        if max(max_over_streak, max_under_streak) > 11u {

            let num_chunks_x = (input_image_size.x + 7u) / 8u;
            let chunk_id = workgroup_id.y * num_chunks_x + workgroup_id.x;

            let chunk_index = atomicAdd(&chunk_counters[chunk_id], 1u);
            //let chunk_index = atomicAdd(&workgroup_counter, 1u);

            var feature: Corner;

            feature.x = global_id.x;
            feature.y = global_id.y;

            chunk_corners[chunk_id][chunk_index] = feature;
        }

    }

    // workgroupBarrier();

    // if local_invocation_id.x == 0 && local_invocation_id.y == 0 {
    //     let num_chunks_x = (input_image_size.x + 7u) / 8u;
    //     let chunk_id = workgroup_id.y * num_chunks_x + workgroup_id.x;
    //     chunk_counters[chunk_id] = atomicLoad(&workgroup_counter);
    // }

}

@compute
@workgroup_size(1, 1, 1)
fn reset_chunk_stride() {
    chunk_stride = 1u;
}

@compute
@workgroup_size(1, 1, 1)
fn increment_chunk_stride() {
    chunk_stride += 1u;
}

@compute
@workgroup_size(16, 1, 1)
fn compute_integral_indices(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let num_chunks_x = (input_image_size.x + 7u) / 8u;
    let num_chunks_y = (input_image_size.y + 7u) / 8u;
    let num_chunks = num_chunks_x * num_chunks_y;

    let chunk_id = global_id.x;

    if chunk_id >= num_chunks {
        return;
    }

    let half_stride = 1u << (chunk_stride - 1u);

    let stride_id = (chunk_id >> chunk_stride) << chunk_stride;
    let local_id = chunk_id & (1u << stride_id - 1u);

    if local_id >= half_stride {
        chunk_counters_global[chunk_id] += chunk_counters_global[stride_id + half_stride - 1u];
    }
}

@compute
@workgroup_size(8, 8, 1)
fn load_into_full_array(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let num_chunks_x = (input_image_size.x + 7u) / 8u;
    let num_chunks_y = (input_image_size.y + 7u) / 8u;
    let num_chunks = num_chunks_x * num_chunks_y;

    let chunk_id = global_id.x;

    if chunk_id >= num_chunks {
        return;
    }

    let num_chunk_features = chunk_counters[chunk_id];

    let feature_id = global_id.y;

    if feature_id >= num_chunk_features {
        return;
    }

    let storage_index = chunk_counters_global[chunk_id] + feature_id;

    let feature = chunk_corners[chunk_id][feature_id];

    latest_corners[storage_index] = feature;

    let vis_r = 255u;
    let vis_g = 0u;
    let vis_b = 0u;
    
    let image_index = feature.x + feature.y * input_image_size.x;

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
@compute
@workgroup_size(8, 8, 1)
fn compute_brief_descriptors(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    _ = input_image[0];
    _ = input_image_size.x;

    _ = integral_image_in[0];
    _ = integral_image_out[0];

    _ = integral_image_stride;

    _ = latest_corners[0];

    _ = latest_descriptors[0];
}
