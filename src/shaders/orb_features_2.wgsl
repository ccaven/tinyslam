
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
var<storage, read_write> latest_corner_count: atomic<u32>;

// Group 0c = BRIEF Descriptor Data
@group(0) @binding(4)
var<storage, read_write> latest_descriptors: array<BriefDescriptor>;

// Group 1 = Integral Image (different bind group b/c ping pong)
@group(1) @binding(0)
var<storage, read_write> integral_image_in: array<f32>;

@group(1) @binding(1)
var<storage, read_write> integral_image_out: array<f32>;

@group(1) @binding(2)
var<storage, read_write> integral_image_stride: u32;

@group(1) @binding(3)
var<storage, read_write> integral_image_vis: array<u32>;

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

    let img_intensity = read_image_intensity(i32(global_id.x), i32(global_id.y));

    let r = img_intensity.x;
    let g = img_intensity.y;
    let b = img_intensity.z;
    let a = img_intensity.w;

    let gray = r * 0.229 + g * 0.587 + 0.114 * b;

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

    let o = 10;

    let area_a = read_integral_image_out(i32(x) + o, i32(y) + o);
    let area_b = read_integral_image_out(i32(x) - o, i32(y) + o);
    let area_c = read_integral_image_out(i32(x) + o, i32(y) - o);
    let area_d = read_integral_image_out(i32(x) - o, i32(y) - o);

    let average = (area_d + area_a - area_b - area_c) / ((f32(2 * o) + 1.0) * (f32(2 * o) + 1.0));
    
    let vis_r = u32(average * 255.0);
    let vis_g = 128u;
    let vis_b = 128u;

    integral_image_vis[image_index] = (vis_b << 16) | (vis_g << 8) | vis_r;

}

/*
COMPUTE KERNEL 2: FAST Feature Extraction



*/

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

    // Stage 1: workgroup

    let is_corner = true; // TODO

    if is_corner {
        // let index = atomicAdd(&workgroup_counter, 1u);
        // var corner: Corner;
        // corner.x = global_id.x;
        // corner.y = global_id.y;
        // workgroup_corners[index] = corner;
    }

    workgroupBarrier();

    // ???
    // Known: workgroup_counter is no longer going to change
    // Known: workgroup_corners is no longer going to change

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

    _ = latest_corner_count;

    _ = latest_descriptors[0];
}
