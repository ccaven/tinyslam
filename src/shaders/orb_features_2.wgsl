
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

@group(0) @binding(2)
var<storage, read_write> grayscale_image: array<f32>;

// Group 1 = Integral Image
@group(1) @binding(0)
var<storage, read_write> integral_image_in: array<f32>;

@group(1) @binding(1)
var<storage, read_write> integral_image_out: array<f32>;

@group(1) @binding(2)
var<uniform> integral_image_stride: u32;

// Group 2 = FAST Corner Data
@group(2) @binding(0)
var<storage, read_write> latest_corners: array<Corner>;

@group(2) @binding(1)
var<storage, read_write> latest_corner_count: atomic<u32>;

// Group 3 = BRIEF Descriptor Data
@group(3) @binding(0)
var<storage, read_write> latest_descriptors: array<BriefDescriptor>;

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
    integral_image_out[l] = gray;
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

@compute
@workgroup_size(8, 8, 1)
fn compute_integral_image(
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

    let x = global_id.x;
    let y = global_id.y;
    
    let image_index = y * input_image_size.x + x;

    var current_val = read_integral_image_in(x, y);

    let bx = x - (x % integral_image_stride);
    let by = y - (y % integral_image_stride);

    if x >= bx + integral_image_stride / 2 {
        current_val += read_integral_image_in(
            bx + integral_image_stride / 2 - 1, 
            y
        );
    }

    if y >= by + integral_image_stride / 2 {
        current_val += read_integral_image_in(
            x, 
            by + integral_image_stride / 2 - 1
        );
    }

    if x >= bx + integral_image_stride / 2 && y >= by + integral_image_stride / 2 {
        current_val += read_integral_image_in(
            bx + integral_image_stride / 2 - 1, 
            by + integral_image_stride / 2 - 1
        );
    }

    integral_image_out[image_index] = current_val;
}


/*
COMPUTE KERNEL 2: FAST Feature Extraction

*/
@compute
@workgroup_size(8, 8, 1)
fn compute_fast_corners(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    _ = input_image[0];
    _ = input_image_size.x;

    _ = integral_image_in[0];
    _ = integral_image_out[0];

    _ = integral_image_stride;

    _ = latest_corners[0];

    _ = latest_corner_count;
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
