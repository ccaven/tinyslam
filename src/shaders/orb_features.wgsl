
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

@group(3) @binding(0)
var<storage, read_write> v_previous_descriptors: array<BriefDescriptor>;

@group(3) @binding(1)
var<storage, read_write> v_previous_index: u32;

@group(3) @binding(2)
var<storage, read_write> v_matches: array<u32>;

@group(3) @binding(3)
var<storage, read_write> v_match_index: atomic<u32>;

@group(3) @binding(4)
var<storage, read_write> v_previous_features: array<FeatureData>;
/*
HELPER FUNCTIONS

read_image_intensity - technically there's one byte per pixel, which are packed into u32s
*/

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

/*
ORB FEATURE CONSTANTS

CORNERS_16 - the FAST corner offsets
BRIEF_QUERIES - the non-rotated BRIEF query positions (x1, y1, x2, y2)

*/
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

var<private> BRIEF_QUERIES: array<vec4i, 256> = array(
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

/*
COMPUTE KERNEL 1: FEATURE EXTRACTION

global_id.x is the x coordinate of the current pixel
global_id.y is the y coordinate of the current pixel

num_workgroups_x is the width of the image divided by the workgroup size
num_workgroups_y is the height of the image divided by the workgroup size

TODO: Properly implement BRIEF feature descriptors
 1. Add pass to compute integral image

*/
@compute
@workgroup_size(8, 8, 1)
fn compute_orb(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    
    if global_id.x < 3 || global_id.y < 3 || global_id.x > image_size.x - 4 || global_id.y > image_size.y - 4 {
        return;
    }

    let center_value = read_image_intensity(i32(global_id.x), i32(global_id.y));

    // Shortcut to discard pixels before full FAST check
    var num_over = 0u;
    var num_under = 0u;
    
    for (var i = 0u; i < 4u; i ++) {
        let corner_value = read_image_intensity(
            i32(global_id.x) + CORNERS_4[i].x,
            i32(global_id.y) + CORNERS_4[i].y
        );

        let diff = i32(corner_value) - i32(center_value);
        
        if diff > i32(v_threshold) {
            num_over ++;
        } else if diff < -i32(v_threshold) {
            num_under ++;
        }
    }

    if num_over < 3 && num_under < 3 {
        return;
    }

    // Full FAST-16
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

    if 11 < max_over_streak || max_under_streak > 11 {
        let index = atomicAdd(&v_index, 1u);

        var data: FeatureData;

        data.x = global_id.x;
        data.y = global_id.y;

        v_features[index] = data;
    }
    
}

/*
COMPUTE KERNEL 2: BRIEF FEATURE DESCRIPTORS

global_id.x is the index of the current FAST feature

num_workgroups_x is the maximum number of features divided by the workgroup size

*/
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

    var centroid = vec2f(0.0, 0.0);

    for (var i = 0u; i < 16u; i ++) {
        centroid += vec2f(CORNERS_16[i]) * f32(read_image_intensity(
            i32(feature.x) + CORNERS_16[i].x,
            i32(feature.y) + CORNERS_16[i].y
        ));
    }

    var angle = atan2(centroid.y, centroid.x);
    let ct = cos(angle);
    let st = sin(angle);
    let rotation_matrix = mat2x2f(
        ct, -st,
        st, ct
    );
    // let rotation_matrix = mat2x2f(
    //     1.0, 0.0,
    //     0.0, 1.0
    // );

    // Now, rotate BRIEF descriptors by angle and run tests
    var descriptor: BriefDescriptor;
    var data: array<u32, 8> = array(0, 0, 0, 0, 0, 0, 0, 0);
    for (var i = 0u; i < 256u; i ++) {
        let pos_a = rotation_matrix * vec2f(BRIEF_QUERIES[i].xy);
        let pos_b = rotation_matrix * vec2f(BRIEF_QUERIES[i].zw);

        let pos_ai = vec2i(round(pos_a));
        let pos_bi = vec2i(round(pos_b));

        let intensity_a = read_image_intensity(
            i32(feature.x) + pos_ai.x, 
            i32(feature.y) + pos_ai.y
        );
        let intensity_b = read_image_intensity(
            i32(feature.x) + pos_bi.x, 
            i32(feature.y) + pos_bi.y
        );

        let j = i / 8;
        let k = i & 7;

        if intensity_b > intensity_a {
            data[j] |= 1u << k;
        }
    }    
    descriptor.data = data;
    v_descriptors[global_id.x] = descriptor;
}

/*
COMPUTE KERNEL 3: FEATURE MATCHING

global_id.x is the index of the feature in the current image
global_ud.y is the index of the feature in the prior image

num_workgroups_x is the maximum number of features divided by the workgroup size
num_workgroups_y is the maximum number of features divided by the workgroup size
*/
@compute
@workgroup_size(8, 8, 1)
fn compute_matches(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let id_a = global_id.x;
    let id_b = global_id.y;

    if id_a >= v_index || id_b >= v_previous_index {
        return;
    }
    
    let feature_a = v_features[id_a];
    let feature_b = v_previous_features[id_b];

    var bad_bits = 0u;
    for (var i = 0u; i < 8u; i ++) {
        bad_bits += countOneBits(v_descriptors[id_a].data[i] ^ v_previous_descriptors[id_b].data[i]);
    }

    if bad_bits < 6u {
        let index = atomicAdd(&v_match_index, 1u);
        v_matches[index] = (global_id.x << 16) | global_id.y;
    }
}