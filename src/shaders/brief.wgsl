struct Feature {
    x: u32,
    y: u32,
    angle: f32,
    octave: u32
}

@group(0) @binding(0)
var<storage, read> corners: array<Feature>;

@group(0) @binding(1)
var<storage, read> counter: u32;

@group(0) @binding(2)
var<storage, read_write> descriptors: array<array<u32, 8>>;

@group(0) @binding(3)
var blur_hierarchy: texture_2d<f32>;

@compute
@workgroup_size(8, 8, 1)
fn brief(
    @builtin(global_invocation_id) global_id: vec3u
) {
    let feature_id = global_id.y;

    if feature_id >= counter {
        return;
    }

    let corner = corners[feature_id];
    let octave = i32(corner.octave);

    let pos = vec2i(i32(corner.x), i32(corner.y));
    let angle = corner.angle;
    let ct = cos(angle);
    let st = sin(angle);
    let rotation_matrix = mat2x2f(
        ct, -st,
        st, ct
    );

    var bits = 0u;

    for (var i = 0u; i < 32u; i ++) {

        let descriptor_index = global_id.x << 5u | i;
        let descriptor_info = brief_descriptors[descriptor_index];

        let unrotated_point_a = vec2f(descriptor_info.xy);
        let unrotated_point_b = vec2f(descriptor_info.zw);
        
        let rotated_point_a = rotation_matrix * unrotated_point_a;
        let rotated_point_b = rotation_matrix * unrotated_point_b;

        let texel_a = vec2i(rotated_point_a) + pos;
        let texel_b = vec2i(rotated_point_b) + pos;

        let value_a = textureLoad(blur_hierarchy, texel_a, octave);
        let value_b = textureLoad(blur_hierarchy, texel_b, octave);

        if value_a.x > value_b.x {
            bits |= 1u << i;
        }
    }

    descriptors[feature_id][global_id.x] = bits;
}

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

