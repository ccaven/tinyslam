@group(0) @binding(0)
var<storage, read> latest_descriptors: array<array<u32, 8>>;

@group(0) @binding(1)
var<storage, read> previous_descriptors: array<array<u32, 8>>;

@group(0) @binding(2)
var<storage, read> latest_corners_counter: u32;

@group(0) @binding(3)
var<storage, read> previous_corners_counter: u32;

@group(0) @binding(4)
var<storage, read_write> feature_matches: array<vec2u>;

@group(0) @binding(5)
var<storage, read_write> feature_matches_counter: atomic<u32>;

@compute
@workgroup_size(8, 8, 1)
fn feature_matching(
    @builtin(global_invocation_id) global_id: vec3u
) { 
    if (global_id.x >= latest_corners_counter || global_id.y >= previous_corners_counter) {
        return;
    }

    // Check for matches
    var is_match = false;
    var workgroup_index: u32;

    let threshold = 64u;
    var bad_bits = 0u;
    for (var i = 0u; i < 8u; i ++) {
        bad_bits += countOneBits(latest_descriptors[global_id.x][i] ^ previous_descriptors[global_id.y][i]);
    }
    if bad_bits < threshold {
        // It is a match!
        let index = atomicAdd(&feature_matches_counter, 1u);
        feature_matches[index] = global_id.xy;
    }
}