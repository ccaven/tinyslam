@group(0) @binding(0)
var<storage, read> latest_descriptors: array<array<u32, 8>>;

@group(0) @binding(1)
var<storage, read> previous_descriptors: array<array<u32, 8>>;

@group(0) @binding(2)
var<storage, read> latest_corners_counter: u32;

@group(0) @binding(3)
var<storage, read> previous_corners_counter: u32;

@group(0) @binding(4)
var<storage, read_write> feature_matches: array<atomic<u32>>;

var<workgroup> workgroup_score: atomic<u32>;

@compute
@workgroup_size(1, 64, 1)
fn feature_matching(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_index) local_index: u32
) { 
    if (global_id.x >= latest_corners_counter || global_id.y >= previous_corners_counter) {
        return;
    }

    var bad_bits = 0u;
    for (var i = 0u; i < 8u; i ++) {
        bad_bits += countOneBits(latest_descriptors[global_id.x][i] ^ previous_descriptors[global_id.y][i]);
    }
    let score = 256u - bad_bits;

    let item = (score << 16u) | global_id.y;

    atomicMax(&workgroup_score, item);

    workgroupBarrier();

    if local_index == 0 {
        atomicMax(&feature_matches[global_id.x], workgroup_score);
    }
}