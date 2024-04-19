@group(0) @binding(0)
var<storage, read> latest_descriptors: array<array<u32, 8>>;

@group(0) @binding(1)
var<storage, read> previous_descriptors: array<array<u32, 8>>;

@group(0) @binding(2)
var<storage, read_write> matches: array<u32>;

@group(0) @binding(3)
var<storage, read_write> matches_counter: atomic<u32>;

var<workgroup> counter: atomic<u32>;
var<workgroup> workgroup_global_index: u32;
var<private> workgroup_index: u32;
var<private> is_match: u32;

@compute
@workgroup_size(8, 8, 1)
fn feature_matching(
    @builtin(global_invocation_id) global_id: vec3u
) {
    
}