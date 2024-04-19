@group(0) @binding(0)
var texture_sampler: sampler;

@group(0) @binding(1)
var texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f
}

const SAMPLE_COUNT: u32 = 6u;

var<private> offsets: array<f32, SAMPLE_COUNT> = array(
    -4.378621204796657,
    -2.431625915613778,
    -0.4862426846689485,
    1.4588111840004858,
    3.4048471718931532,
    5.0
);

var<private> weights: array<f32, SAMPLE_COUNT> = array(
    0.09461172151436463,
    0.20023097066826712,
    0.2760751120037518,
    0.24804559825032563,
    0.14521459357563646,
    0.035822003987654526
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32
) -> VertexOutput {
    var vertices = array(
        vec2f(-1.0, 3.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, -1.0)
    );

    let position = vertices[vertex_index];
    var output: VertexOutput;
    output.position = vec4f(position, 0.0, 1.0);
    output.texcoord = position * 0.5 + 0.5;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let color = textureSample(
        texture,
        texture_sampler,
        input.texcoord
    );

    var result = vec4f(0.0);
    for (var i = 0u; i < SAMPLE_COUNT; i ++) {
        let offset = vec2f(offsets[i], 0.0);
        let weight = weights[i];
        result += textureSample(texture, texture_sampler, input.texcoord + offset) * weight;
    }
    
    return result;
}