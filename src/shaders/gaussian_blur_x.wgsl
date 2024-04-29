@group(0) @binding(0)
var texture_sampler: sampler;

@group(0) @binding(1)
var texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f
}

const SAMPLE_COUNT: u32 = 4u;

var<private> offsets: array<f32, SAMPLE_COUNT> = array(
    -2.2273038885157046,
    -0.4391873198428642,
    1.3243948342247673,
    3.0
);

var<private> weights: array<f32, SAMPLE_COUNT> = array(
    0.13748623236806098,
    0.5037756553768409,
    0.32748695702046415,
    0.031251155234634016
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