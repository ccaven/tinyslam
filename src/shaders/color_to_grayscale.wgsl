@group(0) @binding(0)
var texture_sampler: sampler;

@group(0) @binding(1)
var texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f
}

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
    
    let rgba_coefs = vec4f(0.229, 0.587, 0.114, 0.0);
    let gray = dot(color, rgba_coefs);
    return vec4f(gray, 0.0, 0.0, 0.0);
}