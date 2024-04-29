@group(0) @binding(0)
var input_image: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @location(0) corner_position: vec2u,
) -> VertexOutput {
    var output: VertexOutput;

    let dim = vec2f(textureDimensions(input_image));

    let pos = vec2f(-1, -1) + vec2f(corner_position) / dim * 4.0;

    var vertices = array(
        // Triangle 1
        vec2f(-1.0, -1.0),
        vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0),
        // Triangle 2
        vec2f(-1.0, 1.0),
        vec2f(1.0, -1.0),
        vec2f(1.0, 1.0)
    );

    output.texcoord = vertices[vertex_index] * 0.5 + 0.5;

    let corner_size = 0.02;
    let corrected_pos = vertices[vertex_index] * dim.yx / f32(dim.x);

    output.position = vec4f(pos + corrected_pos * corner_size, 0.0, 1.0);

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {

    let t = input.texcoord * 2.0 - 1.0;
    let m = sqrt(dot(t, t));

    if m < 0.7 || m > 0.9 {
        discard;
    }

    return vec4f(input.texcoord, 0.0, 1.0);
}