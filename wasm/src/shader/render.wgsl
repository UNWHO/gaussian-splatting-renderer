@group(0) @binding(6) var<uniform> screen: vec2u;
@group(0) @binding(8) var<storage, read_write> out: array<vec3f>;


@vertex
fn vert_main(@builtin(vertex_index)index : u32) -> @builtin(position) vec4f {
 var pos = array<vec2<f32>, 6>(
      vec2<f32>( 1.0,  1.0),
      vec2<f32>( 1.0, -1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>( 1.0,  1.0),
      vec2<f32>(-1.0, -1.0),
      vec2<f32>(-1.0,  1.0)
  );

  return vec4f(pos[index], 0, 1);
}

@fragment
fn frag_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let index = u32(pos.y) * screen.x + u32(pos.x);

    return vec4f(out[index], 1);
}