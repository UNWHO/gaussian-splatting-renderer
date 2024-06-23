struct Splat {
    mean: vec2f,
    radius: f32,
    depth: f32,

    cov: vec3f,
    tiles: u32,

    color: vec3f,
    opacity: f32,
    

    min: vec2u,
    max: vec2u

    // color_clamped: vec3u,
}
const TILE = vec2u(8, 8);
const BG = vec3f(0.0);

@group(0) @binding(1) var<storage, read_write> splats: array<Splat>;
@group(0) @binding(6) var<uniform> screen: vec2u;
@group(0) @binding(8) var<storage, read_write> out: array<vec3f>;

@group(1) @binding(1) var<storage, read_write> values: array<u32>;
@group(1) @binding(4) var<storage, read_write> range: array<vec2u>;

@compute @workgroup_size(8, 8)
fn main(
    @builtin(workgroup_id) workgroup_id : vec3<u32>, 
    @builtin(num_workgroups) num_workgroups: vec3<u32>, 
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let pixel = global_invocation_id.xy;
    let inside = pixel.x < screen.x && pixel.y < screen.y;
    if(!inside) {
        return;
    }
    
    let pixel_id = pixel.y * screen.x + pixel.x;

    let tile = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    let tileRange = range[tile];

    var t = f32(1.0);
    var color = vec3f(0.0);

    for(var i=tileRange.x; i < tileRange.y; i++) {
        let index = values[i];
        let splat = splats[index];

        let distance = splat.mean.xy - vec2f(pixel);

        let power = 
            -0.5f * 
            (
                splat.cov.x * distance.x * distance.x +
                splat.cov.z * distance.y * distance.y
            )
            - splat.cov.y * distance.x * distance.y;

        if(power > 0.0) {
            continue;
        }
        

        let alpha = min(0.99, splat.opacity * exp(power));
        if (alpha < (1.0 / 255.0)) {
			continue;
        }
        let test_t = t * (1.0 - alpha);
        if(test_t < 0.0001) {
            break;
        }

        color = color + (splat.color * alpha * t);
        t = test_t;
    }


    out[pixel_id] = color + t * BG;
    

    return;
}