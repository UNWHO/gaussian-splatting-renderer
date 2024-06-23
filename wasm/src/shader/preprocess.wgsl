struct Gaussian {
    mean: vec3f,
    norm: vec3f,
    sh: array<vec3f, 16>,
    scale: vec3f,
    opacity: f32,
    rotation: vec4f,
}

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


@group(0) @binding(0) var<storage> gaussians: array<Gaussian>;
@group(0) @binding(1) var<storage, read_write> splats: array<Splat>;
@group(0) @binding(2) var<uniform> viewMat: mat4x4f;
@group(0) @binding(3) var<uniform> projMat: mat4x4f;
@group(0) @binding(4) var<uniform> focal: vec2f;
@group(0) @binding(5) var<uniform> tanFov: vec2f;
@group(0) @binding(6) var<uniform> screen: vec2u;
@group(0) @binding(7) var<uniform> camera: vec3f;

// @group(1) @binding(0) var<storage, read_write> keys: array<u32>;
// @group(1) @binding(1) var<storage, read_write> values: array<u32>;
// @group(1) @binding(2) var<storage, read_write> size: atomic<u32>;
// @group(1) @binding(2) var<storage, read_write> prefix_sum: array<u32>;
// @group(1) @binding(2) var<storage, read_write> range: array<vec2u>;


const TILE = vec2u(8, 8);

const SH_C0 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const  SH_C2 = array<f32, 5>(
    1.0925484305920792,
	-1.0925484305920792,
	0.31539156525252005,
	-1.0925484305920792,
	0.5462742152960396
);
 const  SH_C3 = array<f32, 7>(
	-0.5900435899266435,
	2.890611442640554,
	-0.4570457994644658,
	0.3731763325901154,
	-0.4570457994644658,
	1.445305721320277,
	-0.5900435899266435
 );

 const far = f32(100.0);


@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup_id : vec3<u32>, 
    @builtin(num_workgroups) num_workgroups: vec3<u32>, 
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
    // @builtin(global_index) global_index: vec3<u32>
) {
    let global_index = global_invocation_id.x;

    if(global_index >= arrayLength(&gaussians)) {
        return;
    }

    
    splats[global_index].radius = 0.0f;
    splats[global_index].tiles = 0u;

    let gaussian = gaussians[global_index];
    
    // TODO: view frustrum culling


    let mean4 = projMat * vec4f(gaussian.mean, 1.0f);
    // let mean4 =  projMat * vec4f(asd, 1.0f);
    let w = 1.0f / (mean4.w + 0.0000001f);
    let mean3 = vec3(mean4.x * w, mean4.y * w, mean4.z * w);

    // compute 3d covariance
    var scaleMat = mat3x3f();
    scaleMat[0][0] = gaussian.scale.x;
    scaleMat[1][1] = gaussian.scale.y;
    scaleMat[2][2] = gaussian.scale.z;


    let r = gaussian.rotation.x;
	let x = gaussian.rotation.y;
	let y = gaussian.rotation.z;
	let z = gaussian.rotation.w;
    let rotMat = mat3x3f(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    let M = scaleMat * rotMat;
    

    let sigma3 = transpose(M) * M;
    let cov3 = array<f32, 6>(
        sigma3[0][0], sigma3[0][1], sigma3[0][2], 
        sigma3[1][1], sigma3[1][2], sigma3[2][2]
    );
    
    // splat to 2d covariance
    let viewMean = viewMat * vec4f(gaussian.mean, 1.0f);
    var mean = viewMean;

    // viewFrustrum culling
    if(viewMean.z <= 0.01 || viewMean.z > 1000.0) {
        return;
    }
    
    let lim = tanFov * 1.3f;
    let temp = mean.xy / mean.z;

    mean.x = min(lim.x, max(-lim.x, temp.x)) * mean.z;
    mean.y = min(lim.y, max(-lim.y, temp.y)) * mean.z;

    let J = mat3x3f(
        focal.x / mean.z, 0.0f, -(focal.x * mean.x) / (mean.z * mean.z),
		0.0f, focal.y / mean.z, -(focal.y * mean.y) / (mean.z * mean.z),
		0, 0, 0
    );

    let W = mat3x3f(
        viewMat[0][0], viewMat[1][0], viewMat[2][0],
		viewMat[0][1], viewMat[1][1], viewMat[2][1],
		viewMat[0][2], viewMat[1][2], viewMat[2][2]
    );

    let T = W * J;

    let VrK = mat3x3f(
        cov3[0], cov3[1], cov3[2],
		cov3[1], cov3[3], cov3[4],
		cov3[2], cov3[4], cov3[5]
    );

    let sigma2 = transpose(T) * VrK * T;
    let cov2 = vec3f(
        sigma2[0][0] + 0.3f, 
        sigma2[0][1], 
        sigma2[1][1] + 0.3f
    );

    let determinant = cov2.x * cov2.z - cov2.y * cov2.y;
    if(determinant == 0.0f) {
        return;
    }

    let conic = vec3f(cov2.z, -cov2.y, cov2.x) / determinant;

    // find tile
    let mid = 0.5f * (cov2.x + cov2.z);
    let lambda = vec2f(
        mid + sqrt(max(0.1f, mid * mid - determinant)),
        mid - sqrt(max(0.1f, mid * mid - determinant)),
    );
    let radius = ceil(3.0f * sqrt(max(lambda.x, lambda.y)));
    let pixel = vec2f(
        ((mean3.x + 1.0) * f32(screen.x) - 1.0) * 0.5,
        ((mean3.y + 1.0) * f32(screen.y) - 1.0) * 0.5,
    );

    let tileRange = vec2u(
        (screen.x + TILE.x - 1) / TILE.x,
        (screen.y + TILE.y - 1) / TILE.y,
    );

    let minTile = vec2u(
        min(tileRange.x, max(0u, u32(pixel.x - radius) / TILE.x)),
        min(tileRange.y, max(0u, u32(pixel.y - radius) / TILE.y))
    );
    let maxTile = vec2u(
        min(tileRange.x, max(0u, (u32(pixel.x + radius) + TILE.x - 1) / TILE.x)),
        min(tileRange.y, max(0u, (u32(pixel.y + radius) + TILE.y - 1) / TILE.y))
    );

    let touched = (maxTile.x - minTile.x) * (maxTile.y - minTile.y);
    if(touched == 0) {
        return;
    }

    // sh to rgb
    let dir = normalize(gaussian.mean - camera);
    var rgb = SH_C0 * gaussian.sh[0];

    {
        let x = dir.x; 
        let y = dir.y;
        let z = dir.z;

        rgb = 
            rgb -
            SH_C1 * dir.y * gaussian.sh[1] +
            SH_C1 * dir.z * gaussian.sh[2] -
            SH_C1 * dir.x * gaussian.sh[3];

        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let yz = y * z;
        let xz = x * z;

        rgb = rgb +
				SH_C2[0] * xy * gaussian.sh[4] +
				SH_C2[1] * yz * gaussian.sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * gaussian.sh[6] +
				SH_C2[3] * xz * gaussian.sh[7] +
				SH_C2[4] * (xx - yy) * gaussian.sh[8];

        rgb = rgb +
					SH_C3[0] * y * (3.0f * xx - yy) * gaussian.sh[9] +
					SH_C3[1] * xy * z * gaussian.sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * gaussian.sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * gaussian.sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * gaussian.sh[13] +
					SH_C3[5] * z * (xx - yy) * gaussian.sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * gaussian.sh[15];
    }

    rgb += 0.5;
    
    // let clamped = vec3u(
    //     u32(rgb.x < 0),
    //     u32(rgb.y < 0),
    //     u32(rgb.z < 0),
    // );

    rgb.x = max(rgb.x, 0.0);
    rgb.y = max(rgb.y, 0.0);
    rgb.z = max(rgb.z, 0.0);

  
    // save
    splats[global_index].radius = radius;
    splats[global_index].mean = pixel;
    splats[global_index].color = rgb;
    // splats[global_index].color_clamped = clamped;
    splats[global_index].cov = conic;
    splats[global_index].opacity = gaussian.opacity;
    
    splats[global_index].tiles = touched;
    // splats[global_index].tiles = 3u;
    splats[global_index].depth = viewMean.z;

    splats[global_index].min = minTile;
    splats[global_index].max = maxTile;

    // keys[global_index] = 
    // keys[global_index] = viewMean.z;
}
