

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

// @group(0) @binding(0) var<storage> gaussians: array<Gaussian>;
@group(0) @binding(1) var<storage, read_write> splats: array<Splat>;
@group(0) @binding(2) var<uniform> viewMat: mat4x4f;
@group(0) @binding(3) var<uniform> projMat: mat4x4f;
@group(0) @binding(4) var<uniform> focal: vec2f;
@group(0) @binding(5) var<uniform> tanFov: vec2f;
@group(0) @binding(6) var<uniform> screen: vec2u;
@group(0) @binding(7) var<uniform> camera: vec3f;

@group(1) @binding(0) var<storage, read_write> keys: array<u32>;
@group(1) @binding(1) var<storage, read_write> values: array<u32>;
@group(1) @binding(2) var<storage, read_write> size: u32;
@group(1) @binding(3) var<storage, read_write> prefix_sum: array<u32>;
@group(1) @binding(4) var<storage, read_write> range: array<vec2u>;
@group(1) @binding(5) var<storage, read_write> dispatch_sorter: vec3u;
@group(1) @binding(6) var<storage, read_write> dispatch_range: vec3u;
// @group(1) @binding(5) var<storage, read_write> dispatch_range: vec3u;

const HISTO_BLOCK_KVS = 3840u;
const WG_SIZE = 8u;

var<workgroup> section: array<u32, 128>;
@compute @workgroup_size(64)
fn compute_prefix_sum(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
  let index = workgroup_id.x * 128 + local_index;
  let n = arrayLength(&splats);

  if(index < n) {
    section[local_index] = splats[index].tiles;
  } 
  if(index + 64 < n) {
     section[local_index + 64] = splats[index + 64].tiles;
  } 

  for (var stride = 1u; stride <= 64; stride = stride << 1) {
        workgroupBarrier();
        let i = ((local_index + 1u) * stride * 2) - 1;
        if (i < 128) {
            section[i] += section[i - stride];
        }
    }

    for (var stride = 32u; stride > 0; stride = stride >> 1) {
        workgroupBarrier();
        let i = ((local_index + 1) * stride * 2) - 1;
        if (i + stride < 128) {
            section[i + stride] += section[i];
        }
    }

    workgroupBarrier();
    if (index < n) {
        prefix_sum[index] = section[local_index];
    }
    if (index + 64 < n) {
        prefix_sum[index + 64] = section[local_index + 64];
    }

    // if(index == n-1) {
    //     size = section[local_index];

    //     dispatch_sorter.x = (section[local_index] + HISTO_BLOCK_KVS - 1) / HISTO_BLOCK_KVS;
    //     dispatch_sorter.y = 1u;
    //     dispatch_sorter.z = 1u;

    //     let root = ceil(sqrt(f32(size)));
    //     let c = (u32(root) + WG_SIZE - 1) / WG_SIZE;
    //     dispatch_range.x = c;
    //     dispatch_range.y = c;
    //     dispatch_range.z = 1u;

    // }
    // if(index + 64 == n-1) {
    //     size = section[local_index + 64];

    //     dispatch_sorter.x = (section[local_index + 64] + HISTO_BLOCK_KVS - 1) / HISTO_BLOCK_KVS;
    //     dispatch_sorter.y = 1u;
    //     dispatch_sorter.z = 1u;

    //     let root = ceil(sqrt(f32(size)));
    //     let c = (u32(root) + WG_SIZE - 1) / WG_SIZE;
    //     dispatch_range.x = c;
    //     dispatch_range.y = c;
    //     dispatch_range.z = 1u;
    // }
}

@compute @workgroup_size(128)
fn finish_prefix_sum(@builtin(local_invocation_index) local_index: u32) {
    let n = arrayLength(&splats);
    let num_kernel = (n + 128 - 1) / 128;

    for(var i=1u; i<num_kernel; i++) {
        let offset = i * 128;
        let temp = prefix_sum[offset - 1];

        let index = offset + local_index;

        if(index < n) {
            prefix_sum[index] = prefix_sum[index] + temp;
        }
    }

    if(local_index == n % 128) {
        size = prefix_sum[n - 1];

        dispatch_sorter.x = (size + HISTO_BLOCK_KVS - 1) / HISTO_BLOCK_KVS;
        dispatch_sorter.y = 1u;
        dispatch_sorter.z = 1u;

        let root = ceil(sqrt(f32(size)));
        let c = (u32(root) + WG_SIZE - 1) / WG_SIZE;
        dispatch_range.x = c;
        dispatch_range.y = c;
        dispatch_range.z = 1u;
    }
}


@compute @workgroup_size(64)
fn copy_key_value(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let index = global_invocation_id.x;

    if(index >= arrayLength(&splats)) {
        return;
    }

    var offset = 0u;

    if(index > 0) {
        offset = prefix_sum[index - 1];
    }

    let splat = splats[index];
    let far = 1000.0;
    let depth = min(65535u, u32(splat.depth / far * 65535));

    let num_tile = (screen.x + 8 - 1) / 8;

    var i = 0u;
    for(var x = splat.min.x; x < splat.max.x; x++) {
        for(var y = splat.min.y; y < splat.max.y; y++) {
            let tileId = y * num_tile + x;
            let key = (tileId << 16) + depth;

            keys[offset + i] = key;
            values[offset + i] = index;

            i++;
        }
    }
}

@compute @workgroup_size(1)
fn sort_depth(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tile = workgroup_id.x;
    let tileRange = range[tile];

    for(var i=tileRange.x; i < tileRange.y - 1; i++) {
        let depth1 = splats[values[i]].depth;

        for(var j=i+1; j<tileRange.y; j++) {
            let depth2 = splats[values[j]].depth;

            if(depth1 < depth2) {
                let temp = values[i];
                values[i] = values[j];
                values[j] = temp;
            }
        }
    }
}
