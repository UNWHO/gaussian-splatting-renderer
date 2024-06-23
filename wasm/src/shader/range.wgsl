@group(1) @binding(0) var<storage, read_write> keys: array<u32>;
@group(1) @binding(1) var<storage, read_write> values: array<u32>;
@group(1) @binding(2) var<storage, read_write> size: u32;
@group(1) @binding(3) var<storage, read_write> prefix_sum: array<u32>;
@group(1) @binding(4) var<storage, read_write> range: array<vec2u>;
@group(1) @binding(5) var<storage, read_write> dispatch_sorter: vec3u;
@group(1) @binding(6) var<storage, read_write> dispatch_range: vec3u;

@compute @workgroup_size(8, 8)
fn compute_range(
    @builtin(workgroup_id) workgroup_id : vec3<u32>, 
    @builtin(num_workgroups) num_workgroups: vec3<u32>, 
    @builtin(local_invocation_index) local_invocation_index: u32,
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let workgroup_index = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    let index = workgroup_index * 64 + local_invocation_index;

    if(index >= size) {
        return;
    }

    let currTile = keys[index] >> 16;

    if(index == 0) {
        range[currTile].x = 0u;
        return;
    }

    let prevTile = keys[index - 1] >> 16;
    if(prevTile != currTile) {
        range[currTile].x = index;
        range[prevTile].y = index;

        return;
    }

    if(index == size - 1) {
        range[currTile].y = size;
    }
}