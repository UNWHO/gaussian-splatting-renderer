use std::{borrow::Cow, num::NonZeroU32, ops::Deref};

use js_sys::Math::sqrt;
use log::debug;
use nalgebra::{Matrix3x4, Matrix4, Matrix4x3, Rotation3, Translation3, Vector4};
use wasm_bindgen::prelude::*;
use web_sys::console::{self, debug};
use wgpu::{
    util::DeviceExt, BindGroupLayoutEntry, BindingType, ComputePipelineDescriptor, FragmentState,
    MultisampleState, PipelineLayoutDescriptor, PowerPreference, PrimitiveState,
    RenderPipelineDescriptor, ShaderStages, VertexAttribute, VertexBufferLayout, VertexState,
};
use wgpu_sort::{utils::guess_workgroup_size, GPUSorter};
use winit::{event_loop::EventLoop, window};

const SPLAT_SIZE: u64 = 64;
const NUM_SLPAT: u32 = 600000 * 2;
const WG_SIZE: u64 = 64;

const TILE_SZ: u32 = 8;

#[wasm_bindgen]
pub async fn render(gaussians: &[f32], num_gaussian: u64, cam_param: &[f32], size_param: &[u32]) {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");

    let SCREEN_X: u32 = size_param[0];
    let SCREEN_Y: u32 = size_param[1];

    let NUM_TILE_X: u32 = (SCREEN_X + TILE_SZ - 1) / TILE_SZ;
    let NUM_TILE_Y: u32 = (SCREEN_Y + TILE_SZ - 1) / TILE_SZ;
    let NUM_TILE: u32 = NUM_TILE_X * NUM_TILE_Y;

    let FOCAL_X: f32 = cam_param[9];
    let FOCAL_Y: f32 = cam_param[10];

    let event_loop = EventLoop::new().unwrap();
    let mut builder = window::WindowBuilder::new();

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    let window = builder.build(&event_loop).unwrap();

    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(&window).unwrap();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let mut limit = wgpu::Limits::downlevel_defaults();
    limit.max_buffer_size = 2147483640;
    limit.max_storage_buffer_binding_size = 2147483640;
    limit.max_compute_workgroup_storage_size = 32768;
    limit.max_storage_buffers_per_shader_stage = 10;

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: limit,
            },
            None,
        )
        .await
        .unwrap();

    let size = window.inner_size();
    let config = surface
        .get_default_config(&adapter, SCREEN_X, SCREEN_Y)
        .unwrap();
    surface.configure(&device, &config);

    // create buffer
    let gaussian_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Gaussian"),
        contents: bytemuck::cast_slice(gaussians),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let splat_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Splat"),
        size: num_gaussian * SPLAT_SIZE,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,

        mapped_at_creation: false,
    });

    let cam_position = nalgebra::Point3::<f32>::from_slice(&cam_param[0..3]);
    let cam_target = nalgebra::Point3::<f32>::from_slice(&cam_param[3..6]);
    let cam_up = nalgebra::Vector3::<f32>::from_column_slice(&cam_param[6..9]);

    let view_matrix = Matrix4::look_at_lh(&cam_position, &cam_target, &cam_up);

    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera"),
        contents: bytemuck::cast_slice(cam_position.coords.as_slice()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let view_matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewMat"),
        contents: bytemuck::cast_slice(view_matrix.as_slice()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let screen = nalgebra::Vector2::<u32>::new(SCREEN_X, SCREEN_Y);
    let screen_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("screen"),
        contents: bytemuck::cast_slice(screen.as_slice()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let fov = nalgebra::Vector2::new(
        // f32::to_radians(45.0), f32::to_radians(45.0)
        2.0 * f32::atan(SCREEN_X as f32 / (2.0 * FOCAL_X)),
        2.0 * f32::atan(SCREEN_Y as f32 / (2.0 * FOCAL_Y)),
    );
    let tan_fov = nalgebra::Vector2::new((fov.x * 0.5).tan(), (fov.y * 0.5).tan());
    let tan_fov_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("tan_fov"),
        contents: bytemuck::cast_slice(tan_fov.as_slice()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let focal = nalgebra::Vector2::<f32>::new(
        // screen.x as f32 / (2.0 * (fov.x * 0.5).tan()),
        // screen.y as f32 / (2.0 * (fov.y * 0.5).tan()),
        FOCAL_X, FOCAL_Y,
    );
    let focal_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("focal"),
        contents: bytemuck::cast_slice(focal.as_slice()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let near = 0.01;
    let far = 1000.0;
    let aspect = screen.x as f32 / screen.y as f32;
    let proj_matrix = nalgebra::Perspective3::<f32>::new(aspect, fov.y, near, far);
    let vp_matrix = proj_matrix.as_matrix() * view_matrix;

    let proj_matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("projection"),
        contents: bytemuck::cast_slice(vp_matrix.as_slice()),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: SCREEN_X as u64 * SCREEN_Y as u64 * 16,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("default bind group"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let sort_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sorter bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("Pipeline layout"),
        bind_group_layouts: &[&bind_group_layout, &sort_bind_group_layout],
        push_constant_ranges: &[],
    });

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader/preprocess.wgsl"))),
    });

    let preprocess_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Preprocess pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader/rasterize.wgsl"))),
    });

    let rasterize_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("rasterize pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("default bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gaussian_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: splat_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: view_matrix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: proj_matrix_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: focal_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: tan_fov_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: screen_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: camera_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // sorting
    let subgroup_size = guess_workgroup_size(&device, &queue).await.unwrap();
    let sorter = GPUSorter::new(&device, subgroup_size);

    let sort_buffers = sorter.create_sort_buffers(&device, NonZeroU32::new(NUM_SLPAT).unwrap());
    let sort_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sort size"),
        contents: bytemuck::cast_slice((0 as u32).to_ne_bytes().as_slice()),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    let prefix_sum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prefix sum buffer"),
        size: num_gaussian * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let range_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("range buffer"),
        contents: bytemuck::cast_slice(vec![0u32; NUM_TILE as usize * 8].as_slice()),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });

    let sort_dispatch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sort dispatch buffer"),
        size: 4 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });

    let range_dispatch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("range dispatch buffer"),
        size: 4 * 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::INDIRECT,
        mapped_at_creation: false,
    });

    let sort_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sort bind group"),
        layout: &sort_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sort_buffers.keys().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sort_buffers.values().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sort_size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: prefix_sum_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: range_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: sort_dispatch_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: range_dispatch_buffer.as_entire_binding(),
            },
        ],
    });

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("util copute shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader/util.wgsl"))),
    });

    let prefix_sum_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("prefix sum pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "compute_prefix_sum",
    });

    let finish_prefix_sum_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("finish prefix sum pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "finish_prefix_sum",
    });

    let copy_pair_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("copy_pair pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "copy_key_value",
    });

    // let sort_depth_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    //     label: Some("sort_depth pipeline"),
    //     layout: Some(&pipeline_layout),
    //     module: &cs_module,
    //     entry_point: "sort_depth",
    // });

    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("range compute shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader/range.wgsl"))),
    });
    let range_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("range pipeline"),
        layout: Some(&pipeline_layout),
        module: &cs_module,
        entry_point: "compute_range",
    });

    // render pipeline
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader/render.wgsl"))),
    });

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: Some("Render pipeline"),
        layout: Some(&pipeline_layout),
        vertex: VertexState {
            module: &cs_module,
            entry_point: "vert_main",
            buffers: &[],
        },
        fragment: Some(FragmentState {
            module: &cs_module,
            entry_point: "frag_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
    });

    let frame = surface
        .get_current_texture()
        .expect("Failed to acquire next swap chain texture");

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: 4 as u64,
        // size: SPLAT_SIZE * num_gaussian,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("preprocessor encoder"),
    });

    // preprocess
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&preprocess_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_bind_group(1, &sort_bind_group, &[]);
        pass.dispatch_workgroups(((num_gaussian + WG_SIZE - 1) / WG_SIZE) as u32, 1, 1);
    }

    // prefix sum
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&prefix_sum_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_bind_group(1, &sort_bind_group, &[]);
        pass.dispatch_workgroups(
            ((num_gaussian + WG_SIZE * 2 - 1) / (WG_SIZE * 2)) as u32,
            1,
            1,
        );
    }

    // finish prefix sum
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&finish_prefix_sum_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_bind_group(1, &sort_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // copy key-value pair
    {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&copy_pair_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_bind_group(1, &sort_bind_group, &[]);
            pass.dispatch_workgroups(((num_gaussian + WG_SIZE - 1) / WG_SIZE) as u32, 1, 1);
        }
    }

    // sort
    encoder.copy_buffer_to_buffer(&sort_size_buffer, 0, sort_buffers.state_buffer(), 0, 4);
    sorter.sort_indirect(&mut encoder, &sort_buffers, &sort_dispatch_buffer);
    // sorter.sort(&mut encoder, &queue, &sort_buffers, None);

    // compute range
    {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&range_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_bind_group(1, &sort_bind_group, &[]);

            let size = sqrt(NUM_SLPAT as f64).ceil() as u64;
            let size = ((size + 8 - 1) / 8) as u32;

            // pass.dispatch_workgroups_indirect(&range_dispatch_buffer, 0);

            pass.dispatch_workgroups(size, size, 1);
        }
    }

    // rasterize
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&rasterize_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_bind_group(1, &sort_bind_group, &[]);
        pass.dispatch_workgroups(NUM_TILE_X, NUM_TILE_Y, 1);
    }
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_bind_group(1, &sort_bind_group, &[]);
        pass.set_pipeline(&render_pipeline);
        pass.draw(0..6, 0..1);
    }

    encoder.copy_buffer_to_buffer(&sort_size_buffer, 0, &staging_buffer, 0, 4 as u64);
    queue.submit(Some(encoder.finish()));
    frame.present();

    device.poll(wgpu::Maintain::Wait);

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::Maintain::Wait);
    receiver
        .await
        .expect("communicaiton failed")
        .expect("buffer reading failed");

    buffer_slice
        .get_mapped_range()
        .chunks((8) as usize)
        .for_each(|chunk| {
            let mut jsArr: [JsValue; 4] = [1.into(), 1.into(), 1.into(), 1.into()];
            for (i, _) in [0].iter().enumerate() {
                let bytes = &chunk[i * 4..i * 4 + 4];
                let f = u32::from_ne_bytes(bytes.try_into().expect("dafs"));
                let js: JsValue = f.into();

                jsArr[i] = js;
                // console::log_2(&label[i].into(), &js);
            }

            console::log_1(&jsArr[0]);
        });

    staging_buffer.unmap();
}
