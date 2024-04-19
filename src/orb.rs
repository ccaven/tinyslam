// TODO
//  - Convert input_image to a texture<f32>
//  - Merge orb.buffers["input_image"] with vis.base_texture (literally combine)
//  - Implement feature matching + visualization
//  - Make sure bit FAST checker is actually faster than manual
//  - Try different camera / try to turn off autofocus?
//  - (Optional) Replace output_viz with storage texture

use std::{collections::HashMap, num::NonZeroU64};
use std::sync::Arc;

use crate::compute::ComputeProgram;
use tiny_wgpu::Compute;

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE;
const UNIFORM: wgpu::BufferUsages = wgpu::BufferUsages::UNIFORM;
const COPY_DST: wgpu::BufferUsages = wgpu::BufferUsages::COPY_DST;
const COPY_SRC: wgpu::BufferUsages = wgpu::BufferUsages::COPY_SRC;

const TEXTURE_BINDING: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING;
const TEXTURE_COPY_DST: wgpu::TextureUsages = wgpu::TextureUsages::COPY_DST;
const TEXTURE_COPY_SRC: wgpu::TextureUsages = wgpu::TextureUsages::COPY_SRC;

const RGBA8_UNORM: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct OrbConfig {
    pub image_size: wgpu::Extent3d,
    pub max_features: u32,
    pub max_matches: u32
}

pub struct OrbParams {
    pub compute_matches: bool
}

pub struct OrbProgram {
    pub config: OrbConfig,
    pub module: wgpu::ShaderModule,
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub bind_groups: HashMap<String, wgpu::BindGroup>,
    pub bind_group_layouts: HashMap<String, wgpu::BindGroupLayout>,
    pub pipelines: HashMap<String, wgpu::ComputePipeline>
}

impl OrbProgram {}

impl ComputeProgram for OrbProgram {
    type Config = OrbConfig;
    type Params = OrbParams;

    fn init(config: Self::Config, compute: Arc<Compute>) -> Self {
        
        let module = compute.device.create_shader_module(wgpu::include_wgsl!("shaders/orb.wgsl"));
        
        let mut buffers = HashMap::new();

        let bytes_per_feature = 3 * 4;

        // name, usage, size
        let buffer_info = [
            ("counter_reset", STORAGE | COPY_SRC, 4),
            ("input_image", STORAGE | COPY_DST | COPY_SRC, config.image_size.width * config.image_size.height * 4),
            ("input_image_size", UNIFORM | COPY_DST, 8),
            
            ("latest_corners", STORAGE | COPY_SRC, config.max_features * bytes_per_feature),
            ("previous_corners", STORAGE | COPY_DST, config.max_features * bytes_per_feature),

            ("latest_descriptors", STORAGE | COPY_SRC, 256 * config.max_features),
            ("previous_descriptors", STORAGE | COPY_DST, 256 * config.max_features),

            ("latest_corners_counter", STORAGE | COPY_DST | COPY_SRC, 4),
            ("previous_corners_counter", STORAGE | COPY_DST, 4),

            ("corner_matches", STORAGE, config.max_matches * 4),
            ("corner_match_counter", STORAGE | COPY_DST, 4),

            ("integral_image_out", STORAGE | COPY_SRC, config.image_size.width * config.image_size.height * 4),
            ("integral_image_stride", STORAGE | COPY_DST, 4),
            ("integral_image_vis", STORAGE | COPY_SRC | COPY_DST, config.image_size.width * config.image_size.height * 4),

            ("integral_image_precomputed_strides", STORAGE | COPY_SRC | COPY_DST, 4 * 32)
        ];

        for (name, usage, size) in buffer_info {
            let buffer = compute.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: size as u64,
                usage: usage,
                mapped_at_creation: false
            });

            buffers.insert(name.to_owned(), buffer);
        }

        let mut textures = HashMap::new();
        let texture_info = [
            ("input_image", config.image_size, TEXTURE_BINDING | TEXTURE_COPY_DST, RGBA8_UNORM)
        ];

        for (name, size, usage, format) in texture_info {
            let label = Some(name);
            let texture = compute.device.create_texture(&wgpu::TextureDescriptor {
                size,
                usage,
                format,
                label,
                dimension: wgpu::TextureDimension::D2,
                mip_level_count: 1,
                sample_count: 1,
                view_formats: &[]
            });

            textures.insert(name, texture);
        }

        let mut bind_group_layouts = HashMap::new();
        let mut bind_groups = HashMap::new();

        struct BindGroupInfo<'a> {
            name: &'a str,
            buffers: Vec<(u32, &'a str, u32, bool)>,
            textures: Vec<(u32, &'a str)>
        }

        let bind_group_infos = [
            BindGroupInfo {
                name: "main",
                buffers: vec![
                    (0, "input_image", 4, true),
                    (1, "input_image_size", 8, true),
                    (2, "latest_corners", bytes_per_feature, false),
                    (3, "latest_descriptors", 256, false),
                    (4, "latest_corners_counter", 4, false),
                    (5, "integral_image_out", 4, false),
                    (6, "integral_image_stride", 4, false),
                    (7, "integral_image_vis", 4, false),
                    (8, "previous_corners", bytes_per_feature, true),
                    (9, "previous_descriptors", 256, true),
                    (10, "previous_corners_counter", 4, true)
                ],
                textures: vec![
                    (0, "input_image")
                ]
            }
        ];

        for bind_group_info in bind_group_infos.iter() {

            

            for (binding, name, min_binding_size, read_only) in bind_group_info.buffers.iter() {

            }

            for (binding, name) in bind_group_info.textures.iter() {

            }
        }

        // bind group name, ( binding, buffer name, min binding size, read only )
        let bind_group_info = [
            ("main", vec![
                (0, "input_image", 4, true),
                (1, "input_image_size", 8, true),
                (2, "latest_corners", bytes_per_feature, false),
                (3, "latest_descriptors", 256, false),
                (4, "latest_corners_counter", 4, false),
                (5, "integral_image_out", 4, false),
                (6, "integral_image_stride", 4, false),
                (7, "integral_image_vis", 4, false),
                (8, "previous_corners", bytes_per_feature, true),
                (9, "previous_descriptors", 256, true),
                (10, "previous_corners_counter", 4, true),
            ]),
        ];

        for i in 0..bind_group_info.len() {

            let (group_name, entries) = &bind_group_info[i];

            let group_name = group_name.to_owned();

            let mut bind_group_layout_entries = Vec::<wgpu::BindGroupLayoutEntry>::new();
            let mut bind_group_entries = Vec::<wgpu::BindGroupEntry>::new();

            for (binding, buffer_name, min_binding_size, read_only) in entries {

                let buffer_name = buffer_name.to_owned();
                let read_only = read_only.to_owned();
                let binding = binding.to_owned();
                let min_binding_size = min_binding_size.to_owned();

                let ty = if buffers[buffer_name].usage().contains(UNIFORM) {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only }
                };

                bind_group_layout_entries.push(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { 
                        ty, 
                        has_dynamic_offset: false, 
                        min_binding_size: Some(NonZeroU64::new(min_binding_size.into()).unwrap())
                    },
                    count: None
                });

                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: buffers[buffer_name].as_entire_binding()
                });
            }

            let bind_group_layout = compute.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &bind_group_layout_entries,
                label: Some(group_name)
            });

            bind_groups.insert(group_name.to_owned(), compute.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(group_name),
                layout: &bind_group_layout,
                entries: &bind_group_entries
            }));

            bind_group_layouts.insert(group_name.to_owned(), bind_group_layout);

        }
    
        let mut pipelines = HashMap::new();

        // entry point, ( bind group names )
        let pipeline_info = [
            "compute_grayscale",
            "compute_integral_image_x",
            "compute_integral_image_y",
            "visualize_box_blur",
            "compute_fast_corners",
            "visualize_features",
            "compute_brief_descriptors",
            "match_features"
        ];

        let pipeline_layout = compute.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Generic pipeline layout"),
            bind_group_layouts: &[&bind_group_layouts["main"]],
            push_constant_ranges: &[
                wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..4
                }
            ]
        });

        for entry_point in pipeline_info {
            let pipeline = compute.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point
            });

            pipelines.insert(entry_point.to_owned(), pipeline);              
        }

        {
            let precomputed_strides = (1u32..=32).collect::<Vec<u32>>();
            compute.queue.write_buffer(
                &buffers["integral_image_precomputed_strides"], 
                0, 
                bytemuck::cast_slice(&precomputed_strides[..])
            );
        }

        Self {
            config,
            module,
            buffers,
            bind_group_layouts,
            bind_groups,
            pipelines
        }

    }

    fn run(&mut self, params: Self::Params, compute: Arc<Compute>) {
        
        let mut encoder = compute.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["compute_grayscale"]);

            cpass.set_bind_group(0, &self.bind_groups["main"], &[]);

            cpass.dispatch_workgroups(
                (self.config.image_size.width + 7) / 8,
                (self.config.image_size.height + 7) / 8,
                1
            );
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["compute_integral_image_x"]);
            cpass.set_bind_group(0, &self.bind_groups["main"], &[]);

            let stride_x = self.config.image_size.width.ilog2() + 2;
            for i in 2..=stride_x {
                cpass.set_push_constants(0, bytemuck::cast_slice(&[i]));
                cpass.dispatch_workgroups(
                    (self.config.image_size.width / 2 + 15) / 16,
                    (self.config.image_size.height + 3) / 4,
                    1
                );
            }
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["compute_integral_image_y"]);
            cpass.set_bind_group(0, &self.bind_groups["main"], &[]);

            let stride_y = self.config.image_size.height.ilog2() + 2;
            for i in 2..=stride_y {
                cpass.set_push_constants(0, bytemuck::cast_slice(&[i]));
                cpass.dispatch_workgroups(
                    (self.config.image_size.width + 7) / 8,
                    (self.config.image_size.height / 2 + 7) / 8,
                    1
                );
            }
        }
        
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["visualize_box_blur"]);
            cpass.set_bind_group(0, &self.bind_groups["main"], &[]);
            cpass.dispatch_workgroups(
                (self.config.image_size.width + 7) / 8,
                (self.config.image_size.height + 7) / 8,
                1
            );
        }

        encoder.copy_buffer_to_buffer(&self.buffers["counter_reset"], 0, &self.buffers["latest_corners_counter"], 0, 4);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["compute_fast_corners"]);
            cpass.set_bind_group(0, &self.bind_groups["main"], &[]);
            cpass.dispatch_workgroups(
                (self.config.image_size.width + 7) / 8,
                (self.config.image_size.height + 7) / 8,
                1
            );
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            }); 

            cpass.set_pipeline(&self.pipelines["compute_brief_descriptors"]);
            cpass.set_bind_group(0, &self.bind_groups["main"], &[]);
            cpass.dispatch_workgroups(
                (self.config.max_features + 1) / 2, // one per 2 features
                8, // one per u32
                1
            );
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            }); 
            cpass.set_pipeline(&self.pipelines["match_features"]);
            cpass.set_bind_group(0, &self.bind_groups["main"], &[]);
            cpass.dispatch_workgroups(
                (self.config.max_features + 7) / 8,
                (self.config.max_features + 7) / 8,
                1
            );
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            }); 
            cpass.set_pipeline(&self.pipelines["visualize_features"]);
            cpass.set_bind_group(0, &self.bind_groups["main"], &[]);
            cpass.dispatch_workgroups(
                (self.config.max_features + 63) / 64,
                1,
                1
            );
        }

        // Copy data to previous frame buffers
        encoder.copy_buffer_to_buffer(
            &self.buffers["latest_descriptors"],
            0,
            &self.buffers["previous_descriptors"],
            0,
            self.buffers["latest_descriptors"].size()
        );

        encoder.copy_buffer_to_buffer(
            &self.buffers["latest_corners"],
            0,
            &self.buffers["previous_corners"],
            0,
            self.buffers["latest_corners"].size()
        );

        encoder.copy_buffer_to_buffer(
            &self.buffers["latest_corners_counter"], 
            0, 
            &self.buffers["previous_corners_counter"], 
            0, 
            self.buffers["latest_corners_counter"].size()
        );

        compute.queue.submit(Some(encoder.finish()));
    }
}