
use std::{collections::HashMap, num::NonZeroU64};
use std::sync::Arc;

use wgpu::BufferUsages;

use crate::compute::{Compute, ComputeProgram};

const STORAGE: BufferUsages = BufferUsages::STORAGE;
const UNIFORM: BufferUsages = BufferUsages::UNIFORM;
const COPY_DST: BufferUsages = BufferUsages::COPY_DST;
const COPY_SRC: BufferUsages = BufferUsages::COPY_SRC;

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
    pub pipelines: HashMap<String, wgpu::ComputePipeline>,
    pub pipeline_bind_groups: HashMap<String, Vec<(u32, String)>>
}

impl OrbProgram {}

impl ComputeProgram for OrbProgram {
    type Config = OrbConfig;
    type Params = OrbParams;

    fn init(config: Self::Config, compute: Arc<Compute>) -> Self {
        
        let module = compute.device.create_shader_module(wgpu::include_wgsl!("shaders/orb_features_2.wgsl"));
        
        let mut buffers = HashMap::new();

        let max_features_per_chunk = 64;
        let num_chunks_x = (config.image_size.width + 7) / 8;
        let num_chunks_y = (config.image_size.height + 7) / 8;
        let num_chunks = num_chunks_x * num_chunks_y;

        // name, usage, size
        let buffer_info = [
            ("counter_reset", STORAGE | COPY_SRC, 4),
            ("input_image", STORAGE | COPY_DST | COPY_SRC, config.image_size.width * config.image_size.height * 4),
            ("input_image_size", UNIFORM | COPY_DST, 8),
            ("latest_corners", STORAGE | COPY_SRC, config.max_features * 8),
            ("chunk_corners", STORAGE | COPY_SRC, num_chunks * max_features_per_chunk * 8),
            ("chunk_counters", STORAGE | COPY_SRC | COPY_DST, num_chunks * 4),
            ("chunk_counters_reset", STORAGE | COPY_SRC, num_chunks * 4),
            ("chunk_counters_global", STORAGE | COPY_DST, num_chunks * 4),
            ("chunk_stride", STORAGE, 4),
            ("latest_descriptors", STORAGE | COPY_SRC, 256 * config.max_features),
            ("previous_corners", STORAGE | COPY_DST, config.max_features * 8),
            ("previous_corner_count", STORAGE | COPY_DST, 4),
            ("corner_matches", STORAGE, config.max_matches * 4),
            ("corner_match_counter", STORAGE | COPY_DST, 4),
            ("integral_image_in", STORAGE | COPY_DST, config.image_size.width * config.image_size.height * 4),
            ("integral_image_out", STORAGE | COPY_SRC, config.image_size.width * config.image_size.height * 4),
            ("integral_image_stride", STORAGE | COPY_DST, 4),
            ("integral_image_vis", STORAGE | COPY_SRC, config.image_size.width * config.image_size.height * 4),
            
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

        let mut bind_group_layouts = HashMap::new();
        let mut bind_groups = HashMap::new();

        // bind group name, ( binding, buffer name, min binding size, read only )
        let bind_group_info = [
            ("input_image", vec![
                (0, "input_image", 4, true),
                (1, "input_image_size", 8, true),
                (2, "latest_corners", 8, false),
                (3, "chunk_corners", 8 * max_features_per_chunk, false),
                (4, "chunk_counters", 4, false),
                (5, "chunk_counters_global", 4, false),
                (6, "chunk_stride", 4, false),
                (7, "latest_descriptors", 256, false)
            ]),
            ("integral_image", vec![
                (0, "integral_image_in", 4, false),
                (1, "integral_image_out", 4, false),
                (2, "integral_image_stride", 4, false),
                (3, "integral_image_vis", 4, false)
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

        // Add another bind group with the same layout, with with image_in and image_out flipped
        bind_groups.insert("integral_image_2".to_owned(), compute.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("integral_image_2"),
            layout: &bind_group_layouts["integral_image"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers["integral_image_out"].as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers["integral_image_in"].as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers["integral_image_stride"].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers["integral_image_vis"].as_entire_binding(),
                },
            ]
        }));
    
        let mut pipelines = HashMap::new();
        let mut pipeline_bind_groups = HashMap::new();

        // entry point, ( bind group names )
        let pipeline_info = [
            ("compute_grayscale", vec![
                "input_image", "integral_image"
            ]),
            ("increment_stride", vec![
                "input_image", "integral_image"
            ]),
            ("zero_stride", vec![
                "input_image", "integral_image"
            ]),
            ("compute_integral_image", vec![
                "input_image", "integral_image"
            ]),
            ("box_blur_visualization", vec![
                "input_image", "integral_image"
            ]),
            ("compute_fast_corners", vec![
                "input_image", "integral_image"
            ]),
            ("reset_chunk_stride", vec![
                "input_image", "integral_image"
            ]),
            ("increment_chunk_stride", vec![
                "input_image", "integral_image"
            ]),
            ("compute_integral_indices", vec![
                "input_image", "integral_image"
            ]),
            ("load_into_full_array", vec![
                "input_image", "integral_image"
            ]),
            ("compute_brief_descriptors", vec![
                "input_image", "integral_image"
            ]),
            // ("compute_matches", vec![
            //     "brief_descriptors", "feature_matches"
            // ])
        ];

        for (entry_point, bind_group_names) in pipeline_info {
            let mut bind_group_names_vec = Vec::new();

            for bind_group_name in bind_group_names {
                let mut j = 0;

                for i in 0..bind_group_info.len() {
                    if bind_group_info[i].0 == bind_group_name {
                        j = i;
                    }
                }

                bind_group_names_vec.push((j as u32, bind_group_name.to_owned()));
            }
            

            pipeline_bind_groups.insert(entry_point.to_owned(), bind_group_names_vec);

            let mut bind_group_layouts_vec = Vec::new();

            for i in 0..bind_group_info.len() {
                let name = bind_group_info[i].0;
                
                bind_group_layouts_vec.push(&bind_group_layouts[name]);
            }

            let pipeline_layout = compute.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry_point),
                bind_group_layouts: &bind_group_layouts_vec,
                push_constant_ranges: &[]
            });

            let pipeline = compute.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point
            });

            pipelines.insert(entry_point.to_owned(), pipeline);            

            
        }

        Self {
            config,
            module,
            buffers,
            bind_group_layouts,
            bind_groups,
            pipelines,
            pipeline_bind_groups
        }

    }

    fn run(&mut self, params: Self::Params, compute: Arc<Compute>) {
        
        let mut encoder = compute.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None
        });

        // Reset counters to zero
        encoder.copy_buffer_to_buffer(
            &self.buffers["counter_reset"],
            0,
            &self.buffers["corner_match_counter"],
            0,
            4
        );

        
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["compute_grayscale"]);

            cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
            cpass.set_bind_group(1, &self.bind_groups["integral_image"], &[]);

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

            cpass.set_pipeline(&self.pipelines["zero_stride"]);
            cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
            
            cpass.set_bind_group(1, &self.bind_groups["integral_image"], &[]);

            cpass.dispatch_workgroups(1, 1, 1);
        }

        let max_stride = u32::max(
            self.config.image_size.width,
            self.config.image_size.height
        ).ilog2();

        for i in 0..=max_stride {

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None
                });

                cpass.set_pipeline(&self.pipelines["increment_stride"]);
                cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
                
                cpass.set_bind_group(1, &self.bind_groups["integral_image"], &[]);

                cpass.dispatch_workgroups(1, 1, 1);
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None
                });

                cpass.set_pipeline(&self.pipelines["compute_integral_image"]);

                cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
                
                // Ping pong to avoid unnecessary copy
                cpass.set_bind_group(1, &self.bind_groups[if i % 2 == 0 { "integral_image" } else { "integral_image_2" } ], &[]);

                cpass.dispatch_workgroups(
                    (self.config.image_size.width + 7) / 8,
                    (self.config.image_size.height + 7) / 8,
                    1
                );
            }
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["box_blur_visualization"]);

            cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
            
            // Ping pong to avoid unnecessary copy
            cpass.set_bind_group(1, &self.bind_groups[if max_stride % 2 == 0 { "integral_image" } else { "integral_image_2" } ], &[]);

            cpass.dispatch_workgroups(
                (self.config.image_size.width + 7) / 8,
                (self.config.image_size.height + 7) / 8,
                1
            );
        }

        //// FEATURE EXTRACTION
        
        encoder.copy_buffer_to_buffer(
            &self.buffers["chunk_counters_reset"],
            0,
            &self.buffers["chunk_counters"],
            0,
            self.buffers["chunk_counters"].size()
        );

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["compute_fast_corners"]);

            cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
            
            // Ping pong to avoid unnecessary copy
            cpass.set_bind_group(1, &self.bind_groups[if max_stride % 2 == 0 { "integral_image" } else { "integral_image_2" } ], &[]);

            cpass.dispatch_workgroups(
                (self.config.image_size.width + 7) / 8,
                (self.config.image_size.height + 7) / 8,
                1
            );
        }

        encoder.copy_buffer_to_buffer(
            &self.buffers["chunk_counters"], 
            0,
            &self.buffers["chunk_counters_global"],
            0,
            self.buffers["chunk_counters"].size()
        );

        {

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["reset_chunk_stride"]);

            cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
            
            // Ping pong to avoid unnecessary copy
            cpass.set_bind_group(1, &self.bind_groups[if max_stride % 2 == 0 { "integral_image" } else { "integral_image_2" } ], &[]);

            cpass.dispatch_workgroups(
                1,
                1,
                1
            );
        }

        let num_chunks_x = (self.config.image_size.width + 7) / 8;
        let num_chunks_y = (self.config.image_size.height + 7) / 8;
        let num_chunks = num_chunks_x * num_chunks_y;
        let max_features_per_chunk = 64;

        let mut stride = 2;
        while stride < 2 * num_chunks {

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None
                });
    
                cpass.set_pipeline(&self.pipelines["compute_integral_indices"]);
    
                cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
                
                // Ping pong to avoid unnecessary copy
                cpass.set_bind_group(1, &self.bind_groups[if max_stride % 2 == 0 { "integral_image" } else { "integral_image_2" } ], &[]);
    
                cpass.dispatch_workgroups(
                    (num_chunks + 15) / 16,
                    1,
                    1
                );
            }

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None
                });
    
                cpass.set_pipeline(&self.pipelines["increment_chunk_stride"]);
    
                cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
                
                // Ping pong to avoid unnecessary copy
                cpass.set_bind_group(1, &self.bind_groups[if max_stride % 2 == 0 { "integral_image" } else { "integral_image_2" } ], &[]);
    
                cpass.dispatch_workgroups(
                    1,
                    1,
                    1
                );
            }

            stride *= 2;
        }

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.pipelines["load_into_full_array"]);
            cpass.set_bind_group(0, &self.bind_groups["input_image"], &[]);
                
            // Ping pong to avoid unnecessary copy
            cpass.set_bind_group(1, &self.bind_groups[if max_stride % 2 == 0 { "integral_image" } else { "integral_image_2" } ], &[]);
    
            cpass.dispatch_workgroups(
                (num_chunks + 7) / 8,
                (max_features_per_chunk + 7) / 8,
                1
            );
        }

        compute.queue.submit(Some(encoder.finish()));
    }
}