/*
TODO: 
 - Render blur texture using a normal vertex/fragment shader, samplers, two passes
 - Should get performance from 1.0ms to 0.5ms
*/

use wgpu::{
    BufferUsages, TextureUsages
};

use tiny_wgpu::{
    Compute,
    Storage,
    ComputeProgram,
    BindGroupItem
};

pub struct OrbConfig {
    pub image_size: wgpu::Extent3d,
    pub max_features: u32,
    pub max_matches: u32
}

pub struct OrbParams {
    pub record_keyframe: bool
}

pub struct OrbProgram<'a> {
    pub config: OrbConfig,
    pub compute: Compute,
    pub storage: Storage<'a>
}

impl<'a> ComputeProgram<'a> for OrbProgram<'a> {
    fn compute(&self) -> &Compute {
        &self.compute
    }
    fn storage(&self) -> &Storage<'a> {
        &self.storage
    }
    fn storage_mut(&mut self) -> &mut Storage<'a> {
        &mut self.storage
    }
}

impl OrbProgram<'_> {
    pub fn init(&mut self) {

        self.add_module("color_to_grayscale", wgpu::include_wgsl!("shaders/color_to_grayscale.wgsl"));
        self.add_module("gaussian_blur_x", wgpu::include_wgsl!("shaders/gaussian_blur_x.wgsl"));
        self.add_module("gaussian_blur_y", wgpu::include_wgsl!("shaders/gaussian_blur_y.wgsl"));
        self.add_module("corner_detector", wgpu::include_wgsl!("shaders/corner_detector.wgsl"));
        self.add_module("feature_descriptors", wgpu::include_wgsl!("shaders/feature_descriptors.wgsl"));
        self.add_module("feature_matching", wgpu::include_wgsl!("shaders/feature_matching.wgsl"));
        
        self.add_module("corner_visualization", wgpu::include_wgsl!("shaders/corner_visualization.wgsl"));
        self.add_module("matches_visualization", wgpu::include_wgsl!("shaders/matches_visualization.wgsl"));

        self.add_module("corner_visualization_2", wgpu::include_wgsl!("shaders/corner_visualization_2.wgsl"));

        self.add_texture(
            "visualization",
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
            wgpu::TextureFormat::Rgba8Unorm,
            self.config.image_size
        );

        self.add_texture(
            "input_image", 
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC, 
            wgpu::TextureFormat::Rgba8Unorm, 
            self.config.image_size
        );
        
        self.add_sampler(
            "input_image_sampler",
            wgpu::SamplerDescriptor {
                label: None,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                lod_max_clamp: 1.0,
                lod_min_clamp: 0.0,
                compare: None,
                anisotropy_clamp: 1,
                border_color: None
            }
        );

        let half_size = wgpu::Extent3d { 
            width: self.config.image_size.width / 2, 
            height: self.config.image_size.height / 2, 
            depth_or_array_layers: 1
        };

        self.add_texture(
            "grayscale_image",
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            wgpu::TextureFormat::R16Float,
            half_size
        );

        self.add_texture(
            "gaussian_blur_x",
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            wgpu::TextureFormat::R16Float,
            half_size
        );

        self.add_texture(
            "gaussian_blur",
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            wgpu::TextureFormat::R16Float,
            half_size
        );

        self.add_buffer(
            "latest_corners",
            BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            (self.config.max_features * 8) as u64
        );

        self.add_buffer(
            "latest_corners_counter",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            4
        );

        self.add_staging_buffer("latest_corners_counter");

        self.add_buffer(
            "previous_corners",
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            (self.config.max_features * 8) as u64
        );

        self.add_buffer(
            "previous_corners_counter",
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            4
        );

        self.add_buffer(
            "feature_matches",
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            (self.config.max_features * 4) as u64
        );

        self.add_buffer(
            "latest_descriptors",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            (self.config.max_features * 8 * 4) as u64
        );

        self.add_buffer(
            "previous_descriptors",
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
            (self.config.max_features * 8 * 4) as u64
        );

        // Stage 1: Color to grayscale
        {
            self.add_bind_group("color_to_grayscale", &[
                BindGroupItem::Sampler { label: "input_image_sampler" },
                BindGroupItem::Texture { label: "input_image" }
            ]);

            self.add_render_pipelines(
                "color_to_grayscale", 
                &["color_to_grayscale"], 
                &[("color_to_grayscale", ("vs_main", "fs_main"))],
                &[],
                &[Some(wgpu::TextureFormat::R16Float.into())], 
                &[]
            );
        }

        // Stage 2: Gaussian blur x
        {
            self.add_bind_group("gaussian_blur_x", &[
                BindGroupItem::Sampler { label: "input_image_sampler" },
                BindGroupItem::Texture { label: "grayscale_image" }
            ]);

            self.add_render_pipelines(
                "gaussian_blur_x", 
                &["gaussian_blur_x"], 
                &[("gaussian_blur_x", ("vs_main", "fs_main"))],
                &[],
                &[Some(wgpu::TextureFormat::R16Float.into())], 
                &[]
            );
        }
        
        // Stage 3: Gaussian blur y
        {
            self.add_bind_group("gaussian_blur_y", &[
                BindGroupItem::Sampler { label: "input_image_sampler" },
                BindGroupItem::Texture { label: "gaussian_blur_x" }
            ]);

            self.add_render_pipelines(
                "gaussian_blur_y", 
                &["gaussian_blur_y"],
                &[("gaussian_blur_y", ("vs_main", "fs_main"))],
                &[],
                &[Some(wgpu::TextureFormat::R16Float.into())], 
                &[]
            );
        }

        // Stage 4: Corner detection
        {
            self.add_bind_group("corner_detector", &[
                BindGroupItem::Texture { label: "grayscale_image" },
                BindGroupItem::StorageBuffer { label: "latest_corners", min_binding_size: 8, read_only: false },
                BindGroupItem::StorageBuffer { label: "latest_corners_counter", min_binding_size: 4, read_only: false }
            ]);

            self.add_compute_pipelines("corner_detector", &["corner_detector"], &["corner_detector"], &[]);
        }

        // Corner visualization
        {
            self.add_bind_group("corner_visualization", &[
                BindGroupItem::StorageTexture { label: "visualization", access: wgpu::StorageTextureAccess::WriteOnly },
                BindGroupItem::StorageBuffer { label: "latest_corners", min_binding_size: 8, read_only: true },
                BindGroupItem::StorageBuffer { label: "latest_corners_counter", min_binding_size: 4, read_only: true },
                BindGroupItem::Texture { label: "gaussian_blur" },
                BindGroupItem::Texture { label: "grayscale_image" },
            ]);

            self.add_compute_pipelines("corner_visualization", &["corner_visualization"], &["corner_visualization"], &[]);
        }

        // Stage 5: Feature descriptors
        {
            self.add_bind_group("feature_descriptors", &[
                BindGroupItem::Texture { label: "gaussian_blur" },
                BindGroupItem::StorageBuffer { label: "latest_corners", min_binding_size: 8, read_only: true },
                BindGroupItem::StorageBuffer { label: "latest_corners_counter", min_binding_size: 4, read_only: true },
                BindGroupItem::StorageBuffer { label: "latest_descriptors", min_binding_size: 8 * 4, read_only: false },
                BindGroupItem::Texture { label: "grayscale_image" }
            ]);

            self.add_compute_pipelines("feature_descriptors", &["feature_descriptors"], &["feature_descriptors"], &[]);
        }

        // Stage 6: Feature matching
        {
            self.add_bind_group("feature_matching", &[
                BindGroupItem::StorageBuffer { label: "latest_descriptors", min_binding_size: 8 * 4, read_only: true },
                BindGroupItem::StorageBuffer { label: "previous_descriptors", min_binding_size: 8 * 4, read_only: true },
                BindGroupItem::StorageBuffer { label: "latest_corners_counter", min_binding_size: 4, read_only: true },
                BindGroupItem::StorageBuffer { label: "previous_corners_counter", min_binding_size: 4, read_only: true },
                BindGroupItem::StorageBuffer { label: "feature_matches", min_binding_size: 8, read_only: false }
            ]);
            
            self.add_compute_pipelines("feature_matching", &["feature_matching"], &["feature_matching"], &[]);
        }

        {
            self.add_bind_group("matches_visualization", &[
                BindGroupItem::StorageTexture { label: "visualization", access: wgpu::StorageTextureAccess::WriteOnly },
                BindGroupItem::StorageBuffer { label: "latest_corners", min_binding_size: 8, read_only: true },
                BindGroupItem::StorageBuffer { label: "latest_corners_counter", min_binding_size: 4, read_only: true },
                BindGroupItem::StorageBuffer { label: "previous_corners", min_binding_size: 8, read_only: true },
                BindGroupItem::StorageBuffer { label: "previous_corners_counter", min_binding_size: 4, read_only: true },
                BindGroupItem::StorageBuffer { label: "feature_matches", min_binding_size: 8, read_only: true }
            ]);

            self.add_compute_pipelines("matches_visualization", &["matches_visualization"], &["matches_visualization"], &[]);
        }

        {
            self.add_bind_group("corner_visualization_2", &[
                BindGroupItem::Texture { label: "input_image" }
            ]);

            self.add_render_pipelines(
                "corner_visualization_2", 
                &["corner_visualization_2"], 
                &[("corner_visualization_2", ("vs_main", "fs_main"))], 
                &[], 
                &[Some(wgpu::TextureFormat::Rgba8Unorm.into())], 
                &[
                    wgpu::VertexBufferLayout {
                        array_stride: 2 * 4,
                        attributes: &[
                            wgpu::VertexAttribute { shader_location: 0, offset: 0, format: wgpu::VertexFormat::Uint32x2 }
                        ],
                        step_mode: wgpu::VertexStepMode::Instance
                    }
                ]
            );
        }
    }

    pub fn run(&self, params: OrbParams) {
        let mut encoder = self.compute().device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None
        });

        encoder.clear_buffer(&self.storage().buffers["latest_corners_counter"], 0, None);

        // Grayscale image
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { 
                        view: &self.storage().texture_views["grayscale_image"], 
                        resolve_target: None, 
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store
                        },
                    })
                ],
                ..Default::default()
            });

            rpass.set_pipeline(&self.storage().render_pipelines["color_to_grayscale"]);
            rpass.set_bind_group(0, &self.storage().bind_groups["color_to_grayscale"], &[]);
            rpass.draw(0..3, 0..1);
        }

        // Corner detector
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });

            cpass.set_pipeline(&self.storage().compute_pipelines["corner_detector"]);
            cpass.set_bind_group(0, &self.storage().bind_groups["corner_detector"], &[]);
            cpass.dispatch_workgroups(
                (self.config.image_size.width / 2 + 7) / 8,
                (self.config.image_size.height / 2 + 7) / 8,
                1
            );
        }

        // Wait for device to finish
        self.copy_buffer_to_staging(&mut encoder, "latest_corners_counter");

        self.compute().queue.submit(Some(encoder.finish()));

        self.prepare_staging_buffer("latest_corners_counter");

        self.compute().device.poll(wgpu::MaintainBase::Wait);

        // Retrieve data from staging buffer
        let corner_count: u32 = {
            let mut dst = [0u8; 4];
            
            self.read_staging_buffer("latest_corners_counter", &mut dst);

            *bytemuck::cast_slice(&dst).iter().next().unwrap()
        };

        println!("Corner count: {}", corner_count);

        let mut encoder = self.compute().device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None
        });

        // Visualize corners
        {
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTextureBase { 
                    texture: &self.storage().textures["input_image"], 
                    mip_level: 0, 
                    origin: wgpu::Origin3d::ZERO, 
                    aspect: wgpu::TextureAspect::All
                },
                wgpu::ImageCopyTextureBase { 
                    texture: &self.storage().textures["visualization"], 
                    mip_level: 0, 
                    origin: wgpu::Origin3d::ZERO, 
                    aspect: wgpu::TextureAspect::All
                },
                self.config.image_size.clone()
            );

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { 
                label: None, 
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.storage().texture_views["visualization"],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store
                    }
                })],
                ..Default::default()
            });

            rpass.set_pipeline(&self.storage().render_pipelines["corner_visualization_2"]);
            rpass.set_bind_group(0, &self.storage().bind_groups["corner_visualization_2"], &[]);
            rpass.set_vertex_buffer(0, self.storage().buffers["latest_corners"].slice(..));
            rpass.draw(0..6, 0..corner_count);
        }
        
        

        // Gaussian blur x
        // {
        //     let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        //         label: None,
        //         color_attachments: &[
        //             Some(wgpu::RenderPassColorAttachment { 
        //                 view: &self.storage().texture_views["gaussian_blur_x"], 
        //                 resolve_target: None, 
        //                 ops: wgpu::Operations {
        //                     load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
        //                     store: wgpu::StoreOp::Store
        //                 },
        //             })
        //         ],
        //         ..Default::default()
        //     });

        //     rpass.set_pipeline(&self.storage().render_pipelines["gaussian_blur_x"]);
        //     rpass.set_bind_group(0, &self.storage().bind_groups["gaussian_blur_x"], &[]);
        //     rpass.draw(0..3, 0..1);
        // }

        // Gaussian blur y
        // {
        //     let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        //         label: None,
        //         color_attachments: &[
        //             Some(wgpu::RenderPassColorAttachment { 
        //                 view: &self.storage().texture_views["gaussian_blur"], 
        //                 resolve_target: None, 
        //                 ops: wgpu::Operations {
        //                     load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
        //                     store: wgpu::StoreOp::Store
        //                 },
        //             })
        //         ],
        //         ..Default::default()
        //     });

        //     rpass.set_pipeline(&self.storage().render_pipelines["gaussian_blur_y"]);
        //     rpass.set_bind_group(0, &self.storage().bind_groups["gaussian_blur_y"], &[]);
        //     rpass.draw(0..3, 0..1);
        // }

        // Corner visualization
        // {
        //     encoder.copy_texture_to_texture(
        //         wgpu::ImageCopyTextureBase { 
        //             texture: &self.storage().textures["input_image"], 
        //             mip_level: 0, 
        //             origin: wgpu::Origin3d::ZERO, 
        //             aspect: wgpu::TextureAspect::All
        //         },
        //         wgpu::ImageCopyTextureBase { 
        //             texture: &self.storage().textures["visualization"], 
        //             mip_level: 0, 
        //             origin: wgpu::Origin3d::ZERO, 
        //             aspect: wgpu::TextureAspect::All
        //         },
        //         self.config.image_size.clone()
        //     );

        //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //         label: None,
        //         timestamp_writes: None
        //     });

        //     cpass.set_pipeline(&self.storage().compute_pipelines["corner_visualization"]);
        //     cpass.set_bind_group(0, &self.storage().bind_groups["corner_visualization"], &[]);
        //     cpass.dispatch_workgroups(
        //         (self.config.max_features + 63) / 64,
        //         1,
        //         1
        //     );
        // }

        // Feature descriptors
        // {
        //     let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        //         label: None,
        //         timestamp_writes: None
        //     });

        //     cpass.set_pipeline(&self.storage().compute_pipelines["feature_descriptors"]);
        //     cpass.set_bind_group(0, &self.storage().bind_groups["feature_descriptors"], &[]);
        //     cpass.dispatch_workgroups(
        //         (self.config.max_features + 63) / 64,
        //         1,
        //         1
        //     );
        // }

        // Feature matching
        // We want the _best_ match for each feature
        // 
        {
            // let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            //     label: None,
            //     timestamp_writes: None
            // });

            // cpass.set_pipeline(&self.storage().compute_pipelines["feature_matching"]);
            // cpass.set_bind_group(0, &self.storage().bind_groups["feature_matching"], &[]);
            // cpass.dispatch_workgroups(
            //     self.config.max_features,
            //     (self.config.max_features + 63) / 64,
            //     1
            // );
        }

        {
            // let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            //     label: None,
            //     timestamp_writes: None
            // });

            // cpass.set_pipeline(&self.storage().compute_pipelines["matches_visualization"]);
            // cpass.set_bind_group(0, &self.storage().bind_groups["matches_visualization"], &[]);
            // cpass.dispatch_workgroups(
            //     (self.config.max_features + 63) / 64,
            //     1,
            //     1
            // );
        }

        if params.record_keyframe {
            encoder.copy_buffer_to_buffer(
                &self.storage().buffers["latest_corners"], 
                0, 
                &self.storage().buffers["previous_corners"], 
                0, 
                self.storage().buffers["previous_corners"].size()
            );

            encoder.copy_buffer_to_buffer(
                &self.storage().buffers["latest_corners_counter"], 
                0, 
                &self.storage().buffers["previous_corners_counter"], 
                0, 
                self.storage().buffers["previous_corners_counter"].size()
            );

            encoder.copy_buffer_to_buffer(
                &self.storage().buffers["latest_descriptors"], 
                0, 
                &self.storage().buffers["previous_descriptors"], 
                0, 
                self.storage().buffers["previous_descriptors"].size()
            );
        }

        self.compute().queue.submit(Some(encoder.finish()));
    }

    pub fn write_input_image(&self, bytes: &[u8]) {
        self.compute().queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.storage().textures["input_image"],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: (4 * self.config.image_size.width).into(),
                rows_per_image: None,
            },
            self.config.image_size
        );
    }
}