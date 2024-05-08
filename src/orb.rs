/*
TODO: 
 - Render blur texture using a normal vertex/fragment shader, samplers, two passes
 - Should get performance from 1.0ms to 0.5ms
*/

use wgpu::{
    BufferUsages, ShaderStages, TextureUsages
};

use tiny_wgpu::{
    Storage, Compute, ComputeProgram, BindGroupItem, ComputeKernel, RenderKernel
};

pub struct OrbConfig {
    pub image_size: wgpu::Extent3d,
    pub max_features: u32,
    pub hierarchy_depth: u32
}

pub struct OrbProgram {
    pub config: OrbConfig,
    pub compute: Compute,
    pub storage: Storage
}

impl ComputeProgram for OrbProgram {
    fn compute(&self) -> &Compute {
        &self.compute
    }
    fn storage(&self) -> &Storage {
        &self.storage
    }
    fn storage_mut(&mut self) -> &mut Storage {
        &mut self.storage
    }
}

// Define the labels we are using for the image hierarchy
seq_macro::seq!(N in 0..10 {
    const MAX_HIERARCHY_DEPTH: usize = 10;

    const IMAGE_HIERARCHY_BLIT_BIND_GROUPS: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blit_bind_group_{}", N as u32),
        )*
    ];

    const IMAGE_HIERARCHY_VIEWS: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_view_{}", N as u32),
        )*
    ];

    const IMAGE_HIERARCHY_BLUR_TMP_VIEWS: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_tmp_view_{}", N as u32),
        )*
    ];

    const IMAGE_HIERARCHY_BLUR_VIEWS: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_view_{}", N as u32),
        )*
    ];

    const IMAGE_HIERARCHY_BLUR_TMP_BIND_GROUPS: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_tmp_bind_group_{}", N as u32),
        )*
    ];

    const IMAGE_HIERARCHY_BLUR_BIND_GROUPS: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_view_{}", N as u32),
        )*
    ];
});

impl OrbProgram {
    pub fn init(&mut self) {

        self.add_module("color_to_grayscale", wgpu::include_wgsl!("shaders/grayscale.wgsl"));
        self.add_module("blit", wgpu::include_wgsl!("shaders/blit.wgsl"));
        self.add_module("gaussian_blur_x", wgpu::include_wgsl!("shaders/gaussian_blur_x.wgsl"));
        self.add_module("gaussian_blur_y", wgpu::include_wgsl!("shaders/gaussian_blur_y.wgsl"));
        self.add_module("fast", wgpu::include_wgsl!("shaders/fast.wgsl"));
        self.add_module("brief", wgpu::include_wgsl!("shaders/brief.wgsl"));

        self.add_texture(
            "input_image", 
            TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::COPY_SRC, 
            wgpu::TextureFormat::Rgba8Unorm, 
            self.config.image_size
        );
        
        self.add_sampler(
            "linear_sampler",
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

        self.add_bind_group("color_to_grayscale", &[
            BindGroupItem::Sampler { label: "linear_sampler" },
            BindGroupItem::Texture { label: "input_image" }
        ]);

        self.add_render_pipelines(
            "color_to_grayscale", 
            &["color_to_grayscale"], 
            &[RenderKernel { label: "color_to_grayscale", vertex: "vs_main", fragment: "fs_main" }],
            &[],
            &[Some(wgpu::TextureFormat::R16Float.into())], 
            &[],
            None,
            None
        );

        self.initialize_image_hierarchy();

        self.add_buffer(
            "corners",
            BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            // Each feature is 12 u32
            (self.config.max_features * 12 * 4) as u64
        );

        self.add_buffer(
            "counter",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            4
        );

        self.add_staging_buffer("counter");

        self.add_bind_group("fast", &[
            BindGroupItem::TextureView { label: "image_hierarchy_all", sample_type: wgpu::TextureSampleType::Float { filterable: true } },
            BindGroupItem::StorageBuffer { label: "corners", min_binding_size: 12 * 4, read_only: false },
            BindGroupItem::StorageBuffer { label: "counter", min_binding_size: 4, read_only: false }
        ]);

        self.add_compute_pipelines(
            "fast", 
            &["fast"], 
            &[ComputeKernel { label: "fast", entry_point: "compute_fast" }],
            &[wgpu::PushConstantRange { range: 0..4, stages: ShaderStages::COMPUTE }],
            None
        );

        self.add_buffer("descriptors", BufferUsages::STORAGE, (self.config.max_features * 8 * 4) as u64);

        self.add_bind_group("descriptors", &[
            BindGroupItem::StorageBuffer { label: "corners", min_binding_size: 12 * 4, read_only: true },
            BindGroupItem::StorageBuffer { label: "counter", min_binding_size: 4, read_only: true },
            BindGroupItem::StorageBuffer { label: "descriptors", min_binding_size: 8 * 4, read_only: false },
            BindGroupItem::TextureView { label: "blur_hierarchy_all", sample_type: wgpu::TextureSampleType::Float { filterable: true } }
        ]);

        self.add_compute_pipelines(
            "brief", 
            &[ "descriptors" ], 
            &[ComputeKernel { label: "brief", entry_point: "brief" }], 
            &[], 
            None
        );
    }

    fn initialize_image_hierarchy(&mut self) {
        
        {
            let texture = self.compute().device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                mip_level_count: self.config.hierarchy_depth,
                size: self.config.image_size,
                format: wgpu::TextureFormat::R16Float,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_DST,
                view_formats: &[]
            });

            self.storage_mut().textures.insert("image_hierarchy", texture);
        }

        {
            let texture = &self.storage().textures["image_hierarchy"];
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(self.config.hierarchy_depth),
                base_array_layer: 0,
                array_layer_count: None
            });
            self.storage_mut().texture_views.insert("image_hierarchy_all", view);
        }

        for mip in 0..self.config.hierarchy_depth {
            let texture = &self.storage().textures["image_hierarchy"];

            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None
            });

            self.storage_mut().texture_views.insert(IMAGE_HIERARCHY_VIEWS[mip as usize], view);
        }

        for target_mip in 1..self.config.hierarchy_depth as usize {
            self.add_bind_group(IMAGE_HIERARCHY_BLIT_BIND_GROUPS[target_mip], &[
                BindGroupItem::Sampler { label: "linear_sampler" },
                BindGroupItem::TextureView { 
                    label: IMAGE_HIERARCHY_VIEWS[target_mip - 1],
                    sample_type: wgpu::TextureSampleType::Float { filterable: true }
                }
            ]);
        }

        self.add_render_pipelines(
            "blit",
            &[ IMAGE_HIERARCHY_BLIT_BIND_GROUPS[1] ],
            &[ RenderKernel { label: "image_hierarchy_mipmap", vertex: "vs_main", fragment: "fs_main" } ],
            &[],
            &[ Some(wgpu::TextureFormat::R16Float.into()) ],
            &[],
            None,
            None
        );

        {
            let texture = self.compute().device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                mip_level_count: self.config.hierarchy_depth,
                size: self.config.image_size,
                format: wgpu::TextureFormat::R16Float,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[]
            });

            self.storage_mut().textures.insert("blur_tmp_hierarchy", texture);
        }

        {
            let texture = self.compute().device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                mip_level_count: self.config.hierarchy_depth,
                size: self.config.image_size,
                format: wgpu::TextureFormat::R16Float,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[]
            });

            self.storage_mut().textures.insert("blur_hierarchy", texture);
        }

        {
            let texture = &self.storage().textures["blur_hierarchy"];
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(self.config.hierarchy_depth),
                base_array_layer: 0,
                array_layer_count: None
            });
            self.storage_mut().texture_views.insert("blur_hierarchy_all", view);
        }

        for mip in 0..self.config.hierarchy_depth {
            let texture = &self.storage().textures["blur_tmp_hierarchy"];

            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None
            });

            self.storage_mut().texture_views.insert(IMAGE_HIERARCHY_BLUR_TMP_VIEWS[mip as usize], view);
        }

        for mip in 0..self.config.hierarchy_depth {
            let texture = &self.storage().textures["blur_hierarchy"];

            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                format: None,
                dimension: None,
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: None
            });

            self.storage_mut().texture_views.insert(IMAGE_HIERARCHY_BLUR_VIEWS[mip as usize], view);
        }

        for target_mip in 0..self.config.hierarchy_depth as usize {
            self.add_bind_group(IMAGE_HIERARCHY_BLUR_TMP_BIND_GROUPS[target_mip], &[
                BindGroupItem::Sampler { label: "linear_sampler" },
                BindGroupItem::TextureView { 
                    label: IMAGE_HIERARCHY_VIEWS[target_mip],
                    sample_type: wgpu::TextureSampleType::Float { filterable: true }
                }
            ]);

            self.add_bind_group(IMAGE_HIERARCHY_BLUR_BIND_GROUPS[target_mip], &[
                BindGroupItem::Sampler { label: "linear_sampler" },
                BindGroupItem::TextureView { 
                    label: IMAGE_HIERARCHY_BLUR_TMP_VIEWS[target_mip],
                    sample_type: wgpu::TextureSampleType::Float { filterable: true }
                }
            ]);
        }

        self.add_render_pipelines(
            "gaussian_blur_x",
            &[ IMAGE_HIERARCHY_BLUR_TMP_BIND_GROUPS[1] ],
            &[ RenderKernel { label: "gaussian_blur_x", vertex: "vs_main", fragment: "fs_main" } ],
            &[],
            &[ Some(wgpu::TextureFormat::R16Float.into()) ],
            &[],
            None,
            None
        );

        self.add_render_pipelines(
            "gaussian_blur_x",
            &[ IMAGE_HIERARCHY_BLUR_BIND_GROUPS[1] ],
            &[ RenderKernel { label: "gaussian_blur_y", vertex: "vs_main", fragment: "fs_main" } ],
            &[],
            &[ Some(wgpu::TextureFormat::R16Float.into()) ],
            &[],
            None,
            None
        );
    }

    fn generate_hierarchy(&self, encoder: &mut wgpu::CommandEncoder) {
        // Render mip levels for un-blurred image
        for target_mip in 1..(self.config.hierarchy_depth as usize) {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.storage().texture_views[IMAGE_HIERARCHY_VIEWS[target_mip]],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store
                    }
                })],
                ..Default::default()
            });

            rpass.set_pipeline(&self.storage().render_pipelines["image_hierarchy_mipmap"]);
            rpass.set_bind_group(0, &self.storage().bind_groups[IMAGE_HIERARCHY_BLIT_BIND_GROUPS[target_mip]], &[]);
            rpass.draw(0..3, 0..1);
        }

        // Render mip levels for blurred image
        for target_mip in 0..(self.config.hierarchy_depth as usize) {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.storage().texture_views[IMAGE_HIERARCHY_BLUR_TMP_VIEWS[target_mip]],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store
                    }
                })],
                ..Default::default()
            });

            rpass.set_pipeline(&self.storage().render_pipelines["gaussian_blur_x"]);
            rpass.set_bind_group(0, &self.storage().bind_groups[IMAGE_HIERARCHY_BLUR_TMP_BIND_GROUPS[target_mip]], &[]);
            rpass.draw(0..3, 0..1);
        }

        for target_mip in 0..(self.config.hierarchy_depth as usize) {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.storage().texture_views[IMAGE_HIERARCHY_BLUR_VIEWS[target_mip]],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store
                    }
                })],
                ..Default::default()
            });
            
            rpass.set_pipeline(&self.storage().render_pipelines["gaussian_blur_y"]);
            rpass.set_bind_group(0, &self.storage().bind_groups[IMAGE_HIERARCHY_BLUR_BIND_GROUPS[target_mip]], &[]);
            rpass.draw(0..3, 0..1);
        }
    }

    pub fn extract_corners(&self) -> u32 {
        let mut encoder = self.compute().device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None
        });

        encoder.clear_buffer(&self.storage().buffers["counter"], 0, None);

        // Grayscale image
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { 
                        view: &self.storage().texture_views[&IMAGE_HIERARCHY_VIEWS[0]], 
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

        self.generate_hierarchy(&mut encoder);

        // Compute corners
        let mut width = self.config.image_size.width;
        let mut height = self.config.image_size.height;

        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_pipeline(&self.storage().compute_pipelines["fast"]);
            cpass.set_bind_group(0, &self.storage().bind_groups["fast"], &[]);

            for i in 0..(self.config.hierarchy_depth as usize) {
                cpass.set_push_constants(0, bytemuck::cast_slice(&[ i as u32 ]));
                cpass.dispatch_workgroups(
                    (width + 7) / 8,
                    (height + 7) / 8,
                    1
                ); 

                width /= 2;
                height /= 2;
            }
        }

        // Compute all descriptors
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());

            cpass.set_pipeline(&self.storage().compute_pipelines["brief"]);
            cpass.set_bind_group(0, &self.storage().bind_groups["descriptors"], &[]);

            cpass.dispatch_workgroups(
                1,
                (self.config.max_features + 7) / 8, 
                1
            );
        }

        // Wait for device to finish
        self.copy_buffer_to_staging(&mut encoder, "counter");

        self.compute().queue.submit(Some(encoder.finish()));

        self.prepare_staging_buffer("counter");

        self.compute().device.poll(wgpu::MaintainBase::Wait);

        // Retrieve data from staging buffer
        let corner_count: u32 = {
            let mut dst = [0u8; 4];
            
            self.read_staging_buffer("counter", &mut dst);

            *bytemuck::cast_slice(&dst).iter().next().unwrap()
        };

        return corner_count;
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