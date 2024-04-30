/*
TODO: 
 - Render blur texture using a normal vertex/fragment shader, samplers, two passes
 - Should get performance from 1.0ms to 0.5ms
*/

use wgpu::{
    BufferUsages, ShaderStages, TextureUsages
};

use tiny_wgpu::{
    BindGroupItem, Compute, ComputeKernel, ComputeProgram, RenderKernel, Storage
};

pub struct OrbConfig {
    pub image_size: wgpu::Extent3d,
    pub max_features: u32,
    pub max_matches: u32,
    pub hierarchy_depth: u32
}

pub struct OrbParams {
    pub record_keyframe: bool
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
    const IMAGE_HIERARCHY_NAMES: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_{}", N as u32),
        )*
    ];
    const IMAGE_HIERARCHY_BLUR_TMP_NAMES: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_tmp_{}", N as u32),
        )*
    ];
    const IMAGE_HIERARCHY_BLUR_NAMES: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_{}", N as u32),
        )*
    ];
    const IMAGE_HIERARCHY_BLUR_TMP_BIND_GROUP_NAMES: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_tmp_bind_group_{}", N as u32),
        )*
    ];
    const IMAGE_HIERARCHY_BLUR_BIND_GROUP_NAMES: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_bind_group_{}", N as u32),
        )*
    ];
    const IMAGE_HIERARCHY_BLUR_X_KERNEL_NAMES: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_x_kernel_{}", N as u32),
        )*
    ];
    const IMAGE_HIERARCHY_BLUR_Y_KERNEL_NAMES: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blur_y_kernel_{}", N as u32),
        )*
    ];
    const IMAGE_HIERARCHY_BLIT_BIND_GROUP_NAMES: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blit_bind_group_{}", N as u32),
        )*
    ];
    const IMAGE_HIERARCHY_BLIT_KERNEL_NAMES: [&str; MAX_HIERARCHY_DEPTH] = [
        #(
            const_format::formatcp!("image_hierarchy_blit_kernel_{}", N as u32),
        )*
    ];
});

impl OrbProgram {
    pub fn init(&mut self) {

        self.add_module("color_to_grayscale", wgpu::include_wgsl!("shaders/grayscale.wgsl"));
        self.add_module("blit", wgpu::include_wgsl!("shaders/blit.wgsl"));
        self.add_module("gaussian_blur_x", wgpu::include_wgsl!("shaders/gaussian_blur_x.wgsl"));
        self.add_module("gaussian_blur_y", wgpu::include_wgsl!("shaders/gaussian_blur_y.wgsl"));
        self.add_module("corner_detector", wgpu::include_wgsl!("shaders/corner_detector.wgsl"));
        self.add_module("feature_descriptors", wgpu::include_wgsl!("shaders/feature_descriptors.wgsl"));

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

        self.add_texture(
            "grayscale",
            TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            wgpu::TextureFormat::R16Float,
            self.config.image_size
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

        let mut cur_width = self.config.image_size.width;
        let mut cur_height = self.config.image_size.height;

        for i in 0..(self.config.hierarchy_depth as usize) {
            // Add textures            
            self.add_texture(
                &IMAGE_HIERARCHY_NAMES[i],
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                wgpu::TextureFormat::R16Float,
                wgpu::Extent3d {
                    width: cur_width,
                    height: cur_height,
                    depth_or_array_layers: 1
                }
            );

            if i > 0 {
                // Add pipeline to blit from i - 1 to i
                self.add_bind_group(&IMAGE_HIERARCHY_BLIT_BIND_GROUP_NAMES[i], &[
                    BindGroupItem::Sampler { label: "linear_sampler" },
                    BindGroupItem::Texture { label: &IMAGE_HIERARCHY_NAMES[i - 1] }
                ]);
            }

            self.add_texture(
                &IMAGE_HIERARCHY_BLUR_TMP_NAMES[i],
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                wgpu::TextureFormat::R16Float,
                wgpu::Extent3d {
                    width: cur_width,
                    height: cur_height,
                    depth_or_array_layers: 1
                }
            );

            self.add_bind_group(&IMAGE_HIERARCHY_BLUR_TMP_BIND_GROUP_NAMES[i], &[
                BindGroupItem::Sampler { label: "linear_sampler" },
                BindGroupItem::Texture { label: &IMAGE_HIERARCHY_NAMES[i] }
            ]);

            self.add_render_pipelines(
                "gaussian_blur_x", 
                &[&IMAGE_HIERARCHY_BLUR_TMP_BIND_GROUP_NAMES[i]], 
                &[RenderKernel { label: &IMAGE_HIERARCHY_BLUR_X_KERNEL_NAMES[i], vertex: "vs_main", fragment: "fs_main" }], 
                &[], 
                &[Some(wgpu::TextureFormat::R16Float.into())], 
                &[], 
                None,
                None
            );

            self.add_texture(
                &IMAGE_HIERARCHY_BLUR_NAMES[i],
                TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                wgpu::TextureFormat::R16Float,
                wgpu::Extent3d {
                    width: cur_width,
                    height: cur_height,
                    depth_or_array_layers: 1
                }
            );
            
            self.add_bind_group(&IMAGE_HIERARCHY_BLUR_BIND_GROUP_NAMES[i], &[
                BindGroupItem::Sampler { label: "linear_sampler" },
                BindGroupItem::Texture { label: &IMAGE_HIERARCHY_BLUR_TMP_NAMES[i] }
            ]);

            self.add_render_pipelines(
                "gaussian_blur_y", 
                &[&IMAGE_HIERARCHY_BLUR_BIND_GROUP_NAMES[i]], 
                &[RenderKernel { label: &IMAGE_HIERARCHY_BLUR_Y_KERNEL_NAMES[i], vertex: "vs_main", fragment: "fs_main" }], 
                &[], 
                &[Some(wgpu::TextureFormat::R16Float.into())], 
                &[], 
                None,
                None
            );

            cur_width /= 2;
            cur_height /= 2;
        }

        self.add_buffer(
            "latest_corners",
            BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            // Each feature is 4 u32
            (self.config.max_features * 4 * 4) as u64
        );

        self.add_buffer(
            "latest_corners_counter",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            4
        );

        self.add_staging_buffer("latest_corners_counter");

        self.add_bind_group("corner_detector", &[
            BindGroupItem::Texture { label: "grayscale_image" },
            BindGroupItem::StorageBuffer { label: "latest_corners", min_binding_size: 8, read_only: false },
            BindGroupItem::StorageBuffer { label: "latest_corners_counter", min_binding_size: 4, read_only: false }
        ]);

        self.add_compute_pipelines(
            "corner_detector", 
            &[&IMAGE_HIERARCHY_BLIT_BIND_GROUP_NAMES[1], "corner_detector"], 
            &[ComputeKernel { label: "corner_detector", entry_point: "corner_detector" }],
            &[wgpu::PushConstantRange { range: 0..4, stages: ShaderStages::COMPUTE }],
            None
        );

        self.add_buffer(
            "latest_descriptors",
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            // Each descriptor is 8 u32
            (self.config.max_features * 8 * 4) as u64
        );

        self.add_bind_group("feature_descriptors", &[
            BindGroupItem::Texture { label: "gaussian_blur" },
            BindGroupItem::StorageBuffer { label: "latest_corners", min_binding_size: 8, read_only: true },
            BindGroupItem::StorageBuffer { label: "latest_corners_counter", min_binding_size: 4, read_only: true },
            BindGroupItem::StorageBuffer { label: "latest_descriptors", min_binding_size: 8 * 4, read_only: false },
            BindGroupItem::Texture { label: "grayscale_image" }
        ]);

        self.add_compute_pipelines(
            "feature_descriptors", 
            &["feature_descriptors"], 
            &[ComputeKernel { label: "feature_descriptors", entry_point: "feature_descriptors" }], 
            &[],
            None
        );
        
    }

    pub fn run(&self) {
        let mut encoder = self.compute().device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None
        });

        encoder.clear_buffer(&self.storage().buffers["latest_corners_counter"], 0, None);

        // Grayscale image
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

        // Compute image hierarchy
        for i in 0..(self.config.hierarchy_depth as usize) {

            if i > 0 {

                // Downsample from previous layer
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { 
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment { 
                            view: &self.storage().texture_views[&IMAGE_HIERARCHY_NAMES[i - 1]], 
                            resolve_target: None, 
                            ops: wgpu::Operations { 
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), 
                                store: wgpu::StoreOp::Store
                            } 
                        })], 
                        ..Default::default()
                    });

                    rpass.set_pipeline(&self.storage().render_pipelines[&IMAGE_HIERARCHY_BLIT_KERNEL_NAMES[i]]);
                    rpass.set_bind_group(0, &self.storage().bind_groups[&IMAGE_HIERARCHY_BLIT_BIND_GROUP_NAMES[i]], &[]);
                    rpass.draw(0..3, 0..1);
                }

            }

            // Compute gaussian blur
            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { 
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { 
                        view: &self.storage().texture_views[&IMAGE_HIERARCHY_BLUR_TMP_NAMES[i]], 
                        resolve_target: None, 
                        ops: wgpu::Operations { 
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), 
                            store: wgpu::StoreOp::Store
                        } 
                    })], 
                    ..Default::default()
                });

                rpass.set_pipeline(&self.storage().render_pipelines[&IMAGE_HIERARCHY_BLUR_X_KERNEL_NAMES[i]]);
                rpass.set_bind_group(0, &self.storage().bind_groups[&IMAGE_HIERARCHY_BLUR_TMP_BIND_GROUP_NAMES[i]], &[]);
                rpass.draw(0..3, 0..1);
            }

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { 
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment { 
                        view: &self.storage().texture_views[&IMAGE_HIERARCHY_BLUR_NAMES[i - 1]], 
                        resolve_target: None, 
                        ops: wgpu::Operations { 
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), 
                            store: wgpu::StoreOp::Store
                        } 
                    })], 
                    ..Default::default()
                });

                rpass.set_pipeline(&self.storage().render_pipelines[&IMAGE_HIERARCHY_BLUR_Y_KERNEL_NAMES[i]]);
                rpass.set_bind_group(0, &self.storage().bind_groups[&IMAGE_HIERARCHY_BLUR_BIND_GROUP_NAMES[i]], &[]);
                rpass.draw(0..3, 0..1);
            }

            // Detect corners on this layer
            {
                let mut cpass = encoder.begin_compute_pass(&Default::default());

                cpass.set_pipeline(&self.storage().compute_pipelines["corner_detector"]);
                
                cpass.set_push_constants(0, bytemuck::cast_slice(&[ i as u32 ]));
                
                cpass.set_bind_group(0, &self.storage().bind_groups[&IMAGE_HIERARCHY_BLIT_BIND_GROUP_NAMES[i]], &[]);

                cpass.set_bind_group(1, &self.storage().bind_groups["corner_detector"], &[]);
                
                let image_width = self.storage().textures[&IMAGE_HIERARCHY_NAMES[i]].width();
                let image_height = self.storage().textures[&IMAGE_HIERARCHY_NAMES[i]].height();

                cpass.dispatch_workgroups(
                    (image_width + 7) / 8,
                    (image_height + 7) / 8,
                    1
                );
            }

            // TODO: Compute descriptors on this layer
            {
                
            }

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