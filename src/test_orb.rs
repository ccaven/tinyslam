use std::sync::Arc;
use crate::compute::{Compute, ComputeProgram};

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct OrbData {
    pub x: u32,
    pub y: u32
}

pub struct OrbFeatureExtractorConfig {
    pub image_width: usize,
    pub image_height: usize,
    pub max_features: usize
}
pub struct OrbFeatureExtractor {
    config: OrbFeatureExtractorConfig,
    module: wgpu::ShaderModule,
    image_buffer: wgpu::Buffer,
    feature_storage_buffer: wgpu::Buffer,
    feature_staging_buffer: wgpu::Buffer,
    feature_index_buffer: wgpu::Buffer,
    feature_index_staging_buffer: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup
}

impl OrbFeatureExtractor {
    pub fn write_image_buffer(&self, compute: Arc<Compute>, image_rgba: &[u8]) {
        compute.queue.write_buffer(&self.image_buffer, 0, image_rgba);
    }

    pub fn get_features(&self) -> Vec<OrbData> {
        // Read from feature_staging_buffer
        let data = self.feature_staging_buffer.slice(..);

        // TODO: do we need to wait before accessing data?
        let data = data.get_mapped_range();

        let data = data.chunks_exact(std::mem::size_of::<OrbData>());

        let data: Vec<OrbData> = data.map(|x| unsafe {
            std::ptr::read(x.as_ptr() as *const OrbData)
        }).collect();

        data
    }

    pub fn get_index(&self, compute: Arc<Compute>) -> Option<(u32, Vec<u32>)> {
        // Read from feature_staging_buffer
        let index_data = self.feature_index_staging_buffer.slice(..);
        let feature_data = self.feature_staging_buffer.slice(..);

        // TODO: do we need to wait before accessing data?
        let (index_sender, index_receiver) = flume::bounded(1);
        let (feature_sender, feature_receiver) = flume::bounded(1);

        index_data.map_async(wgpu::MapMode::Read, move |v| index_sender.send(v).unwrap());
        feature_data.map_async(wgpu::MapMode::Read, move |v| feature_sender.send(v).unwrap());

        compute.device.poll(wgpu::MaintainBase::Wait);
        
        let Ok(Ok(_)) = pollster::block_on(index_receiver.recv_async()) else { return None };
        let Ok(Ok(_)) = pollster::block_on(feature_receiver.recv_async()) else { return None };

        let index_data = index_data.get_mapped_range();
        let index_data: &[u32] = bytemuck::cast_slice(&index_data);
        let index_data = *index_data.iter().next().unwrap();

        let feature_data = feature_data.get_mapped_range();
        let feature_data: &[u32] = bytemuck::cast_slice(&feature_data);
        let feature_data = feature_data.iter().map(|x| *x).collect();

        Some((index_data, feature_data))

    }
}

impl ComputeProgram<OrbFeatureExtractorConfig> for OrbFeatureExtractor {
    fn init(config: OrbFeatureExtractorConfig, compute: Arc<Compute>) -> OrbFeatureExtractor {
        let image_buffer_size = config.image_width * config.image_height * std::mem::size_of::<u32>();
        let feature_buffer_size = 1024 * 4 * 2;

        let module = compute.device.create_shader_module(wgpu::include_wgsl!("shaders/orb_features.wgsl"));

        let image_buffer = compute.device.create_buffer(&wgpu::BufferDescriptor {
            size: image_buffer_size as u64,
            label: Some("Image Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let feature_index_buffer = compute.device.create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<u32>() as u64,
            label: Some("Index Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        let feature_index_staging_buffer = compute.device.create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<u32>() as u64,
            label: Some("Index Buffer"),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let feature_storage_buffer = compute.device.create_buffer(&wgpu::BufferDescriptor {
            size: feature_buffer_size as u64,
            label: Some("Feature Storage Buffer"),
            usage: wgpu::BufferUsages::STORAGE | 
                   wgpu::BufferUsages::COPY_SRC | 
                   wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let feature_staging_buffer = compute.device.create_buffer(&wgpu::BufferDescriptor {
            size: feature_buffer_size as u64,
            label: Some("Feature Staging Buffer"),
            usage: wgpu::BufferUsages::MAP_READ | 
                   wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let pipeline = compute.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "main"
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let bind_group = compute.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                // wgpu::BindGroupEntry {
                //     binding: 0,
                //     resource: feature_storage_buffer.as_entire_binding()
                // },
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: feature_index_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: feature_storage_buffer.as_entire_binding()
                },
                // TODO: feature_index is a single atomic u32
            ]
        });

        Self {
            config,
            module,
            image_buffer,
            feature_storage_buffer,
            feature_staging_buffer,
            feature_index_buffer,
            feature_index_staging_buffer,
            pipeline,
            bind_group
        }
    }

    fn run(&mut self, compute: Arc<Compute>) {
        let mut encoder = compute.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None
        });

        compute.queue.write_buffer(&self.feature_index_buffer, 0, bytemuck::cast_slice(&[0u32]));

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });
            
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);

            let num_workgroups_x = 16;
            let num_workgroups_y = 16;
            let num_workgroups_z = 1;

            cpass.dispatch_workgroups(
                num_workgroups_x as u32,
                num_workgroups_y as u32,
                num_workgroups_z as u32
            );
        }

        let feature_buffer_size = 1024 * 4 * 2;
        
        encoder.copy_buffer_to_buffer(
            &self.feature_storage_buffer,
            0,
            &self.feature_staging_buffer,
            0,
            feature_buffer_size as u64
        );

        encoder.copy_buffer_to_buffer(
            &self.feature_index_buffer,
            0,
            &self.feature_index_staging_buffer,
            0,
            std::mem::size_of::<u32>() as u64
        );

        compute.queue.submit(Some(encoder.finish()));
    }
}

