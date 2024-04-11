/*

TODO:
 - Set up compute shader and workgroups to compute FAST features and log them into the buffer
 -

*/

use std::sync::Arc;

use crate::compute::{Compute, ComputeProgram};
use crate::buffers::{StagingStorageBufferPair, StorageStagingBufferPair};

pub struct OrbFeatureExtractorConfig {
    pub image_width: usize,
    pub image_height: usize,
    pub max_features: usize,
    pub threshold: u32
}
pub struct OrbFeatureExtractor {
    config: OrbFeatureExtractorConfig,
    module: wgpu::ShaderModule,
    image: StagingStorageBufferPair,
    image_buffer: wgpu::Buffer,
    features: StorageStagingBufferPair,
    counter: StorageStagingBufferPair,
    pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>
}

impl OrbFeatureExtractor {
    pub fn write_image_buffer(&self, compute: Arc<Compute>, data: &[u8]) {
        compute.queue.write_buffer(&self.image_buffer, 0, data);
        compute.queue.submit([]);
    }

    pub fn get_features(&self, compute: Arc<Compute>) -> Option<(u32, Vec<u32>)> {
        let (feature_slice, feature_receiver) = self.features.map_async();
        let (index_slice, index_receiver) = self.counter.map_async();
        
        compute.device.poll(wgpu::Maintain::Wait);

        let Ok(Ok(_)) = pollster::block_on(feature_receiver.recv_async()) else { return None; };
        let Ok(Ok(_)) = pollster::block_on(index_receiver.recv_async()) else { return None; };

        let feature_data = feature_slice.get_mapped_range();
        let feature_data: &[u32] = bytemuck::cast_slice(&feature_data);
        let feature_data: Vec<u32> = feature_data.iter().map(|x| *x).collect();

        let index_data = index_slice.get_mapped_range();
        let index_data: &[u32] = bytemuck::cast_slice(&index_data);
        let index_data = *index_data.iter().next().unwrap();

        Some((index_data, feature_data))
    }
}

impl ComputeProgram<OrbFeatureExtractorConfig> for OrbFeatureExtractor {
    fn init(config: OrbFeatureExtractorConfig, compute: Arc<Compute>) -> OrbFeatureExtractor {
        let image_buffer_size = config.image_width * config.image_height;
        let feature_buffer_size = config.max_features * 4 * 2;

        let module = compute.device.create_shader_module(wgpu::include_wgsl!("shaders/orb_features.wgsl"));

        let threshold_buffer = compute.device.create_buffer(&wgpu::BufferDescriptor {
            size: 4, // a size u32,
            label: Some("Threshold Buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let image_buffer = compute.device.create_buffer(&wgpu::BufferDescriptor {
            size: image_buffer_size as u64,
            label: Some("Image Buffer"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let image_size_buffer = compute.device.create_buffer(&wgpu::BufferDescriptor {
            size: 2 * 4,
            label: Some("Image Dimensions Buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false
        });

        {

            compute.queue.write_buffer(
                &threshold_buffer, 
                0, 
                bytemuck::cast_slice(&[config.threshold as u32])
            );
            compute.queue.write_buffer(
                &image_size_buffer, 
                0, 
                bytemuck::cast_slice(&[config.image_width as u32, config.image_height as u32])
            );
            compute.queue.submit([]);

            // let (s, r) = flume::bounded(1);

            // image_size_buffer.slice(..).map_async(wgpu::MapMode::Read, move |x| s.send(x).unwrap() );

            // compute.device.poll(wgpu::Maintain::Wait);

            // pollster::block_on(r.recv_async()).unwrap().unwrap();

            // let slice = image_size_buffer.slice(..).get_mapped_range();
            // println!("{:?}", slice.get(0..2));

            // image_size_buffer.unmap();
        }

        

        let image = StagingStorageBufferPair::new(&compute.device, image_buffer_size);
        let counter = StorageStagingBufferPair::new(&compute.device, 4);
        let features = StorageStagingBufferPair::new(&compute.device, feature_buffer_size);

        let pipeline = compute.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "main"
        });

        let bind_group = compute.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group 1"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: counter.storage.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: features.storage.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: threshold_buffer.as_entire_binding()
                }
            ]
        });

        let bind_group_2 = compute.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group 2"),
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: image_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_size_buffer.as_entire_binding()
                }
            ]
        });

        Self {
            config,
            module,
            image,
            image_buffer,
            features,
            counter,
            pipeline,
            bind_groups: vec![bind_group, bind_group_2]
        }
    }

    fn run(&mut self, compute: Arc<Compute>) {
        let mut encoder = compute.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None
        });

        self.image.copy(&mut encoder);

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None
            });
            
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_groups[0], &[]);
            cpass.set_bind_group(1, &self.bind_groups[1], &[]);
            
            let num_workgroups_x = (self.config.image_width + 15) / 16;
            let num_workgroups_y = (self.config.image_height + 15) / 16;
            let num_workgroups_z = 1;

            cpass.dispatch_workgroups(
                num_workgroups_x as u32,
                num_workgroups_y as u32,
                num_workgroups_z as u32
            );
        }

        self.features.copy(&mut encoder);
        self.counter.copy(&mut encoder);

        compute.queue.submit(Some(encoder.finish()));
    }
}

