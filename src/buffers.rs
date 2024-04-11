use crate::compute::Compute;


/** Helper struct to couple storage and staging buffers */
pub struct StorageStagingBufferPair {
    pub storage: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    pub size_bytes: usize
}

impl StorageStagingBufferPair {
    pub fn new(device: &wgpu::Device, size_bytes: usize) -> Self {
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        Self {
            storage,
            staging,
            size_bytes
        }
    }

    pub fn copy(&self, command_encoder: &mut wgpu::CommandEncoder) {
        command_encoder.copy_buffer_to_buffer(
            &self.storage, 
            0, 
            &self.staging, 
            0,
            self.size_bytes as u64
        );
    }

    pub fn map_async(&self) -> (wgpu::BufferSlice<'_>, flume::Receiver<Result<(), wgpu::BufferAsyncError>>) {
        let mapped_data = self.staging.slice(..);
        let (sender, receiver) = flume::bounded(1);
        mapped_data.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        return (mapped_data, receiver);
    }
}

pub struct StagingStorageBufferPair {
    pub staging: wgpu::Buffer,
    pub storage: wgpu::Buffer,
    pub size_bytes: usize
}

impl StagingStorageBufferPair {
    pub fn new(device: &wgpu::Device, size_bytes: usize) -> Self {
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false
        });

        Self {
            storage,
            staging,
            size_bytes
        }
    }

    pub fn copy(&self, command_encoder: &mut wgpu::CommandEncoder) {
        command_encoder.copy_buffer_to_buffer(
            &self.staging, 
            0, 
            &self.storage, 
            0,
            self.size_bytes as u64
        );
    }

    pub fn write_to_staging(&self, compute: &Compute, data: &[u8]) {
        compute.queue.write_buffer(&self.storage, 0, data);
    }
}