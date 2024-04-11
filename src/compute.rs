use std::sync::Arc;

pub struct Compute {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue
}

impl Compute {
    pub async fn init() -> Arc<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { 
            backends: wgpu::Backends::PRIMARY, 
            flags: wgpu::InstanceFlags::empty(), 
            dx12_shader_compiler: wgpu::Dx12Compiler::Fxc, 
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic 
        });
    
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    
        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                
            }, 
            None
        ).await.unwrap();

        Arc::new(Self {
            instance,
            adapter,
            device,
            queue
        })
    }
}

pub trait ComputeProgram<Config> {
    fn init(config: Config, compute: Arc<Compute>) -> Self;
    fn run(&mut self, compute: Arc<Compute>);
}