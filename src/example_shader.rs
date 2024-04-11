use wgpu::util::DeviceExt;



async fn main () {

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

    let module = device.create_shader_module(
        wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                std::borrow::Cow::Borrowed(
                    include_str!("./shaders/example_shader.wgsl")
                )
            )
        }
    );

    let input = [1f32, 3.0, 5.0];

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor{
        size: input.len() as u64,
        label: None,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false
    });

    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: "main"
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding()
        }]
    });

    

}