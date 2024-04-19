use std::{collections::HashMap, sync::Arc};

use tiny_wgpu::Compute;


pub struct OrbConfig {
    pub image_size: wgpu::Extent3d,
    pub max_features: u32
}

pub struct OrbProgram<'a> {
    config: OrbConfig,
    compute: Arc<Compute>,
    modules: HashMap<&'a str, wgpu::ShaderModule>,
    textures: HashMap<&'a str, wgpu::Texture>,
    buffers: HashMap<&'a str, wgpu::Buffer>,
    bind_groups: HashMap<&'a str, wgpu::BindGroup>,
    bind_group_layouts: HashMap<&'a str, wgpu::BindGroupLayout>,
    compute_pipelines: HashMap<&'a str, wgpu::ComputePipeline>,
    render_pipelines: HashMap<&'a str, wgpu::RenderPipeline>
}

impl OrbProgram<'_> {
    pub fn texture(&self, name: &str) -> &wgpu::Texture {
        &self.textures[name]
    } 
    
    pub fn buffer(&self, name: &str) -> &wgpu::Buffer {
        &self.buffers[name]
    }

    pub fn bind_group(&self, name: &str) -> &wgpu::BindGroup {
        &self.bind_groups[name]
    }

    pub fn compute_pipeline(&self, name: &str) -> &wgpu::ComputePipeline {
        &self.compute_pipelines[name]
    }

    pub fn render_pipeline(&self, name: &str) -> &wgpu::RenderPipeline {
        &self.render_pipelines[name]
    }

    pub fn new(config: OrbConfig, compute: Arc<Compute>) -> Self {
        Self {
            config,
            compute,
            modules: HashMap::new(),
            textures: HashMap::new(),
            buffers: HashMap::new(),
            bind_groups: HashMap::new(),
            bind_group_layouts: HashMap::new(),
            compute_pipelines: HashMap::new(),
            render_pipelines: HashMap::new(),
        }
    }

    
}