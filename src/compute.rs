use std::sync::Arc;
use tiny_wgpu::Compute;

pub trait ComputeProgram {
    type Config;
    type Params;

    fn init(config: Self::Config, compute: Arc<Compute>) -> Self;
    fn run(&mut self, params: Self::Params, compute: Arc<Compute>);
}