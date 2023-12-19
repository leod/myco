use std::{cell::RefCell, marker::PhantomData, rc::Rc, sync::Arc};

use glam::UVec3;
use hashbrown::HashMap;
use wgpu::util::DeviceExt as _;

use crate::{Scalar, Tensor};

pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
}

struct Module {
    module: wgpu::ShaderModule,
    kernels: HashMap<String, Arc<Kernel>>,
}

pub struct Device {
    device: Arc<wgpu::Device>,
    modules: Rc<RefCell<HashMap<String, Module>>>,
}

impl Device {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            modules: Default::default(),
        }
    }

    pub fn zeros<const D: usize, T: Scalar>(&self, size: [u64; D]) -> Tensor<D, T> {
        let num_elements = size.iter().product();

        let contents: Vec<_> = (0..num_elements).map(|_| 0.0).collect();

        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(&contents),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        Tensor {
            size,
            buffer: Arc::new(buffer),
            _phantom: PhantomData,
        }
    }

    #[doc(hidden)]
    pub fn cache_kernel(
        &self,
        module_name: &str,
        kernel_name: &str,
        module_spirv: &[u8],
    ) -> Arc<Kernel> {
        let mut modules = self.modules.borrow_mut();

        let (_, module) = modules
            .raw_entry_mut()
            .from_key(module_name)
            .or_insert_with(|| {
                (
                    module_name.to_string(),
                    self.create_module(module_name, module_spirv),
                )
            });

        module
            .kernels
            .raw_entry_mut()
            .from_key(kernel_name)
            .or_insert_with(|| {
                (
                    kernel_name.to_string(),
                    Arc::new(self.create_kernel(kernel_name, &module.module)),
                )
            })
            .1
            .clone()
    }

    #[doc(hidden)]
    pub fn call(
        &self,
        kernel: &Kernel,
        group_count: UVec3,
        args: &[&wgpu::Buffer],
        debug_marker: &str,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let bind_group_layout = kernel.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &args
                .iter()
                .enumerate()
                .map(|(binding, arg)| wgpu::BindGroupEntry {
                    binding: binding.try_into().unwrap(),
                    resource: arg.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.insert_debug_marker(debug_marker);
        pass.dispatch_workgroups(group_count.x, group_count.y, group_count.z);
    }

    fn create_module(&self, name: &str, spirv: &[u8]) -> Module {
        let source = wgpu::util::make_spirv_raw(spirv);
        let descriptor = wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::SpirV(source),
        };
        let module = self.device.create_shader_module(descriptor);

        Module {
            module,
            kernels: Default::default(),
        }
    }

    fn create_kernel(&self, name: &str, module: &wgpu::ShaderModule) -> Kernel {
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &module,
                entry_point: "main",
            });

        Kernel { pipeline }
    }
}
