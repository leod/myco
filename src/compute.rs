use crate::Device;

#[must_use]
pub struct Computation<'a> {
    pub encode: &'a dyn Fn(&'a Device, &'a mut wgpu::CommandEncoder),
}

impl<'a> Computation<'a> {
    pub fn encode(self, device: &'a Device, encoder: &'a mut wgpu::CommandEncoder) {
        (self.encode)(device, encoder);
    }
}

#[macro_export]
macro_rules! myco {
    {
        $(
            $module:ident :: $kernel:ident [ $group_count:expr ] (
                $(
                    $tensors:expr
                ),*
            );
        )*
    } => {
        $crate::compute::Computation {
            encode: &|device: &$crate::Device, encoder: &mut wgpu::CommandEncoder| {
                $(
                    let kernel = device.cache_kernel(
                        ::std::stringify!($module),
                        ::std::stringify!($kernel),
                        ::std::include_bytes!(
                            ::std::env!(
                                ::std::concat!(
                                    ::std::stringify!($module),
                                    ".spv",
                                ),
                            ),
                        ),
                    );

                    device.call(
                        &kernel,
                        $group_count,
                        &[
                            $($tensors),*
                        ],
                        ::std::concat!(
                            ::std::stringify!($module),
                            "::",
                            ::std::stringify!($kernel),
                        ),
                        encoder,
                    )
                )*
            },
        }
    };
}
