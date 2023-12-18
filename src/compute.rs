#[must_use]
pub struct Computation<'a> {
    pub encode: Box<dyn FnOnce(&'a wgpu::CommandEncoder)>,
}

impl<'a> Computation<'a> {
    pub fn encode(self, encoder: &'a wgpu::CommandEncoder) {
        (self.encode)(encoder);
    }
}

#[macro_export]
macro_rules! myco {
    {
        $(
            $module:ident :: $kernel:ident [ $size:expr ] (
                $(
                    $tensors:expr
                ),*
            );
        )*
    } => {
        $crate::compute::Computation {
            encode: Box::new(&|encoder| {
                $(

                )
            }),
        }
    };
}
