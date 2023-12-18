use std::{marker::PhantomData, sync::Arc};

pub trait Scalar: Copy {}

impl Scalar for f32 {}

pub struct Tensor<const D: usize, T: Scalar> {
    pub size: [u64; D],
    pub(crate) buffer: Arc<wgpu::Buffer>,
    pub(crate) _phantom: PhantomData<T>,
}
