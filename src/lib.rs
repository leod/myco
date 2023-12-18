mod device;
mod tensor;

#[macro_use]
pub mod compute;

pub use device::Device;
pub use tensor::{Scalar, Tensor};

#[cfg(test)]
mod tests {}
