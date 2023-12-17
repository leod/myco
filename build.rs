use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() {
    for kernel in std::fs::read_dir("kernels").expect("Error finding kernels folder") {
        let path = kernel.expect("Invalid path in kernels folder").path();
        SpirvBuilder::new(path, "spirv-unknown-vulkan1.1")
            .print_metadata(MetadataPrintout::Full)
            .build()
            .expect("Kernel failed to compile");
    }
}
