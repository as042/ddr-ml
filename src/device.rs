use burn::prelude::*;

/// Which hardware to train and run on. `main` and every test use the `#[default]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)] // The other variants are there to switch to
pub enum DevicePreference {
    /// burn's pure-Rust `flex` backend
    #[default]
    Cpu,
    /// wgpu, on whichever graphics API it picks
    Gpu,
    /// burn's first enabled backend (GPUs before CPUs), which `BURN_DEVICE` can override
    Auto,
}

impl DevicePreference {
    /// With autodiff, so it works for training as well as inference.
    pub fn device(self) -> Device {
        match self {
            Self::Cpu => Device::flex(),
            Self::Gpu => Device::wgpu(Default::default()),
            Self::Auto => Device::default(),
        }
        .autodiff()
    }
}