use burn::prelude::*;

/// Which hardware to train and run on.
///
/// `main` and every test use `DevicePreference::default()`, so changing the
/// `#[default]` variant below changes it everywhere.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)] // Only the default variant is constructed; the others are there to switch to.
pub enum DevicePreference {
    /// The pure-Rust CPU backend (burn's `flex`). For this project's small model
    /// it trained about 3x faster than this laptop's integrated GPU.
    #[default]
    Cpu,
    /// The GPU through wgpu, on whichever graphics API wgpu picks.
    Gpu,
    /// burn's own choice: the first enabled backend in its priority order (GPUs
    /// before CPUs), which the `BURN_DEVICE` environment variable can override.
    Auto,
}

impl DevicePreference {
    /// The device for this preference, with autodiff enabled so it works for
    /// training as well as inference.
    pub fn device(self) -> Device {
        match self {
            Self::Cpu => Device::flex(),
            Self::Gpu => Device::wgpu(Default::default()),
            Self::Auto => Device::default(),
        }
        .autodiff()
    }
}