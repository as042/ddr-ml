use burn::prelude::*;

/// Inputs and targets for training, one sample per row.
pub struct Dataset {
    /// Inputs, shaped `[num_rows, num_inputs]`.
    pub x: Tensor<2>,
    /// Targets, shaped `[num_rows, num_outputs]`.
    pub y: Tensor<2>,
}

impl Dataset {
    /// `device` needs autodiff for training, as
    /// [`DevicePreference::device`](crate::device::DevicePreference::device) gives it.
    ///
    /// # Panics
    ///
    /// If `inputs` and `targets` differ in row count, or rows differ in length.
    pub fn from_rows(inputs: &[Vec<f32>], targets: &[Vec<f32>], device: &Device) -> Self {
        assert_eq!(inputs.len(), targets.len(), "{} rows of inputs, but {} of targets", inputs.len(), targets.len());
        let to_tensor = |rows: &[Vec<f32>]| {
            let width = rows.first().map_or(0, Vec::len);
            assert!(rows.iter().all(|row| row.len() == width), "rows must all be the same length");
            Tensor::from_data(TensorData::new(rows.concat(), [rows.len(), width]), device)
        };
        Dataset { x: to_tensor(inputs), y: to_tensor(targets) }
    }
}

/// Shuffles into an order that depends only on `seed`, on every run and machine. Not `rand`,
/// whose `StdRng` algorithm may change between versions.
pub fn shuffle<T>(items: &mut [T], seed: u64) {
    // SplitMix64
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    // Fisher-Yates. The modulo's bias is negligible (around 1e-16).
    for i in (1..items.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DevicePreference;

    #[test]
    fn builds_tensors_from_rows() {
        let device = DevicePreference::default().device();
        let dataset = Dataset::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]], &[vec![5.0], vec![6.0]], &device);
        assert_eq!(dataset.x.dims(), [2, 2]);
        assert_eq!(dataset.y.dims(), [2, 1]);
        assert_eq!(dataset.x.into_data().try_to_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dataset.y.into_data().try_to_vec::<f32>().unwrap(), [5.0, 6.0]);
    }

    #[test]
    fn shuffles_the_same_way_every_time() {
        let shuffled = |seed| {
            let mut items: Vec<usize> = (0..10).collect();
            shuffle(&mut items, seed);
            items
        };
        // Pinned: a change to the shuffle would change every run's split
        assert_eq!(shuffled(42), [0, 9, 5, 8, 6, 4, 7, 2, 1, 3]);
        assert_eq!(shuffled(43), [4, 2, 5, 6, 1, 3, 9, 8, 7, 0]);

        let mut empty: [usize; 0] = [];
        shuffle(&mut empty, 42);
        let mut one = [7];
        shuffle(&mut one, 42);
        assert_eq!(one, [7]);
    }
}
