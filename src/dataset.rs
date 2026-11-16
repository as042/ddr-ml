use burn::prelude::*;

/// Inputs and targets for training, one sample per row.
///
/// Rows are split in order: the first `train_ratio` of them train, the next
/// `valid_ratio` validate, and the rest are the test split.
pub struct Dataset {
    /// Inputs, shaped `[num_rows, num_inputs]`.
    pub x: Tensor<2>,
    /// Targets, shaped `[num_rows, num_outputs]`.
    pub y: Tensor<2>,
    pub train_ratio: f64,
    pub valid_ratio: f64,
}