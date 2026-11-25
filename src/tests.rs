//! Large-scale tests that train a full model end to end, about a minute each in release mode.
//! They're `#[ignore]`d. Run them with:
//!
//! ```text
//! cargo test --release -- --ignored --nocapture
//! ```

use burn::prelude::*;
use burn::tensor::Distribution;

use crate::{
    dataset::Dataset,
    device::DevicePreference,
    model::ModelConfig,
    train::{OutputMetrics, TrainingConfig},
};

/// Minimum test-set R² for every output. Data and weights are unseeded, so this leaves a wide
/// margin below the usual ~0.999.
const MIN_R2: f32 = 0.99;

fn assert_fits(metrics: &[OutputMetrics], num_outputs: usize) {
    assert_eq!(metrics.len(), num_outputs, "expected one metrics entry per output");
    for (i, m) in metrics.iter().enumerate() {
        assert!(m.r2 > MIN_R2, "output {i}: R² = {} is below {MIN_R2} (mse = {})", m.r2, m.mse);
    }
}

/// Most R² a model can score on `y` (one column) without learning `term` of it: what getting
/// the rest exactly right would give.
const MAX_R2_WITHOUT_A_TERM: f32 = 0.9;

/// Fails if the test could pass without `term`: each one must matter more than `MIN_R2` forgives.
/// A term's average is no loss: a constant can stand in for it.
fn assert_needed(what: &str, term: &Tensor<2>, y: &Tensor<2>) {
    let spread = |t: &Tensor<2>| {
        let mean = t.clone().mean().into_scalar::<f32>();
        t.clone().sub_scalar(mean).powf_scalar(2.0).sum().into_scalar::<f32>()
    };
    let r2 = 1.0 - spread(term) / spread(y);
    assert!(r2 < MAX_R2_WITHOUT_A_TERM, "{what} barely matters: R² is {r2} without it");
}

#[test]
#[ignore = "slow: trains a full model; run with `cargo test --release -- --ignored`"]
fn one_input_one_output() {
    let device = DevicePreference::default().device();

    // y = 50*sin(x/10) + x + 0.01x^2: about three waves across the range, as tall as the rest
    let x = Tensor::<2>::random([10_000, 1], Distribution::Uniform(-100.0, 100.0), &device);
    let wave = (x.clone() / 10.0).sin().mul_scalar(50.0);
    let curve = x.clone().powf_scalar(2.0) * 0.01;
    let y = wave.clone() + x.clone() + curve.clone();
    assert_needed("the wave", &wave, &y);
    assert_needed("the curve", &curve, &y);
    let dataset = Dataset { x, y };

    let model = ModelConfig::stepped(1, 10, 100, 10, 1).init(&device);
    let (_, test) = model.fit(&dataset, &TrainingConfig::new().with_train_ratio(0.6).with_valid_ratio(0.2));
    assert_fits(&test.metrics, 1);
}

#[test]
#[ignore = "slow: trains a full model; run with `cargo test --release -- --ignored`"]
fn three_inputs_two_outputs_at_different_scales() {
    let device = DevicePreference::default().device();
    let n = 10_000;
    let col = |lo, hi| Tensor::<2>::random([n, 1], Distribution::Uniform(lo, hi), &device);

    // 3 inputs on very different ranges
    let (x0, x1, x2) = (col(-3.0, 3.0), col(0.0, 1.0), col(-100.0, 100.0));

    // 2 outputs, ~4 orders of magnitude apart in scale
    // y0 = sin(2*x0) + x1*x2/100            range ~[-2, 2]
    // y1 = 3.5*x2^2 + 5000*x0*x1            range ~[-15,000, 50,000]
    let (wave, product) = ((x0.clone() * 2.0).sin(), x1.clone() * x2.clone() / 100.0);
    let (curve, interaction) = (x2.clone().powf_scalar(2.0) * 3.5, x0.clone() * x1.clone() * 5000.0);
    let (y0, y1) = (wave.clone() + product.clone(), curve.clone() + interaction.clone());
    assert_needed("y0's wave", &wave, &y0);
    assert_needed("y0's product", &product, &y0);
    assert_needed("y1's curve", &curve, &y1);
    assert_needed("y1's interaction", &interaction, &y1);

    let x = Tensor::cat(vec![x0, x1, x2], 1);
    let y = Tensor::cat(vec![y0, y1], 1);
    let dataset = Dataset { x, y };

    let model = ModelConfig::stepped(3, 10, 100, 10, 2).init(&device);
    let (_, test) = model.fit(&dataset, &TrainingConfig::new().with_train_ratio(0.6).with_valid_ratio(0.2));
    assert_fits(&test.metrics, 2);
}