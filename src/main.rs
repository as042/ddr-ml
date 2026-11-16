mod dataset;
mod device;
mod model;
mod train;
mod sim_loader;
mod chart_analyzer;

#[cfg(test)]
mod tests;

use burn::prelude::*;
use burn::tensor::Distribution;

use dataset::Dataset;
use device::DevicePreference;
use model::ModelConfig;
use train::{TrainedModel, TrainingConfig};

/// The function the network should learn, applied elementwise: 50*sin(x) + x + 3.5x^2.
fn f(x: Tensor<2>) -> Tensor<2> {
    x.clone().sin().mul_scalar(50.0) + x.clone() + x.powf_scalar(2.0) * 3.5
}

fn main() {
    //
    // This is all just a temporary test. This project is WIP!
    //
    let device = DevicePreference::default().device();

    let x = Tensor::<2>::random([10_000, 1], Distribution::Uniform(-100.0, 100.0), &device);
    let y = f(x.clone());
    let dataset = Dataset { x, y, train_ratio: 0.6, valid_ratio: 0.2 };

    let model_config = ModelConfig::new(1, 9, 100, 10, 1);
    let model = model_config.init(&device);
    let (trained, _) = model.fit(&dataset, &TrainingConfig::new());

    let dir = "artifacts";
    trained.save(&model_config, dir).expect("failed to save the model");
    println!("\nsaved the model to {dir}/");
    let reloaded = TrainedModel::load(dir, &device).expect("failed to load the model");
    println!("reloaded the model from {dir}/");

    let x = Tensor::<2>::from_data([[-90.0], [-50.0], [-10.0], [0.0], [10.0], [50.0], [90.0]], &device);
    let original = trained.predict(x.clone());
    let restored = reloaded.predict(x.clone());
    let exact = f(x.clone());

    let max_diff = (original.clone() - restored.clone()).abs().max().into_scalar::<f32>();
    println!("max |original - reloaded| = {max_diff:e}");
    assert_eq!(max_diff, 0.0, "the reloaded model's predictions differ from the original's");

    let to_vec = |t: Tensor<2>| t.into_data().try_to_vec::<f32>().unwrap();
    println!("\n        x     original     reloaded        exact");
    let rows = to_vec(x).into_iter().zip(to_vec(original)).zip(to_vec(restored)).zip(to_vec(exact));
    for (((x, o), r), e) in rows {
        println!("{x:>9.1}  {o:>11.2}  {r:>11.2}  {e:>11.2}");
    }
}