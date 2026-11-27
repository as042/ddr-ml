use std::{error::Error, path::Path};

use burn::{
    module::{Param, ParamId},
    nn::loss::{MseLoss, Reduction},
    optim::{AdamWConfig, GradientsParams},
    prelude::*,
    store::ModuleRecord,
    tensor::{Bytes, Distribution},
};
use serde::{Deserialize, Serialize};

use crate::{
    dataset::{Dataset, scramble, shuffle},
    model::{Model, ModelConfig},
    pca::Pca,
};

/// How [`Model::fit`] rescales each input column, with training-split statistics. Targets are
/// always standardized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum InputScaling {
    /// Subtract the mean, divide by the standard deviation
    #[default]
    Standardize,
    /// Subtract the mean only. For columns whose relative scales carry meaning, such as PCA
    /// components.
    Center,
    /// Use as given
    Unscaled,
}

/// Whether [`ModelConfig::train`] trains on principal components of the inputs. The PCA is
/// fitted on the training split, and the trained model still takes the original inputs.
///
/// [`InputScaling`] then applies to the components. `Standardize` gives the small ones the
/// weight of the large ones, so `Center` usually suits them.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum InputPca {
    #[default]
    Off,
    /// This many leading components
    Components(usize),
    /// The fewest leading components with this share of the variance, above 0 and at most 1
    Variance(f64),
}

impl InputPca {
    /// # Panics
    ///
    /// If off, or the count is 0, more than there are, or reaches a component without variance.
    fn num_components(self, pca: &Pca) -> usize {
        let count = match self {
            InputPca::Off => panic!("the PCA is off"),
            InputPca::Components(count) => count,
            InputPca::Variance(fraction) => pca.components_for_variance(fraction),
        };
        assert!(
            (1..=pca.components.len()).contains(&count),
            "asked for {count} components, but there are 1 to {}",
            pca.components.len()
        );
        // Standardizing a component that is only rounding error would blow it up
        assert!(
            pca.variances[count - 1] > 1e-9 * pca.variances[0],
            "component {count} has no variance: the inputs have fewer independent directions"
        );
        count
    }
}

/// Hyperparameters for [`Model::fit`].
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// AdamW: betas, epsilon, decoupled weight decay (burn default 1e-4)
    #[config(default = "AdamWConfig::new()")]
    pub optimizer: AdamWConfig,

    /// Initial learning rate.
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// Multiplies the learning rate each time `lr_patience` runs out.
    #[config(default = 0.5)]
    pub lr_decay: f64,
    /// Epochs without validation improvement before the learning rate decays.
    #[config(default = 10)]
    pub lr_patience: usize,
    /// Epochs without validation improvement before stopping.
    #[config(default = 25)]
    pub stop_patience: usize,
    /// Relative validation-loss gain needed to reset both patience counters. Smaller gains
    /// still update the best weights.
    #[config(default = 1e-3)]
    pub min_rel_improvement: f64,

    /// Fraction of rows to train on.
    #[config(default = 0.7)]
    pub train_ratio: f64,
    /// Fraction of rows to validate on. The rest are the test split.
    #[config(default = 0.15)]
    pub valid_ratio: f64,
    /// Seeds the row split only.
    #[config(default = 42)]
    pub split_seed: u64,
    /// Seeds each epoch's batch order and, with [`ModelConfig::train`], the starting weights,
    /// dropout and input noise. Vary it with `split_seed` fixed to repeat a run on the same split.
    #[config(default = 42)]
    pub train_seed: u64,

    /// Only [`ModelConfig::train`] can apply it: it sizes the model's inputs to the components.
    #[config(default = "InputPca::Off")]
    pub input_pca: InputPca,
    #[config(default = "InputScaling::Standardize")]
    pub input_scaling: InputScaling,
    /// Std of Gaussian noise added to the scaled training inputs, fresh each batch. Never in
    /// validation, train scoring or the saved model.
    #[config(default = 0.0)]
    pub input_noise: f64,

    /// Rows per gradient step. Rows are reshuffled each epoch, and the partial batch left over
    /// is dropped.
    #[config(default = 256)]
    pub batch_size: usize,
    /// Epoch cap if early stopping never triggers.
    #[config(default = 500)]
    pub max_epochs: usize,
    /// Epochs between loss printouts.
    #[config(default = 1)]
    pub log_interval: usize,
}

/// Per-column mean and population standard deviation, each `[1, num_columns]`. A constant
/// column gets a standard deviation of 1, so standardizing it gives 0s, not NaNs.
fn column_mean_std(t: &Tensor<2>) -> (Tensor<2>, Tensor<2>) {
    let (var, mean) = t.clone().var_mean_bias(0);
    let std = var.sqrt();
    let constant = std.clone().equal_elem(0.0);
    (mean, std.mask_fill(constant, 1.0))
}

/// The mean and standard deviation that `scaling` subtracts and divides by, per input column.
fn input_statistics(x_train: &Tensor<2>, scaling: InputScaling) -> (Tensor<2>, Tensor<2>) {
    let (mean, std) = column_mean_std(x_train);
    match scaling {
        InputScaling::Standardize => (mean, std),
        InputScaling::Center => (mean, std.ones_like()),
        InputScaling::Unscaled => (mean.zeros_like(), std.ones_like()),
    }
}

/// Accuracy of one output column on the test split, in original units.
#[derive(Debug, Clone, Copy)]
pub struct OutputMetrics {
    pub mse: f32,
    /// Largest absolute error of any prediction
    pub max_error: f32,
    /// 1 is a perfect fit, 0 no better than predicting the mean
    pub r2: f32,
}

impl OutputMetrics {
    pub fn rms_error(&self) -> f32 {
        self.mse.sqrt()
    }
}

/// Test-split results from [`Model::fit`].
pub struct TestResults {
    /// One per output column
    pub metrics: Vec<OutputMetrics>,
    /// Dataset rows in the test split, in `predictions` order
    pub rows: Vec<usize>,
    /// Predictions for `rows`, in original units
    pub predictions: Vec<Vec<f32>>,
}

/// Dataset rows to (train on, validate on, test on).
pub type Split = (Vec<usize>, Vec<usize>, Vec<usize>);

/// Seeded random split of `0..num_rows` into (train, validation, test).
pub fn split_rows(num_rows: usize, config: &TrainingConfig) -> Split {
    let n_train = (num_rows as f64 * config.train_ratio).round() as usize;
    let n_val = (num_rows as f64 * config.valid_ratio).round() as usize;
    assert!(n_train + n_val <= num_rows, "train_ratio + valid_ratio must not exceed 1");
    let mut train: Vec<usize> = (0..num_rows).collect();
    shuffle(&mut train, config.split_seed);
    let test = train.split_off(n_train + n_val);
    let val = train.split_off(n_train);
    (train, val, test)
}

/// Training-split positions in batch order for `epoch`: new every epoch, the same every run.
/// The seed is scrambled first, or seed 2's first epoch would be seed 1's second.
fn epoch_order(n_train: usize, seed: u64, epoch: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n_train).collect();
    shuffle(&mut order, scramble(seed).wrapping_add(epoch as u64));
    order
}

/// `x` plus Gaussian noise of standard deviation `std`, from the device's seeded generator.
fn with_noise(x: Tensor<2>, std: f64) -> Tensor<2> {
    if std == 0.0 {
        return x;
    }
    let noise = x.random_like(Distribution::Normal(0.0, std));
    x + noise
}

/// The given rows of `t`, in that order.
fn select_rows(t: &Tensor<2>, rows: &[usize]) -> Tensor<2> {
    let indices: Vec<i64> = rows.iter().map(|&row| row as i64).collect();
    let indices = Tensor::<1, Int>::from_data(TensorData::new(indices, [rows.len()]), &t.device());
    t.clone().select(0, indices)
}

/// A trained model plus its normalization statistics, in validation mode (no gradient
/// tracking, no dropout).
#[derive(Module, Debug)]
pub struct TrainedModel {
    /// Best-validation weights, mapping scaled inputs to standardized outputs.
    /// [`TrainedModel::predict`] converts.
    pub model: Model,
    /// What inputs and outputs are scaled by, each `[1, num_columns]`: training-split
    /// statistics, or 0 and 1 where [`InputScaling`] skips a step. `Param`s only so they're saved
    /// with the module. Never trained.
    pub x_mean: Param<Tensor<2>>,
    pub x_std: Param<Tensor<2>>,
    pub y_mean: Param<Tensor<2>>,
    pub y_std: Param<Tensor<2>>,
}

/// An untrainable `Param`, so the statistic is saved with the weights.
fn stat_param(t: Tensor<2>) -> Param<Tensor<2>> {
    Param::initialized(ParamId::new(), t.without_autodiff())
}

const CONFIG_FILE: &str = "model_config.json";
const WEIGHTS_FILE: &str = "model.bpk";

impl TrainedModel {
    /// Writes the architecture and the weights (with the statistics, in burnpack format) into
    /// `dir`. `config` must be the one the model was built from.
    pub fn save(&self, config: &ModelConfig, dir: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        config.save(dir.join(CONFIG_FILE))?;
        self.clone().save_file(dir.join(WEIGHTS_FILE))?;
        Ok(())
    }

    /// Loads a model saved with [`TrainedModel::save`], in validation mode.
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self, Box<dyn Error>> {
        let dir = dir.as_ref();
        let config = ModelConfig::read(dir.join(CONFIG_FILE))?;

        // Read eagerly: burn's `load_file` reads tensors lazily, so the model would break if
        // the file later changed or was deleted.
        let bytes = std::fs::read(dir.join(WEIGHTS_FILE))?;
        let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(bytes))?;

        let model = config.init(device);
        // Placeholders, overwritten by the record
        let zeros = |n: usize| stat_param(Tensor::zeros([1, n], device));
        let skeleton = TrainedModel {
            x_mean: zeros(model.num_inputs()),
            x_std: zeros(model.num_inputs()),
            y_mean: zeros(model.num_outputs()),
            y_std: zeros(model.num_outputs()),
            model,
        };
        Ok(skeleton.try_load_record(record)?.valid())
    }

    /// For a model trained on the first `num_components` components of `pca`: the same model
    /// taking the inputs the PCA was fitted on. The projection goes into the first layer.
    fn with_pca_inputs(self, pca: &Pca, num_components: usize) -> Self {
        let device = self.x_mean.val().device();
        let width = pca.mean.len();
        let row = |values: &[f64]| {
            let values: Vec<f32> = values.iter().map(|&value| value as f32).collect();
            Tensor::<2>::from_data(TensorData::new(values, [1, width]), &device)
        };
        // [width, num_components]: a component per column
        let loadings: Vec<f32> =
            (0..width).flat_map(|i| pca.components[..num_components].iter().map(move |component| component[i] as f32)).collect();
        let loadings = Tensor::<2>::from_data(TensorData::new(loadings, [width, num_components]), &device);

        // Scaled components = standardized inputs · weights + offsets
        let weights = loadings / self.x_std.val();
        let offsets = (self.x_mean.val() / self.x_std.val()).neg();
        TrainedModel {
            model: self.model.with_input_map(weights, offsets),
            x_mean: stat_param(row(&pca.mean)),
            x_std: stat_param(row(&pca.std)),
            y_mean: self.y_mean,
            y_std: self.y_std,
        }
    }

    /// Predicts targets in original units from raw inputs shaped `[num_rows, num_inputs]`.
    pub fn predict(&self, x: Tensor<2>) -> Tensor<2> {
        let x = (x.without_autodiff() - self.x_mean.val()) / self.x_std.val();
        self.model.forward(x) * self.y_std.val() + self.y_mean.val()
    }
}

impl ModelConfig {
    /// Seeds the device with `config.train_seed`, then builds a model and trains it with
    /// [`Model::fit_on`] on `config`'s split.
    ///
    /// With `config.input_pca` on, the model trains on the components and is then converted to
    /// take `dataset`'s inputs, so it matches this config and saves like any other.
    pub fn train(&self, dataset: &Dataset, config: &TrainingConfig) -> (TrainedModel, TestResults) {
        self.train_on(dataset, config, split_rows(dataset.x.dims()[0], config))
    }

    /// [`ModelConfig::train`] on a given split, not `config`'s.
    pub fn train_on(&self, dataset: &Dataset, config: &TrainingConfig, split: Split) -> (TrainedModel, TestResults) {
        let device = dataset.x.device();
        if config.input_pca == InputPca::Off {
            device.seed(config.train_seed);
            return self.init(&device).fit_on(dataset, config, split);
        }

        let [num_rows, width] = dataset.x.dims();
        assert_eq!(width, self.layer_sizes()[0], "dataset.x has {width} columns but the config takes {} inputs", self.layer_sizes()[0]);
        let values = dataset.x.clone().into_data().try_to_vec::<f32>().unwrap();
        let rows: Vec<Vec<f32>> = values.chunks(width).map(<[f32]>::to_vec).collect();
        let train: Vec<Vec<f32>> = split.0.iter().map(|&row| rows[row].clone()).collect();
        let pca = Pca::fit(&vec!["input"; width], &train);
        let num_components = config.input_pca.num_components(&pca);
        say!(
            "PCA: training on {num_components} of {width} components, {:.1}% of the training split's variance",
            pca.explained_variance_ratio()[..num_components].iter().sum::<f64>() * 100.0
        );

        let scores = pca.transform(&rows, num_components).concat();
        let x = Tensor::from_data(TensorData::new(scores, [num_rows, num_components]), &device);
        let components = Dataset { x, y: dataset.y.clone() };
        let mut sizes = self.layer_sizes().to_vec();
        sizes[0] = num_components;
        let reduced = ModelConfig::new(sizes).with_dropout(self.dropout);
        device.seed(config.train_seed);
        let (trained, test) = reduced.init(&device).fit_on(&components, &config.clone().with_input_pca(InputPca::Off), split);
        (trained.with_pca_inputs(&pca, num_components), test)
    }
}

impl Model {
    /// [`Model::fit_on`] `config`'s split.
    #[cfg(test)]
    pub fn fit(&self, dataset: &Dataset, config: &TrainingConfig) -> (TrainedModel, TestResults) {
        self.fit_on(dataset, config, split_rows(dataset.x.dims()[0], config))
    }

    /// Trains a copy of this model from its current weights, leaving `self` unchanged, on
    /// `split`'s rows. Returns the weights with the best validation loss and their test-split
    /// results (also printed).
    ///
    /// `dataset` must be on the autodiff device `self` was created on. Scaling uses
    /// training-split statistics only. The validation split drives learning-rate decay and
    /// early stopping.
    pub fn fit_on(&self, dataset: &Dataset, config: &TrainingConfig, split: Split) -> (TrainedModel, TestResults) {
        assert!(
            config.input_pca == InputPca::Off,
            "input_pca needs ModelConfig::train, which sizes the model's inputs to the components"
        );
        // A wrong output width would broadcast against the targets and train without error.
        assert_eq!(
            dataset.x.dims()[1],
            self.num_inputs(),
            "dataset.x has {} columns but the model takes {} inputs",
            dataset.x.dims()[1],
            self.num_inputs()
        );
        assert_eq!(
            dataset.y.dims()[1],
            self.num_outputs(),
            "dataset.y has {} columns but the model produces {} outputs",
            dataset.y.dims()[1],
            self.num_outputs()
        );

        let (train_rows, val_rows, test_rows) = split;
        let (n_train, n_val, n_test) = (train_rows.len(), val_rows.len(), test_rows.len());
        assert!(
            n_train >= config.batch_size,
            "training split has {n_train} rows, fewer than batch_size ({})",
            config.batch_size
        );
        assert!(n_val > 0 && n_test > 0, "validation and test splits must be non-empty");

        say!("{} rows: {n_train} to train on, {n_val} to validate, {n_test} to test", dataset.x.dims()[0]);
        say!("train and val mse are on standardized targets, so 1 is no better than predicting the average\n");

        let split = |t: &Tensor<2>| {
            (select_rows(t, &train_rows), select_rows(t, &val_rows), select_rows(t, &test_rows))
        };
        let (x_train, x_val, x_test) = split(&dataset.x);
        let (y_train, y_val, y_test) = split(&dataset.y);

        // Normalization
        let (x_mean, x_std) = input_statistics(&x_train, config.input_scaling);
        let (y_mean, y_std) = column_mean_std(&y_train);
        let norm = |t: Tensor<2>, mean: &Tensor<2>, std: &Tensor<2>| (t - mean.clone()) / std.clone();

        let x_train = norm(x_train, &x_mean, &x_std);
        let y_train = norm(y_train, &y_mean, &y_std);
        let x_val = norm(x_val, &x_mean, &x_std).without_autodiff();
        let y_val = norm(y_val, &y_mean, &y_std).without_autodiff();
        // The training split is scored like validation, to show overfitting.
        let x_train_eval = x_train.clone().without_autodiff();
        let y_train_eval = y_train.clone().without_autodiff();

        let mut model = self.clone().train();
        let mut optim = config.optimizer.init();
        let mut lr = config.learning_rate;
        let num_batches = n_train / config.batch_size;

        let loss_fn = MseLoss::new();
        // MSE in standardized units
        let score = |model: &Model, x: &Tensor<2>, y: &Tensor<2>| {
            loss_fn.forward(model.forward(x.clone()), y.clone(), Reduction::Mean).into_scalar::<f32>()
        };

        let mut best_val = f32::INFINITY;
        let mut best_epoch = 0;
        let mut best_train = None;
        let mut best_model = model.valid();
        // Patience resets only on a `min_rel_improvement` gain over `plateau_val`.
        let mut plateau_val = f32::INFINITY;
        let mut epochs_since_improvement = 0;

        for epoch in 1..=config.max_epochs {
            // Training
            let order = epoch_order(n_train, config.train_seed, epoch);
            let (x_epoch, y_epoch) = (select_rows(&x_train, &order), select_rows(&y_train, &order));
            for b in 0..num_batches {
                let start = b * config.batch_size;
                let xb = with_noise(x_epoch.clone().narrow(0, start, config.batch_size), config.input_noise);
                let yb = y_epoch.clone().narrow(0, start, config.batch_size);

                let loss = loss_fn.forward(model.forward(xb), yb, Reduction::Mean);
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optim.step(lr, model, grads);
            }

            // Validation
            let snapshot = model.valid();
            let val_loss = score(&snapshot, &x_val, &y_val);
            let improved = val_loss < best_val;
            let logging = epoch % config.log_interval == 0;
            // A pass over the training split, so only when it's printed or kept
            let train_loss = (improved || logging).then(|| score(&snapshot, &x_train_eval, &y_train_eval));

            if improved {
                best_val = val_loss;
                best_epoch = epoch;
                best_train = train_loss;
                best_model = snapshot;
            }
            if val_loss < plateau_val * (1.0 - config.min_rel_improvement as f32) {
                plateau_val = val_loss;
                epochs_since_improvement = 0;
            } else {
                epochs_since_improvement += 1;
                if epochs_since_improvement % config.lr_patience == 0 {
                    lr *= config.lr_decay;
                    say!("epoch {epoch:>3}: no improvement for {epochs_since_improvement} epochs, lr -> {lr:e}");
                }
                if epochs_since_improvement >= config.stop_patience {
                    say!("epoch {epoch:>3}: early stop");
                    break;
                }
            }

            if let (true, Some(train_loss)) = (logging, train_loss) {
                say!("epoch {epoch:>3}  train mse = {train_loss:.4e}  val mse = {val_loss:.4e}  (lr = {lr:e})");
            }
        }
        if let Some(best_train) = best_train {
            say!("best epoch {best_epoch}: train mse = {best_train:.4e}, val mse = {best_val:.4e} (the weights kept)");
        }

        // Roll back to the best-validation weights
        let trained = TrainedModel {
            model: best_model,
            x_mean: stat_param(x_mean),
            x_std: stat_param(x_std),
            y_mean: stat_param(y_mean),
            y_std: stat_param(y_std),
        };

        // Final evaluation, in original units. Per output, since the largest-scale output
        // dominates a pooled MSE.
        let pred = trained.predict(x_test);
        let y_test = y_test.without_autodiff();
        let to_vec = |t: Tensor<2>| t.into_data().try_to_vec::<f32>().unwrap();
        let errors = pred.clone() - y_test.clone();
        let mses = to_vec(errors.clone().powf_scalar(2.0).mean_dim(0));
        let max_errors = to_vec(errors.abs().max_dim(0));
        let vars = to_vec(y_test.var_bias(0));
        let metrics: Vec<OutputMetrics> = mses
            .into_iter()
            .zip(max_errors)
            .zip(vars)
            .map(|((mse, max_error), var)| OutputMetrics { mse, max_error, r2: 1.0 - mse / var })
            .collect();

        say!("\ntest results over {n_test} held-out rows (original units):");
        say!("output           mse     rms error     max error          R²");
        for (i, m) in metrics.iter().enumerate() {
            say!("{i:>6}  {:>12.4e}  {:>12.4}  {:>12.4}  {:>10.6}", m.mse, m.rms_error(), m.max_error, m.r2);
        }

        let predictions = to_vec(pred).chunks(self.num_outputs()).map(<[f32]>::to_vec).collect();
        (trained, TestResults { metrics, rows: test_rows, predictions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DevicePreference;
    use crate::model::Activation;

    #[test]
    fn splits_rows_at_random() {
        let config = TrainingConfig::new().with_train_ratio(0.6).with_valid_ratio(0.2).with_split_seed(7);
        let (train, val, test) = split_rows(10, &config);
        assert_eq!((train.len(), val.len(), test.len()), (6, 2, 2));
        // Every row is in exactly one split, in the seeded shuffle's order
        let mut shuffled: Vec<usize> = (0..10).collect();
        shuffle(&mut shuffled, 7);
        assert_eq!([train, val, test].concat(), shuffled);
        assert_ne!(split_rows(10, &config.clone().with_split_seed(8)).0, split_rows(10, &config).0);
        assert_eq!(split_rows(10, &config.clone().with_train_seed(8)), split_rows(10, &config), "train_seed must not move the split");
    }

    #[test]
    fn reorders_batches_each_epoch() {
        let orders: Vec<Vec<usize>> = (1..=5).map(|epoch| epoch_order(10, 7, epoch)).collect();
        for order in &orders {
            let mut sorted = order.clone();
            sorted.sort();
            assert_eq!(sorted, (0..10).collect::<Vec<_>>(), "every row once per epoch");
        }
        assert_ne!(orders[0], orders[1]);
        assert_eq!(orders[0], epoch_order(10, 7, 1), "the same on every run");
        assert_ne!(orders[0], epoch_order(10, 8, 1));
        // A dropped partial batch isn't always the same rows
        assert!(orders.iter().any(|order| order[9] != orders[0][9]));

        // No seed's epochs are another's, a few epochs on
        let all: Vec<Vec<usize>> = (0..6).flat_map(|seed| (1..=40).map(move |epoch| epoch_order(20, seed, epoch))).collect();
        let mut distinct = all.clone();
        distinct.sort();
        distinct.dedup();
        assert_eq!(distinct.len(), all.len());
        assert_ne!(epoch_order(20, 1, 2), epoch_order(20, 2, 1));
    }

    #[test]
    fn adds_noise_of_the_given_size() {
        let device = DevicePreference::default().device();
        device.seed(3);
        let x = Tensor::<2>::ones([1000, 20], &device);
        let unchanged = (with_noise(x.clone(), 0.0) - x.clone()).abs().max().into_scalar::<f32>();
        assert_eq!(unchanged, 0.0);
        let (var, mean) = (with_noise(x.clone(), 0.2) - x).reshape([-1]).var_mean_bias(0);
        let (std, mean) = (var.sqrt().into_scalar::<f32>(), mean.into_scalar::<f32>());
        assert!((std - 0.2).abs() < 0.005 && mean.abs() < 0.005, "std {std}, mean {mean}");
    }

    #[test]
    fn selects_rows_in_order() {
        let device = DevicePreference::default().device();
        let t = Tensor::<2>::from_data([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], &device);
        let rows = select_rows(&t, &[2, 0]).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(rows, [3.0, 30.0, 1.0, 10.0]);
    }

    #[test]
    fn column_mean_std_handles_constant_columns() {
        let device = DevicePreference::default().device();
        let t = Tensor::<2>::from_data([[1.0, 5.0, 0.0], [3.0, 5.0, 0.0]], &device);
        let (mean, std) = column_mean_std(&t);
        let to_vec = |t: Tensor<2>| t.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(to_vec(mean), [2.0, 5.0, 0.0]);
        assert_eq!(to_vec(std), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn scales_inputs_as_configured() {
        let device = DevicePreference::default().device();
        let x = Tensor::<2>::from_data([[1.0, 10.0, 7.0], [3.0, 30.0, 7.0]], &device);
        let to_vec = |t: Tensor<2>| t.into_data().try_to_vec::<f32>().unwrap();
        // (scaling, means, standard deviations)
        let cases = [
            (InputScaling::Standardize, [2.0, 20.0, 7.0], [1.0, 10.0, 1.0]),
            (InputScaling::Center, [2.0, 20.0, 7.0], [1.0, 1.0, 1.0]),
            (InputScaling::Unscaled, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ];
        for (scaling, mean, std) in cases {
            let (actual_mean, actual_std) = input_statistics(&x, scaling);
            assert_eq!(to_vec(actual_mean), mean, "{scaling:?}");
            assert_eq!(to_vec(actual_std), std, "{scaling:?}");
        }
        assert_eq!(TrainingConfig::new().input_scaling, InputScaling::Standardize);
    }

    /// The saved statistics are what `predict` scales by.
    #[test]
    fn fit_keeps_the_configured_input_scaling() {
        let device = DevicePreference::default().device();
        let inputs: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32, 100.0 * (i % 7) as f32]).collect();
        let targets: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32]).collect();
        let dataset = Dataset::from_rows(&inputs, &targets, &device);
        let to_vec = |t: Tensor<2>| t.into_data().try_to_vec::<f32>().unwrap();
        for scaling in [InputScaling::Standardize, InputScaling::Center, InputScaling::Unscaled] {
            let config = TrainingConfig::new().with_batch_size(8).with_max_epochs(1).with_input_scaling(scaling);
            let (trained, _) = ModelConfig::new(vec![2, 4, 1]).train(&dataset, &config);
            let (mean, std) = (to_vec(trained.x_mean.val()), to_vec(trained.x_std.val()));
            assert_eq!(std.iter().all(|&std| std == 1.0), scaling != InputScaling::Standardize, "{scaling:?}: {std:?}");
            assert_eq!(mean.iter().all(|&mean| mean == 0.0), scaling == InputScaling::Unscaled, "{scaling:?}: {mean:?}");
            assert!(to_vec(trained.y_std.val())[0] > 1.0, "targets are always standardized");
        }
    }

    #[test]
    fn trains_through_every_activation() {
        let device = DevicePreference::default().device();
        let inputs: Vec<Vec<f32>> = (0..40).map(|i| vec![(i % 7) as f32 - 3.0, (i % 5) as f32 - 2.0]).collect();
        let targets: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32]).collect();
        let dataset = Dataset::from_rows(&inputs, &targets, &device);
        let config = TrainingConfig::new().with_batch_size(8).with_max_epochs(2).with_input_scaling(InputScaling::Unscaled);
        let outputs = |model: &Model| model.forward(dataset.x.clone().without_autodiff()).into_data().try_to_vec::<f32>().unwrap();
        for activation in Activation::ALL {
            let model = ModelConfig::new(vec![2, 4, 1]).with_activation(activation).init(&device);
            let before = outputs(&model.valid());
            let after = outputs(&model.fit(&dataset, &config).0.model);
            assert!(after.iter().all(|output| output.is_finite()), "{activation:?}: {after:?}");
            assert_ne!(after, before, "{activation:?}");
        }
    }

    #[test]
    fn trains_on_a_given_split() {
        let device = DevicePreference::default().device();
        let inputs: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32, (i % 5) as f32]).collect();
        let targets: Vec<Vec<f32>> = (0..40).map(|i| vec![i as f32]).collect();
        let dataset = Dataset::from_rows(&inputs, &targets, &device);
        let config = TrainingConfig::new().with_batch_size(8).with_max_epochs(2);
        // Every 4th row tested, the next validated, in any order
        let test: Vec<usize> = (0..40).step_by(4).rev().collect();
        let validation: Vec<usize> = (1..40).step_by(4).collect();
        let train = (0..40).filter(|row| row % 4 > 1).collect();
        let (_, results) = ModelConfig::new(vec![2, 4, 1]).train_on(&dataset, &config, (train, validation, test.clone()));
        assert_eq!(results.rows, test);
        assert_eq!(results.predictions.len(), test.len());
    }

    /// Columns 2 and 3 are combinations of 0 and 1, so there are 2 independent directions.
    fn correlated_dataset(device: &Device) -> (Vec<Vec<f32>>, Dataset) {
        let inputs: Vec<Vec<f32>> = (0..60)
            .map(|i| {
                let (a, b) = ((i % 7) as f32, (i % 11) as f32 * 10.0);
                vec![a, b, 3.0 * a + 0.1 * b, 20.0 - a + 0.5 * b]
            })
            .collect();
        let targets: Vec<Vec<f32>> = inputs.iter().map(|row| vec![2.0 * row[0] - 0.05 * row[1]]).collect();
        let dataset = Dataset::from_rows(&inputs, &targets, device);
        (inputs, dataset)
    }

    #[test]
    fn picks_the_number_of_components() {
        let device = DevicePreference::default().device();
        let (inputs, _) = correlated_dataset(&device);
        let pca = Pca::fit(&["a", "b", "c", "d"], &inputs);
        assert_eq!(InputPca::Components(1).num_components(&pca), 1);
        assert_eq!(InputPca::Components(2).num_components(&pca), 2);
        assert_eq!(InputPca::Variance(1.0).num_components(&pca), 2);
        assert_eq!(InputPca::Variance(0.01).num_components(&pca), 1);
        assert_eq!(TrainingConfig::new().input_pca, InputPca::Off);
    }

    #[test]
    #[should_panic(expected = "component 3 has no variance")]
    fn rejects_components_without_variance() {
        let (inputs, _) = correlated_dataset(&DevicePreference::default().device());
        InputPca::Components(3).num_components(&Pca::fit(&["a", "b", "c", "d"], &inputs));
    }

    #[test]
    #[should_panic(expected = "asked for 5 components")]
    fn rejects_more_components_than_inputs() {
        let (inputs, _) = correlated_dataset(&DevicePreference::default().device());
        InputPca::Components(5).num_components(&Pca::fit(&["a", "b", "c", "d"], &inputs));
    }

    #[test]
    #[should_panic(expected = "input_pca needs ModelConfig::train")]
    fn fit_alone_rejects_a_pca() {
        let device = DevicePreference::default().device();
        let (_, dataset) = correlated_dataset(&device);
        let config = TrainingConfig::new().with_batch_size(8).with_input_pca(InputPca::Components(2));
        ModelConfig::new(vec![4, 8, 1]).init(&device).fit(&dataset, &config);
    }

    /// The test predictions come from the model on components. The returned model must give
    /// the same from the original inputs, before and after saving.
    #[test]
    fn a_model_trained_on_components_takes_the_original_inputs() {
        let device = DevicePreference::default().device();
        let (inputs, dataset) = correlated_dataset(&device);
        let model_config = ModelConfig::new(vec![4, 8, 1]);
        for (pca, scaling) in [
            (InputPca::Components(2), InputScaling::Center),
            (InputPca::Components(1), InputScaling::Standardize),
            (InputPca::Variance(0.999), InputScaling::Unscaled),
        ] {
            let config = TrainingConfig::new().with_batch_size(8).with_max_epochs(3).with_input_pca(pca).with_input_scaling(scaling);
            let (trained, test) = model_config.train(&dataset, &config);
            assert_eq!(trained.model.num_inputs(), 4, "{pca:?}");
            assert!(!trained.predict(dataset.x.clone()).is_autodiff(), "{pca:?}");

            let test_inputs: Vec<f32> = test.rows.iter().flat_map(|&row| inputs[row].clone()).collect();
            let x = Tensor::<2>::from_data(TensorData::new(test_inputs, [test.rows.len(), 4]), &device);
            let expected: Vec<f32> = test.predictions.iter().map(|row| row[0]).collect();
            let check = |model: &TrainedModel, what: &str| {
                let actual = model.predict(x.clone()).into_data().try_to_vec::<f32>().unwrap();
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!((actual - expected).abs() < 1e-3, "{pca:?} {what}: predicted {actual}, not {expected}");
                }
            };
            check(&trained, "trained");

            let dir = std::env::temp_dir().join(format!("ddr-trainer-pca-model-{}-{:?}", std::process::id(), scaling));
            trained.save(&model_config, &dir).unwrap();
            let reloaded = TrainedModel::load(&dir, &device).unwrap();
            std::fs::remove_dir_all(&dir).unwrap();
            check(&reloaded, "reloaded");
        }
    }

    /// An untrained model with made-up statistics is enough: a field that failed to load would
    /// keep the loader's placeholder (random weights, or zero statistics, which give NaN).
    #[test]
    fn save_and_load_round_trip() {
        let device = DevicePreference::default().device();
        let config = ModelConfig::stepped(3, 2, 8, 4, 2);
        let trained = TrainedModel {
            model: config.init(&device).valid(),
            x_mean: stat_param(Tensor::from_data([[1.0, -2.0, 3.0]], &device)),
            x_std: stat_param(Tensor::from_data([[0.5, 2.0, 4.0]], &device)),
            y_mean: stat_param(Tensor::from_data([[10.0, -20.0]], &device)),
            y_std: stat_param(Tensor::from_data([[3.0, 0.25]], &device)),
        };

        let dir = std::env::temp_dir().join(format!("ddr-trainer-round-trip-{}", std::process::id()));
        trained.save(&config, &dir).unwrap();
        let reloaded = TrainedModel::load(&dir, &device).unwrap();
        // A loaded model must not depend on its files still existing.
        std::fs::remove_dir_all(&dir).unwrap();

        let x = Tensor::<2>::from_data([[0.3, -1.0, 2.0], [5.0, 0.0, -3.0], [-4.0, 7.5, 0.1]], &device);
        let original = trained.predict(x.clone());
        let restored = reloaded.predict(x);

        assert!(!restored.is_autodiff(), "the loaded model should be in validation mode");
        let max_diff = (original - restored).abs().max().into_scalar::<f32>();
        assert_eq!(max_diff, 0.0, "the reloaded model's predictions differ from the original's");
    }
}