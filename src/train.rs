use std::{error::Error, path::Path};

use burn::{
    module::{Param, ParamId},
    nn::loss::{MseLoss, Reduction},
    optim::{AdamWConfig, GradientsParams},
    prelude::*,
    store::ModuleRecord,
    tensor::{Bytes, Distribution},
};

use crate::{
    dataset::{Dataset, shuffle},
    model::{Model, ModelConfig},
};

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

    /// Std of Gaussian noise added to standardized training inputs, fresh each batch. Never in
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

/// Seeded random split of `0..num_rows` into (train, validation, test).
fn split_rows(num_rows: usize, config: &TrainingConfig) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
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
fn epoch_order(n_train: usize, seed: u64, epoch: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n_train).collect();
    shuffle(&mut order, seed.wrapping_add(epoch as u64));
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
    /// Best-validation weights, mapping standardized inputs to standardized outputs.
    /// [`TrainedModel::predict`] converts.
    pub model: Model,
    /// Training-split statistics, each `[1, num_columns]`. `Param`s only so they're saved with
    /// the module. Never trained.
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
    #[allow(dead_code)] // Only tests use it
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self, Box<dyn Error>> {
        let dir = dir.as_ref();
        let config = ModelConfig::load(dir.join(CONFIG_FILE))?;

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

    /// Predicts targets in original units from raw inputs shaped `[num_rows, num_inputs]`.
    pub fn predict(&self, x: Tensor<2>) -> Tensor<2> {
        let x = (x.without_autodiff() - self.x_mean.val()) / self.x_std.val();
        self.model.forward(x) * self.y_std.val() + self.y_mean.val()
    }
}

impl ModelConfig {
    /// Seeds the device with `config.train_seed`, then builds a model and trains it with
    /// [`Model::fit`].
    pub fn train(&self, dataset: &Dataset, config: &TrainingConfig) -> (TrainedModel, TestResults) {
        let device = dataset.x.device();
        device.seed(config.train_seed);
        self.init(&device).fit(dataset, config)
    }
}

impl Model {
    /// Trains a copy of this model from its current weights, leaving `self` unchanged. Returns
    /// the weights with the best validation loss and their test-split results (also printed).
    ///
    /// `dataset` must be on the autodiff device `self` was created on. Standardization uses
    /// training-split statistics only. The validation split drives learning-rate decay and
    /// early stopping.
    pub fn fit(&self, dataset: &Dataset, config: &TrainingConfig) -> (TrainedModel, TestResults) {
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

        // Split rows into train/validation/test
        let (train_rows, val_rows, test_rows) = split_rows(dataset.x.dims()[0], config);
        let (n_train, n_val, n_test) = (train_rows.len(), val_rows.len(), test_rows.len());
        assert!(
            n_train >= config.batch_size,
            "training split has {n_train} rows, fewer than batch_size ({})",
            config.batch_size
        );
        assert!(n_val > 0 && n_test > 0, "validation and test splits must be non-empty");

        println!("{} rows: {n_train} to train on, {n_val} to validate, {n_test} to test", dataset.x.dims()[0]);
        println!("train and val mse are on standardized targets, so 1 is no better than predicting the average\n");

        let split = |t: &Tensor<2>| {
            (select_rows(t, &train_rows), select_rows(t, &val_rows), select_rows(t, &test_rows))
        };
        let (x_train, x_val, x_test) = split(&dataset.x);
        let (y_train, y_val, y_test) = split(&dataset.y);

        // Normalization
        let (x_mean, x_std) = column_mean_std(&x_train);
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
                    println!("epoch {epoch:>3}: no improvement for {epochs_since_improvement} epochs, lr -> {lr:e}");
                }
                if epochs_since_improvement >= config.stop_patience {
                    println!("epoch {epoch:>3}: early stop");
                    break;
                }
            }

            if let (true, Some(train_loss)) = (logging, train_loss) {
                println!("epoch {epoch:>3}  train mse = {train_loss:.4e}  val mse = {val_loss:.4e}  (lr = {lr:e})");
            }
        }
        if let Some(best_train) = best_train {
            println!("best epoch {best_epoch}: train mse = {best_train:.4e}, val mse = {best_val:.4e} (the weights kept)");
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

        println!("\ntest results over {n_test} held-out rows (original units):");
        println!("output           mse     rms error     max error          R²");
        for (i, m) in metrics.iter().enumerate() {
            println!("{i:>6}  {:>12.4e}  {:>12.4}  {:>12.4}  {:>10.6}", m.mse, m.rms_error(), m.max_error, m.r2);
        }

        let predictions = to_vec(pred).chunks(self.num_outputs()).map(<[f32]>::to_vec).collect();
        (trained, TestResults { metrics, rows: test_rows, predictions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DevicePreference;

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