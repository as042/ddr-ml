use std::{error::Error, path::Path};

use burn::{
    module::{Param, ParamId},
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    store::ModuleRecord,
    tensor::Bytes,
};

use crate::{
    dataset::Dataset,
    model::{Model, ModelConfig},
};

/// Hyperparameters for [`Model::fit`].
///
/// The architecture isn't set here: it's whatever model `fit` is called on.
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Adam settings (betas, epsilon, weight decay).
    #[config(default = "AdamConfig::new()")]
    pub optimizer: AdamConfig,

    /// Initial learning rate.
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// Factor the learning rate is multiplied by each time `lr_patience` runs out.
    #[config(default = 0.5)]
    pub lr_decay: f64,
    /// Decay the learning rate after this many epochs without validation improvement.
    #[config(default = 10)]
    pub lr_patience: usize,
    /// Stop after this many epochs without validation improvement.
    #[config(default = 25)]
    pub stop_patience: usize,
    /// An epoch resets `lr_patience` and `stop_patience` only if its validation
    /// loss is at least this fraction below the last loss that reset them.
    /// Without it, tiny gains at a decayed learning rate reset patience forever.
    /// Any gain, however small, still updates the saved best weights.
    #[config(default = 1e-3)]
    pub min_rel_improvement: f64,

    /// Rows per gradient step. A final partial batch is dropped.
    #[config(default = 256)]
    pub batch_size: usize,
    /// Upper bound on epochs if early stopping never triggers.
    #[config(default = 500)]
    pub max_epochs: usize,
    /// Print the validation loss every this many epochs.
    #[config(default = 10)]
    pub log_interval: usize,
}

/// Per-column mean and population standard deviation of `t`, each shaped
/// `[1, num_columns]` so they broadcast against `t`.
fn column_mean_std(t: &Tensor<2>) -> (Tensor<2>, Tensor<2>) {
    let (var, mean) = t.clone().var_mean_bias(0);
    (mean, var.sqrt())
}

/// Accuracy of one output column on the held-out test split, in original units.
#[derive(Debug, Clone, Copy)]
pub struct OutputMetrics {
    pub mse: f32,
    /// Coefficient of determination: 1 is a perfect fit, 0 is no better than
    /// always predicting the column's mean.
    pub r2: f32,
}

/// A trained model plus the normalization it was trained with: everything
/// needed to make predictions from raw data.
///
/// Everything here is in validation mode, ready for inference: no gradient
/// tracking, and training-only behavior (such as dropout) is off.
///
/// Save it with [`TrainedModel::save`] and read it back with [`TrainedModel::load`].
#[derive(Module, Debug)]
pub struct TrainedModel {
    /// Weights with the best validation loss. They map standardized inputs to
    /// standardized outputs; [`TrainedModel::predict`] handles the conversion.
    /// To train further, pass it to [`Model::fit`], which switches its copy
    /// back to training mode.
    pub model: Model,
    /// Per-column training-split statistics, each shaped `[1, num_columns]`.
    /// They're `Param`s only so that saving the module includes them; they're
    /// never trained.
    pub x_mean: Param<Tensor<2>>,
    pub x_std: Param<Tensor<2>>,
    pub y_mean: Param<Tensor<2>>,
    pub y_std: Param<Tensor<2>>,
}

/// Wraps a normalization statistic as an untrainable `Param`, so it's saved
/// and loaded along with the weights.
fn stat_param(t: Tensor<2>) -> Param<Tensor<2>> {
    Param::initialized(ParamId::new(), t.without_autodiff())
}

/// File names [`TrainedModel::save`] writes inside its directory.
const CONFIG_FILE: &str = "model_config.json";
const WEIGHTS_FILE: &str = "model.bpk";

impl TrainedModel {
    /// Saves the model into `dir` (created if needed) as two files:
    /// `model_config.json`, the architecture, and `model.bpk`, the weights plus
    /// normalization statistics in burn's burnpack format.
    ///
    /// `config` must be the [`ModelConfig`] the model was built from: [`TrainedModel::load`]
    /// uses it to rebuild the model before filling in the saved values.
    pub fn save(&self, config: &ModelConfig, dir: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        config.save(dir.join(CONFIG_FILE))?;
        self.clone().save_file(dir.join(WEIGHTS_FILE))?;
        Ok(())
    }

    /// Loads a model saved with [`TrainedModel::save`] from `dir`, in validation
    /// mode and ready for inference.
    pub fn load(dir: impl AsRef<Path>, device: &Device) -> Result<Self, Box<dyn Error>> {
        let dir = dir.as_ref();
        let config = ModelConfig::load(dir.join(CONFIG_FILE))?;

        // Read the whole weights file up front. Loading straight from the path
        // (burn's `load_file`) is lazy: tensor data is read from the file on first
        // use, so deleting or overwriting the file afterwards breaks the loaded model.
        let bytes = std::fs::read(dir.join(WEIGHTS_FILE))?;
        let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(bytes))?;

        let model = config.init(device);
        // Placeholders of the right shapes, overwritten by the saved values.
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

    /// Predicts targets in original units from raw inputs shaped
    /// `[num_rows, num_inputs]`. Inputs may have autodiff enabled; it's
    /// stripped, so no gradient graph is recorded.
    pub fn predict(&self, x: Tensor<2>) -> Tensor<2> {
        let x = (x.without_autodiff() - self.x_mean.val()) / self.x_std.val();
        self.model.forward(x) * self.y_std.val() + self.y_mean.val()
    }
}

impl Model {
    /// Trains a copy of this model on `dataset` and returns it along with its
    /// accuracy on the held-out test split (also printed), one [`OutputMetrics`]
    /// per column of `dataset.y`.
    ///
    /// Training starts from `self`'s current weights and leaves `self` unchanged.
    /// `self` may be in training or validation mode (for example, the `model`
    /// of an earlier [`TrainedModel`]): `fit` trains a training-mode copy.
    /// `dataset` must be on an autodiff device (e.g. from [`DevicePreference::device`](crate::device::DevicePreference::device)),
    /// the same one `self` was created on.
    ///
    /// Rows are split in order (see [`Dataset`]), so they must already be shuffled.
    /// Inputs and targets are standardized per column using training-split
    /// statistics only. The validation split drives learning-rate decay and early
    /// stopping, and the weights with the best validation loss are the ones returned.
    pub fn fit(&self, dataset: &Dataset, config: &TrainingConfig) -> (TrainedModel, Vec<OutputMetrics>) {
        // A mismatched output width can broadcast against the targets and train
        // without any error, so check both widths up front.
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
        let n_total = dataset.x.dims()[0];
        let n_train = (n_total as f64 * dataset.train_ratio).round() as usize;
        let n_val = (n_total as f64 * dataset.valid_ratio).round() as usize;
        let n_test = n_total
            .checked_sub(n_train + n_val)
            .expect("train_ratio + valid_ratio must not exceed 1");
        assert!(
            n_train >= config.batch_size,
            "training split has {n_train} rows, fewer than batch_size ({})",
            config.batch_size
        );
        assert!(n_val > 0 && n_test > 0, "validation and test splits must be non-empty");

        let split = |t: &Tensor<2>| {
            (
                t.clone().narrow(0, 0, n_train),
                t.clone().narrow(0, n_train, n_val),
                t.clone().narrow(0, n_train + n_val, n_test),
            )
        };
        let (x_train, x_val, x_test) = split(&dataset.x);
        let (y_train, y_val, y_test) = split(&dataset.y);

        // Normalization
        let (x_mean, x_std) = column_mean_std(&x_train);
        let (y_mean, y_std) = column_mean_std(&y_train);
        let norm = |t: Tensor<2>, mean: &Tensor<2>, std: &Tensor<2>| (t - mean.clone()) / std.clone();

        let x_train = norm(x_train, &x_mean, &x_std);
        let y_train = norm(y_train, &y_mean, &y_std);
        // Validation runs on validation-mode snapshots of the model, so
        // its data doesn't need gradient tracking.
        let x_val = norm(x_val, &x_mean, &x_std).without_autodiff();
        let y_val = norm(y_val, &y_mean, &y_std).without_autodiff();

        let mut model = self.clone().train();
        let mut optim = config.optimizer.init();
        let mut lr = config.learning_rate;
        let num_batches = n_train / config.batch_size;

        let loss_fn = MseLoss::new();

        let mut best_val = f32::INFINITY;
        let mut best_model = model.valid();
        // Patience counts epochs since the loss last beat `plateau_val` by at
        // least `min_rel_improvement`. Smaller gains still update `best_model`.
        let mut plateau_val = f32::INFINITY;
        let mut epochs_since_improvement = 0;

        for epoch in 1..=config.max_epochs {
            // Training
            for b in 0..num_batches {
                let start = b * config.batch_size;
                let xb = x_train.clone().narrow(0, start, config.batch_size);
                let yb = y_train.clone().narrow(0, start, config.batch_size);

                let loss = loss_fn.forward(model.forward(xb), yb, Reduction::Mean);
                let grads = GradientsParams::from_grads(loss.backward(), &model);
                model = optim.step(lr, model, grads);
            }

            // Validation
            // Scored on a validation-mode snapshot, so no gradient graph is
            // built and training-only behavior like dropout doesn't add noise.
            let snapshot = model.valid();
            let val_loss = loss_fn
                .forward(snapshot.forward(x_val.clone()), y_val.clone(), Reduction::Mean)
                .into_scalar::<f32>();

            if val_loss < best_val {
                best_val = val_loss;
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
                    println!("epoch {epoch:>3}: early stop (best val mse = {best_val:.4e})");
                    break;
                }
            }

            if epoch % config.log_interval == 0 {
                println!("epoch {epoch:>3}  val mse = {val_loss:.4e}  (lr = {lr:e})");
            }
        }

        // Roll back to the weights that scored best on validation. `best_model`
        // is already a validation-mode snapshot, and `stat_param` strips
        // autodiff from the statistics, so the whole result is ready for inference.
        let trained = TrainedModel {
            model: best_model,
            x_mean: stat_param(x_mean),
            x_std: stat_param(x_std),
            y_mean: stat_param(y_mean),
            y_std: stat_param(y_std),
        };

        // Final evaluation
        // Errors are in original units. Outputs are scored separately because a
        // pooled MSE is dominated by whichever output has the largest scale.
        let pred = trained.predict(x_test);
        let y_test = y_test.without_autodiff();
        let to_vec = |t: Tensor<2>| t.into_data().try_to_vec::<f32>().unwrap();
        let mses = to_vec((pred - y_test.clone()).powf_scalar(2.0).mean_dim(0));
        let vars = to_vec(y_test.var_bias(0));
        let metrics: Vec<OutputMetrics> = mses
            .into_iter()
            .zip(vars)
            .map(|(mse, var)| OutputMetrics { mse, r2: 1.0 - mse / var })
            .collect();

        println!("\ntest results over {n_test} held-out rows (original units):");
        println!("output           mse          R²");
        for (i, m) in metrics.iter().enumerate() {
            println!("{i:>6}  {:>12.4e}  {:>10.6}", m.mse, m.r2);
        }

        (trained, metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DevicePreference;

    /// Saving a `TrainedModel` (plus its `ModelConfig`) and loading it back must
    /// reproduce its predictions exactly. No training is needed: a small untrained
    /// model with made-up statistics is enough to check that every saved value
    /// round-trips. If any field failed to load, it would keep the loader's
    /// placeholder (fresh random weights, or zero statistics, which give NaN).
    /// It also checks that a loaded model keeps working after its file is gone.
    #[test]
    fn save_and_load_round_trip() {
        let device = DevicePreference::default().device();
        let config = ModelConfig::new(3, 2, 8, 4, 2);
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
        // Deliberately delete the files before using the reloaded model: a
        // loaded model must not depend on its files still existing.
        std::fs::remove_dir_all(&dir).unwrap();

        let x = Tensor::<2>::from_data([[0.3, -1.0, 2.0], [5.0, 0.0, -3.0], [-4.0, 7.5, 0.1]], &device);
        let original = trained.predict(x.clone());
        let restored = reloaded.predict(x);

        assert!(!restored.is_autodiff(), "the loaded model should be in validation mode");
        let max_diff = (original - restored).abs().max().into_scalar::<f32>();
        assert_eq!(max_diff, 0.0, "the reloaded model's predictions differ from the original's");
    }
}