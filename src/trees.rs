//! Gradient-boosted trees as the rating model: LightGBM, through `lightgbm3`.

use std::{error::Error, fs, path::Path};

use lightgbm3::{Booster, Dataset};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::train::{OutputMetrics, Split, TestResults};

const CONFIG_FILE: &str = "trees_config.json";
/// LightGBM's text format
const MODEL_FILE: &str = "trees.txt";

/// LightGBM's settings, under its names where it has one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreesConfig {
    pub num_leaves: u32,
    pub learning_rate: f64,
    /// Share of the inputs each tree may split on
    pub feature_fraction: f64,
    /// Share of the training rows each tree is grown on
    pub bagging_fraction: f64,
    pub min_data_in_leaf: u32,
    pub lambda_l2: f64,
    /// Split thresholds drawn at random, not searched for
    pub extra_trees: bool,
    /// Tree cap if early stopping never triggers
    pub max_trees: u32,
    /// Trees without validation improvement before stopping. The trees since the best are dropped.
    pub stop_patience: u32,
    /// Seeds the rows and inputs each tree sees, and the random thresholds
    pub seed: u64,
}

impl Default for TreesConfig {
    fn default() -> Self {
        TreesConfig {
            num_leaves: 31,
            learning_rate: 0.01,
            feature_fraction: 0.3,
            bagging_fraction: 0.8,
            min_data_in_leaf: 20,
            lambda_l2: 1.0,
            extra_trees: true,
            max_trees: 20_000,
            stop_patience: 300,
            seed: 42,
        }
    }
}

impl TreesConfig {
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn describe(&self) -> String {
        format!(
            "model: gradient-boosted trees (LightGBM), {} leaves, learning rate {}, {} of the inputs and {} of the rows a tree, {} rows a leaf at least, L2 {}{}, stopping {} trees after the best",
            self.num_leaves,
            self.learning_rate,
            self.feature_fraction,
            self.bagging_fraction,
            self.min_data_in_leaf,
            self.lambda_l2,
            if self.extra_trees { ", random thresholds" } else { "" },
            self.stop_patience
        )
    }

    /// Trains on `split`'s training rows of `inputs` (a row of statistics each) and `targets`,
    /// stops by its validation rows, and scores its test rows.
    pub fn train_on(&self, inputs: &[Vec<f32>], targets: &[f32], split: &Split) -> Result<(TrainedTrees, TestResults), Box<dyn Error>> {
        let (train, validation, test) = split;
        say!("{} rows: {} to train on, {} to validate, {} to test", inputs.len(), train.len(), validation.len(), test.len());
        let num_inputs = inputs.first().map_or(0, Vec::len);
        let labels = |rows: &[usize]| rows.iter().map(|&row| targets[row]).collect::<Vec<f32>>();
        let train_set = Dataset::from_slice(&flat(inputs, train), &labels(train), num_inputs as i32, true)?;
        // Binned as the training rows are
        let validation_set =
            Dataset::from_slice_with_reference(&flat(inputs, validation), &labels(validation), num_inputs as i32, true, Some(&train_set))?;
        let parameters = json!({
            "objective": "regression",
            "metric": "l2",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l2": self.lambda_l2,
            "extra_trees": self.extra_trees,
            "num_iterations": self.max_trees,
            "early_stopping_rounds": self.stop_patience,
            // LightGBM's seed is an i32
            "seed": self.seed & 0x7fff_ffff,
            "deterministic": true,
            "force_row_wise": true,
            "num_threads": 1,
            "verbosity": -1,
        });
        let trained = TrainedTrees { booster: Booster::train_with_valid(train_set, Some(validation_set), &parameters)? };

        let scored = |rows: &[usize]| -> Result<(Vec<f32>, OutputMetrics), Box<dyn Error>> {
            let predictions = trained.predict(&rows.iter().map(|&row| inputs[row].clone()).collect::<Vec<_>>())?;
            let metrics = metrics(&predictions, &labels(rows));
            Ok((predictions, metrics))
        };
        let (_, train_metrics) = scored(train)?;
        let (_, validation_metrics) = scored(validation)?;
        let (predictions, test_metrics) = scored(test)?;
        say!(
            "{} trees: train RMS {:.4}, validation RMS {:.4}, test RMS {:.4}",
            trained.num_trees(),
            train_metrics.rms_error(),
            validation_metrics.rms_error(),
            test_metrics.rms_error()
        );
        let results = TestResults { metrics: vec![test_metrics], rows: test.clone(), predictions: predictions.into_iter().map(|p| vec![p]).collect() };
        Ok((trained, results))
    }
}

/// The given rows of `inputs`, end to end.
fn flat(inputs: &[Vec<f32>], rows: &[usize]) -> Vec<f32> {
    rows.iter().flat_map(|&row| inputs[row].iter().copied()).collect()
}

fn metrics(predictions: &[f32], targets: &[f32]) -> OutputMetrics {
    let n = targets.len() as f32;
    let mean = targets.iter().sum::<f32>() / n;
    let variance = targets.iter().map(|y| (y - mean).powi(2)).sum::<f32>() / n;
    let errors = || predictions.iter().zip(targets).map(|(p, y)| (p - y).abs());
    let mse = errors().map(|e| e * e).sum::<f32>() / n;
    OutputMetrics { mse, max_error: errors().fold(0.0, f32::max), r2: 1.0 - mse / variance }
}

pub struct TrainedTrees {
    booster: Booster,
}

impl TrainedTrees {
    pub fn num_trees(&self) -> usize {
        self.booster.num_iterations() as usize
    }

    pub fn num_inputs(&self) -> usize {
        self.booster.num_features() as usize
    }

    /// One prediction a row of inputs.
    pub fn predict(&self, rows: &[Vec<f32>]) -> Result<Vec<f32>, Box<dyn Error>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(row) = rows.iter().find(|row| row.len() != self.num_inputs()) {
            return Err(format!("the trees take {} inputs, not {}", self.num_inputs(), row.len()).into());
        }
        let all: Vec<usize> = (0..rows.len()).collect();
        let predicted = self.booster.predict(&flat(rows, &all), self.num_inputs() as i32, true)?;
        Ok(predicted.into_iter().map(|p| p as f32).collect())
    }

    /// Whether `dir` holds trees saved with [`TrainedTrees::save`].
    pub fn saved_in(dir: impl AsRef<Path>) -> bool {
        dir.as_ref().join(MODEL_FILE).is_file()
    }

    /// Writes the trees and `config`, which they were trained with, into `dir`.
    pub fn save(&self, config: &TreesConfig, dir: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        fs::write(dir.join(CONFIG_FILE), serde_json::to_string_pretty(config)?)?;
        fs::write(dir.join(MODEL_FILE), self.booster.save_string()?)?;
        Ok(())
    }

    pub fn load(dir: impl AsRef<Path>) -> Result<TrainedTrees, Box<dyn Error>> {
        let text = fs::read_to_string(dir.as_ref().join(MODEL_FILE))?;
        Ok(TrainedTrees { booster: Booster::from_string(&text)? })
    }

    /// Deletes what [`TrainedTrees::save`] wrote into `dir`, if anything.
    pub fn remove_from(dir: impl AsRef<Path>) -> std::io::Result<()> {
        for file in [CONFIG_FILE, MODEL_FILE] {
            match fs::remove_file(dir.as_ref().join(file)) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
                _ => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::shuffle;

    /// 600 rows of 3 inputs: the target is a step of the first plus the second, and the third is
    /// noise. Rows dealt 400 / 100 / 100.
    fn steps() -> (Vec<Vec<f32>>, Vec<f32>, Split) {
        let mut order: Vec<usize> = (0..600).collect();
        shuffle(&mut order, 7);
        let inputs: Vec<Vec<f32>> = (0..600).map(|i| vec![(i % 30) as f32, (i / 30) as f32, order[i] as f32]).collect();
        let step = |row: &Vec<f32>| if row[0] < 12.0 { 2.0 } else { 9.0 };
        let targets = inputs.iter().map(|row| step(row) + row[1] * 0.5).collect();
        let test = order.split_off(500);
        let validation = order.split_off(400);
        (inputs, targets, (order, validation, test))
    }

    fn quick() -> TreesConfig {
        TreesConfig { num_leaves: 7, learning_rate: 0.1, feature_fraction: 1.0, min_data_in_leaf: 5, stop_patience: 20, ..TreesConfig::default() }
    }

    #[test]
    fn learns_a_step_and_stops_early() {
        let (inputs, targets, split) = steps();
        let (trained, test) = quick().train_on(&inputs, &targets, &split).unwrap();
        assert_eq!(trained.num_inputs(), 3);
        assert!(trained.num_trees() < quick().max_trees as usize, "{} trees", trained.num_trees());
        assert_eq!(test.rows, split.2);
        assert_eq!(test.predictions.len(), 100);
        assert!(test.metrics[0].rms_error() < 0.5, "{:?}", test.metrics[0]);
        assert!(test.metrics[0].r2 > 0.98, "{:?}", test.metrics[0]);
        // A new row either side of the step
        let predicted = trained.predict(&[vec![3.0, 4.0, 0.0], vec![20.0, 4.0, 0.0]]).unwrap();
        assert!((predicted[0] - 4.0).abs() < 1.0 && (predicted[1] - 11.0).abs() < 1.0, "{predicted:?}");
    }

    #[test]
    fn the_seed_decides_the_trees() {
        let (inputs, targets, split) = steps();
        let fit = |seed| quick().with_seed(seed).train_on(&inputs, &targets, &split).unwrap().1.predictions;
        assert_eq!(fit(1), fit(1));
        assert_ne!(fit(1), fit(2));
    }

    #[test]
    fn rejects_rows_of_another_width() {
        let (inputs, targets, split) = steps();
        let (trained, _) = quick().train_on(&inputs, &targets, &split).unwrap();
        assert!(trained.predict(&[vec![1.0, 2.0]]).is_err());
        assert_eq!(trained.predict(&[]).unwrap(), Vec::<f32>::new());
    }

    #[test]
    fn save_and_load_round_trip() {
        let (inputs, targets, split) = steps();
        let (trained, _) = quick().train_on(&inputs, &targets, &split).unwrap();
        let dir = std::env::temp_dir().join(format!("ddr-trainer-trees-{}", std::process::id()));
        assert!(!TrainedTrees::saved_in(&dir));
        trained.save(&quick(), &dir).unwrap();
        assert!(TrainedTrees::saved_in(&dir));
        let loaded = TrainedTrees::load(&dir).unwrap();
        let config: TreesConfig = serde_json::from_str(&fs::read_to_string(dir.join(CONFIG_FILE)).unwrap()).unwrap();
        TrainedTrees::remove_from(&dir).unwrap();
        let emptied = !TrainedTrees::saved_in(&dir) && fs::read_dir(&dir).unwrap().next().is_none();
        TrainedTrees::remove_from(&dir).unwrap();
        fs::remove_dir(&dir).unwrap();
        assert!(emptied);
        assert_eq!(config, quick());
        assert_eq!((loaded.num_trees(), loaded.num_inputs()), (trained.num_trees(), 3));
        assert_eq!(loaded.predict(&inputs).unwrap(), trained.predict(&inputs).unwrap());
    }
}
