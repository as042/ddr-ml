//! The command-line tools (see `main`). Each is a script over the rest of the crate.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    error::Error,
    fmt::Write as _,
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use burn::optim::AdamWConfig;
use burn::prelude::*;
use serde::{Deserialize, Serialize};

use crate::chart_analyzer::{
    AnalyzedSegment, COSTS, ChartStatistics, Costs, Foot, FootedSegment, MAX_COST, SETTINGS, Settings, Source,
    Technique, analyze_chart, analyze_singles, foot_arrows, foot_chart, footing_segments,
};
#[cfg(test)]
use crate::chart_analyzer::{Cost, analyze_single_chart};
use crate::cma_es;
use crate::cost_fit::{self, COLUMNS, SCORED, Sheet};
use crate::dataset::{Dataset, scramble, shuffle};
use crate::device::DevicePreference;
use crate::model::{Activation, ModelConfig};
use crate::pca::Pca;
use crate::sim_loader::{ChartDifficulty, Note, SimFile, SingleChart, StepChart, simfiles_in};
use crate::stats_cache::{StatsCache, analyzer_fingerprint};
use crate::train::{self, Split, TrainedModel, TrainingConfig};

const CACHE_FILE: &str = "artifacts/stats_cache.json";
/// Saved models are folders in here.
const ARTIFACTS: &str = "artifacts";
/// `infer`'s model when it isn't given one.
const DEFAULT_MODEL: &str = "a3_model3";
/// In a model's folder (see `ModelInputs`).
const INPUTS_FILE: &str = "inputs.json";
/// In a model's folder (see `SplitRecord`).
const SPLIT_FILE: &str = "split.json";
const PCA_FILE: &str = "artifacts/pca.json";
/// Untracked (copyrighted), see data/.gitignore.
const PACKS_DIR: &str = "data/packs";
/// Default training data, oldest first (see `current_charts`). The older packs are steps only,
/// with their song folders one folder deeper (`song_root`).
const DATASET: [(&str, Songs); 22] = [
    ("DDR (AC JP) (steps)", Songs::Current), // 1st Mix
    ("DDR 2ndMIX (AC JP) (steps)", Songs::Current),
    ("DDR 2ndMIX LINK v(AC JP) (steps)", Songs::Current),
    ("DDR 3rdMIX (AC JP) (steps)", Songs::Current),
    ("DDR 3rdMIX PLUS (AC JP) (steps)", Songs::Current),
    ("DDR 4thMIX (AC JP) (steps)", Songs::Current),
    ("DDR 5thMIX (AC JP) (steps)", Songs::Current),
    ("DDR MAX 6thMIX (AC JP) (steps)", Songs::Current),
    ("DDR MAX2 7thMIX (AC JP) (steps)", Songs::Current),
    ("DDR EXTREME (AC JP) (steps)", Songs::Current),
    ("DDR SuperNOVA (AC INTL) (steps)", Songs::Current),
    ("DDR SuperNOVA2 (AC JP) (steps)", Songs::Current),
    ("DDR X (AC JP) (steps)", Songs::Current),
    ("DDR X2 (AC JP) (steps)", Songs::Current),
    ("DDR X3 VS 2ndMIX (AC JP) (steps)", Songs::Current),
    ("DDR 2013 (AC JP) (steps)", Songs::Current),
    ("DDR 2014 (AC JP) (steps)", Songs::Current),
    ("DDR A (AC INTL)", Songs::All),
    ("DDR A20 (AC INTL)", Songs::All),
    ("DDR A20 PLUS (AC INTL)", Songs::All),
    ("DDR A3 (AC INTL)", Songs::All),
    ("DDR WORLD (AC INTL)", Songs::All),
];
/// One song in today's game, but several in the older packs (4 types of Tohoku EVOLVED in X3,
/// 2 sides of SUNKiSS♥DROP in SuperNOVA2), and which is today's can't be told: left out.
const EXCLUDED_SONGS: [&str; 2] = ["Tohoku EVOLVED", "SUNKiSS♥DROP"];
/// The pack the chart spreadsheet counts.
const SHEET_PACK: &str = "DDR A3 (AC INTL)";
/// Sanbai's ratings of the current game's charts (see `Sanbai`).
const SANBAI_FILE: &str = "data/sanbai/sanbai_sp_all.csv";
/// `crossval`'s, when it isn't given `--folds`.
const DEFAULT_FOLDS: usize = 5;
/// In `crossval`'s folder: every chart's out-of-fold prediction.
const PREDICTIONS_FILE: &str = "predictions.csv";
/// In `crossval`'s folder (see `CrossvalRecord`).
const CROSSVAL_FILE: &str = "crossval.json";
/// `crossval` prints every 10th epoch of each fold.
const CROSSVAL_LOG_INTERVAL: usize = 10;
/// `train`'s (see `model_setup`).
const WEIGHT_DECAY: f32 = 1.0;
const INPUT_NOISE: f64 = 0.2;
/// Standard deviations (see `ModelConfig::input_clip`)
const INPUT_CLIP: f64 = 10.0;
/// The first seed that put all 5 [ ]DENTITY charts in the test split on the five A-era packs
/// alone (643 did it for the 4 on DDR A3 alone). `DATASET` has more now.
const SPLIT_SEED: u64 = 37565;
/// Test-set predictions to print at the end.
const EXAMPLES: usize = 12;
/// An official rating N spans N.0 up to N + 1 on the model's scale. A chart trains as its middle.
const LEVEL_MIDDLE: f32 = 0.5;
const SCALE_NOTE: &str = "predicted: 14.00 is the easiest 14, 14.50 a typical 14";

/// Which of a pack's songs are training data.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Songs {
    /// All: the A-era packs, rated on today's scale when they came out
    All,
    /// Only songs in today's game at today's ratings (`current_charts`): the older packs, whose
    /// other songs' ratings can't be checked
    Current,
}

/// A folder of song folders, and which of its songs are training data.
struct Pack {
    folder: PathBuf,
    songs: Songs,
}

struct Sample {
    name: String,
    /// The file the chart is in
    simfile: PathBuf,
    /// Its pack's place in the packs it was loaded from
    pack: usize,
    difficulty: ChartDifficulty,
    features: Vec<f32>,
    rating: f32,
}

/// An error if `charts` charts split too small for a tool: fewer than `min_train` to train on,
/// or none to validate or test on. The trainer and the PCA would panic.
fn enough_charts(charts: usize, config: &TrainingConfig, min_train: usize) -> Result<(), String> {
    let (train, val, test) = train::split_rows(charts, config);
    if train.len() >= min_train && !val.is_empty() && !test.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{charts} charts are too few: they split into {} to train on, {} to validate and {} to test, and it takes {min_train} to train on and some of each",
        train.len(),
        val.len(),
        test.len()
    ))
}

/// The statistics a saved model was trained on, saved beside it: the model itself only knows
/// how many it takes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ModelInputs {
    /// `ChartStatistics::features`' names, in order
    features: Vec<String>,
    /// `analyzer_fingerprint` at the time
    analyzer: u64,
}

impl ModelInputs {
    fn now(features: Vec<String>) -> ModelInputs {
        ModelInputs { features, analyzer: analyzer_fingerprint() }
    }

    fn save(&self, dir: &Path) -> Result<(), Box<dyn Error>> {
        Ok(fs::write(dir.join(INPUTS_FILE), serde_json::to_string_pretty(self)?)?)
    }

    fn load(dir: &Path) -> Result<ModelInputs, Box<dyn Error>> {
        Ok(serde_json::from_str(&fs::read_to_string(dir.join(INPUTS_FILE))?)?)
    }

    /// Whether a model trained on these can take `now`: an error if the statistics differ in
    /// name, number or order, and a warning if only the analyzer's source does (the same names
    /// may no longer mean the same).
    fn check(&self, now: &ModelInputs) -> Result<Option<String>, String> {
        if self.features != now.features {
            let differs = self.features.iter().zip(&now.features).position(|(then, now)| then != now);
            let how = match differs {
                Some(i) => format!("statistic {} was {}, and is now {}", i + 1, self.features[i], now.features[i]),
                None => format!("it took {} statistics, and charts now have {}", self.features.len(), now.features.len()),
            };
            return Err(format!("the model was trained on other statistics ({how}): train it again"));
        }
        let changed = self.analyzer != now.analyzer;
        Ok(changed.then(|| {
            "the chart analyzer's source has changed since the model was trained, so its statistics may \
             not mean what they did: train it again to be sure"
                .to_string()
        }))
    }
}

/// Which charts a model trained, validated and was tested on, saved with it.
#[derive(Serialize)]
struct SplitRecord<'a> {
    split_seed: u64,
    train_ratio: f64,
    valid_ratio: f64,
    /// Each split in its own order (`test` in the order `train` prints its examples)
    train: Vec<SplitChart<'a>>,
    validation: Vec<SplitChart<'a>>,
    test: Vec<SplitChart<'a>>,
}

#[derive(Serialize)]
struct SplitChart<'a> {
    chart: &'a str,
    rating: f32,
    simfile: String,
    /// In the samples `train` loaded
    #[serde(skip)]
    row: usize,
}

impl<'a> SplitRecord<'a> {
    /// `training`'s split of `samples`, as `ModelConfig::train` makes it.
    fn new(samples: &'a [Sample], training: &TrainingConfig) -> SplitRecord<'a> {
        let (train, validation, test) = train::split_rows(samples.len(), training);
        let charts = |rows: Vec<usize>| -> Vec<SplitChart<'a>> {
            let chart = |row: usize| {
                let sample = &samples[row];
                SplitChart { chart: &sample.name, rating: sample.rating, simfile: sample.simfile.display().to_string(), row }
            };
            rows.into_iter().map(chart).collect()
        };
        SplitRecord {
            split_seed: training.split_seed,
            train_ratio: training.train_ratio,
            valid_ratio: training.valid_ratio,
            train: charts(train),
            validation: charts(validation),
            test: charts(test),
        }
    }
}

/// Trains a model to predict DDR chart ratings from chart statistics, on every dance-single
/// chart in a folder of song folders (the argument, or the `DATASET` packs), and saves it to
/// `--name`'s folder in `ARTIFACTS`, or a new `new_run_dir`, with its split (`SPLIT_FILE`).
pub fn train(args: &[String]) -> Result<(), Box<dyn Error>> {
    let (name, songs) = take_option(args, "--name")?;
    // Before training, which takes minutes
    let named_dir = name.map(|name| model_dir(&name)).transpose()?;
    let (feature_names, samples) = cached_samples(&song_packs(&songs)?)?;

    // Small batches: with only a few hundred charts, they make more steps per epoch.
    let inputs: Vec<Vec<f32>> = samples.iter().map(|sample| sample.features.clone()).collect();
    let targets: Vec<Vec<f32>> = samples.iter().map(|sample| vec![sample.rating + LEVEL_MIDDLE]).collect();
    let dataset = Dataset::from_rows(&inputs, &targets, &DevicePreference::default().device());
    let (model_config, training) = model_setup(inputs[0].len());
    enough_charts(samples.len(), &training, training.batch_size)?;
    say!("{}", describe_model(&model_config));
    let (trained, test) = model_config.train(&dataset, &training);
    let dir = match named_dir {
        Some(dir) => dir,
        None => new_run_dir(Path::new(ARTIFACTS), "model", SystemTime::now())?,
    };
    trained.save(&model_config, &dir)?;
    ModelInputs::now(feature_names).save(&dir)?;
    let split = SplitRecord::new(&samples, &training);
    // Recomputed, so checked against the split trained on
    if split.test.iter().map(|chart| chart.row).ne(test.rows.iter().copied()) {
        return Err(format!("the split recomputed for {SPLIT_FILE} isn't the one trained on").into());
    }
    fs::write(dir.join(SPLIT_FILE), serde_json::to_string_pretty(&split)?)?;
    say!("\nsaved the model and its split to {}/\n", dir.display());

    say!("{:<48} {:>6} {:>9}", "test chart", "rating", "predicted");
    for (&row, predicted) in test.rows.iter().zip(&test.predictions).take(EXAMPLES) {
        say!("{:<48} {:>6} {:>9.2}", samples[row].name, samples[row].rating, predicted[0]);
    }
    let accuracy = test.metrics[0];
    say!(
        "\n{SCALE_NOTE}\nover all {} test charts: typically {:.2} levels from the middle of their rating (RMS), at most {:.2}",
        test.rows.len(),
        accuracy.rms_error(),
        accuracy.max_error
    );
    Ok(())
}

/// `train`'s model and training settings, which `crossval` uses too.
fn model_setup(num_inputs: usize) -> (ModelConfig, TrainingConfig) {
    let model_config =
        ModelConfig::funnel(num_inputs, 1, 2048, 1).with_activation(Activation::Gelu).with_dropout(0.2).with_input_clip(INPUT_CLIP);
    let training = TrainingConfig::new()
        .with_batch_size(32)
        .with_input_noise(INPUT_NOISE)
        .with_split_seed(SPLIT_SEED)
        .with_optimizer(AdamWConfig::new().with_weight_decay(WEIGHT_DECAY));
    (model_config, training)
}

fn describe_model(model_config: &ModelConfig) -> String {
    format!(
        "model: {} (inputs, hidden layers, output), {} activation, dropout {}, weight decay {WEIGHT_DECAY}, input noise {INPUT_NOISE}, inputs capped at {} sd",
        model_config.shape(),
        model_config.activation.name(),
        model_config.dropout,
        model_config.input_clip
    )
}

/// What a `crossval` run was, saved with its predictions.
#[derive(Serialize)]
struct CrossvalRecord {
    folds: usize,
    split_seed: u64,
    train_seed: u64,
    /// The statistics `--exclude` left out of the model's inputs
    excluded: Vec<String>,
    /// How many it took
    statistics: usize,
    /// `Activation::name`
    activation: &'static str,
    /// `analyzer_fingerprint` at the time
    analyzer: u64,
}

/// Cross-validates `train`'s model on a folder of song folders (the argument, or the `DATASET`
/// packs). The songs are dealt into `--folds` folds (`DEFAULT_FOLDS`), and each fold's charts are
/// predicted by a model trained without its songs (`fold_split`). Prints how far the predictions
/// are from the official ratings, and from Sanbai's (`SANBAI_FILE`) where it has the chart, and
/// saves them all (`PREDICTIONS_FILE`), with what the run was (`CROSSVAL_FILE`), in `--name`'s
/// folder in `ARTIFACTS`, or a new one.
///
/// `--seed` trains with another seed (`TrainingConfig::train_seed`) on the same folds.
/// `--exclude` leaves statistics out of the model's inputs (`kept_columns`).
/// `--activation` follows the hidden layer with another `Activation` than `model_setup`'s.
pub fn crossval(args: &[String]) -> Result<(), Box<dyn Error>> {
    let ([folds, seed, exclude, activation, name], songs) =
        take_options(args, ["--folds", "--seed", "--exclude", "--activation", "--name"])?;
    let folds = match folds {
        None => DEFAULT_FOLDS,
        Some(folds) => {
            let parsed = folds.parse().ok().filter(|&folds: &usize| folds >= 2);
            parsed.ok_or_else(|| format!("--folds takes a whole number, 2 or more, not {folds}"))?
        }
    };
    let seed = seed.map(|seed| seed.parse::<u64>().map_err(|_| format!("--seed takes a whole number, 0 or more, not {seed}")));
    let seed = seed.transpose()?;
    let activation = activation.map(|name| name.parse::<Activation>().map_err(|e| format!("--activation: {e}"))).transpose()?;
    // Before training, which takes minutes
    let named_dir = name.map(|name| model_dir(&name)).transpose()?;
    let patterns: Vec<&str> = exclude.as_deref().map_or(Vec::new(), |list| list.split(',').map(str::trim).collect());
    let (kept, excluded) = kept_columns(&statistic_names(), &patterns)?;
    let (_, samples) = cached_samples(&song_packs(&songs)?)?;
    let (song_of, num_songs) = song_ids(&samples);
    if num_songs < folds {
        return Err(format!("{num_songs} songs are too few for {folds} folds").into());
    }

    let inputs: Vec<Vec<f32>> = samples.iter().map(|sample| kept.iter().map(|&column| sample.features[column]).collect()).collect();
    let targets: Vec<Vec<f32>> = samples.iter().map(|sample| vec![sample.rating + LEVEL_MIDDLE]).collect();
    let dataset = Dataset::from_rows(&inputs, &targets, &DevicePreference::default().device());
    let (model_config, training) = model_setup(kept.len());
    let model_config = match activation {
        Some(activation) => model_config.with_activation(activation),
        None => model_config,
    };
    let training = training.with_log_interval(CROSSVAL_LOG_INTERVAL);
    let training = match seed {
        Some(seed) => training.with_train_seed(seed),
        None => training,
    };
    let fold_of_song = song_folds(num_songs, folds, training.split_seed);
    say!("{}", describe_model(&model_config));
    if !excluded.is_empty() {
        say!("{} statistics left out: {}", excluded.len(), excluded.join(", "));
    }
    say!(
        "{num_songs} songs (a cover counts as the song it covers) in {folds} folds, split seed {}, train seed {}",
        training.split_seed,
        training.train_seed
    );

    let mut predicted = vec![f64::NAN; samples.len()];
    for fold in 0..folds {
        let split = fold_split(&song_of, &fold_of_song, fold, training.valid_ratio, training.split_seed);
        if split.0.len() < training.batch_size || split.1.is_empty() || split.2.is_empty() {
            return Err(format!("fold {} is too small to train, validate and test on", fold + 1).into());
        }
        let test_songs = fold_of_song.iter().filter(|&&of| of == fold).count();
        say!("\n== fold {} of {folds}: {test_songs} songs held out", fold + 1);
        let (_, test) = model_config.train_on(&dataset, &training, split);
        for (&row, prediction) in test.rows.iter().zip(&test.predictions) {
            predicted[row] = prediction[0] as f64;
        }
    }
    let middle = |row: usize| (samples[row].rating + LEVEL_MIDDLE) as f64;
    let errors: Vec<f64> = (0..samples.len()).map(|row| predicted[row] - middle(row)).collect();

    let rms = |errors: &[f64]| (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    say!(
        "\n{SCALE_NOTE}\nout of fold, all {} charts: typically {:.3} levels from the middle of their rating (RMS), at most {:.2}",
        samples.len(),
        rms(&errors),
        errors.iter().fold(0.0, |most: f64, e| most.max(e.abs()))
    );
    say!("{:<8} {:>6} {:>7} {:>9}", "levels", "charts", "RMS", "mean err");
    for (label, levels) in [("1-4", 1.0..=4.0), ("5-8", 5.0..=8.0), ("9-11", 9.0..=11.0), ("12-14", 12.0..=14.0), ("15+", 15.0..=f32::MAX)] {
        let band: Vec<f64> = (0..samples.len()).filter(|&row| levels.contains(&samples[row].rating)).map(|row| errors[row]).collect();
        if !band.is_empty() {
            say!("{label:<8} {:>6} {:>7.3} {:>+9.3}", band.len(), rms(&band), band.iter().sum::<f64>() / band.len() as f64);
        }
    }

    let sanbai_of = sanbai_ratings(&samples)?;
    if let Some(sanbai_of) = &sanbai_of {
        let matched: Vec<usize> = (0..samples.len()).filter(|&row| sanbai_of[row].is_some()).collect();
        let tier = |row: usize| sanbai_of[row].unwrap_or(0.0) - samples[row].rating as f64;
        let mean_tier = matched.iter().map(|&row| tier(row)).sum::<f64>() / matched.len().max(1) as f64;
        // Sanbai moved so its mean tier is the middle of a level, like the model's scale
        let shifted = |row: usize| sanbai_of[row].unwrap_or(0.0) - mean_tier + LEVEL_MIDDLE as f64;
        let to_sanbai: Vec<f64> = matched.iter().map(|&row| predicted[row] - shifted(row)).collect();
        let official: Vec<f64> = matched.iter().map(|&row| middle(row) - shifted(row)).collect();
        let points: Vec<(f32, f64, f64)> = matched.iter().map(|&row| (samples[row].rating, errors[row], tier(row))).collect();
        say!(
            "\nSanbai ({SANBAI_FILE}): {} charts at their official level, mean tier {mean_tier:.3}\n\
             within-level agreement (r): {:+.3}\n\
             RMS from Sanbai moved to the model's scale: {:.3} (the official middle: {:.3})",
            matched.len(),
            within_level_r(&points).unwrap_or(f64::NAN),
            rms(&to_sanbai),
            rms(&official)
        );
    }

    let dir = match named_dir {
        Some(dir) => dir,
        None => new_run_dir(Path::new(ARTIFACTS), "crossval", SystemTime::now())?,
    };
    fs::create_dir_all(&dir)?;
    let record = CrossvalRecord {
        folds,
        split_seed: training.split_seed,
        train_seed: training.train_seed,
        excluded,
        statistics: kept.len(),
        activation: model_config.activation.name(),
        analyzer: analyzer_fingerprint(),
    };
    fs::write(dir.join(CROSSVAL_FILE), serde_json::to_string_pretty(&record)?)?;
    let mut csv = String::from("chart,simfile,fold,rating,predicted,sanbai\n");
    for (row, sample) in samples.iter().enumerate() {
        let sanbai = sanbai_of.as_ref().and_then(|of| of[row]).map_or(String::new(), |rating| rating.to_string());
        let simfile = sample.simfile.display().to_string();
        let fields = [csv_field(&sample.name), csv_field(&simfile)];
        let _ = writeln!(csv, "{},{},{},{},{:.4},{sanbai}", fields[0], fields[1], fold_of_song[song_of[row]] + 1, sample.rating, predicted[row]);
    }
    fs::write(dir.join(PREDICTIONS_FILE), csv)?;
    say!("\nsaved every chart's out-of-fold prediction to {}", dir.join(PREDICTIONS_FILE).display());
    Ok(())
}

/// The model's inputs by name: `ChartStatistics::features`', in order.
fn statistic_names() -> Vec<String> {
    ChartStatistics::default().features().into_iter().map(|(name, _)| name).collect()
}

/// Whether `pattern` is `name`, with each `*` in it standing for any run of characters.
fn matches_pattern(pattern: &str, name: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(mut rest) = parts.next().and_then(|start| name.strip_prefix(start)) else { return false };
    let parts: Vec<&str> = parts.collect();
    // No `*`: the whole name
    let Some((end, between)) = parts.split_last() else { return rest.is_empty() };
    for part in between {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    rest.ends_with(end)
}

/// The columns of `names` that no pattern of `excluded` matches (`matches_pattern`), in order,
/// and the names that one does. An error if a pattern matches none, or they leave no column.
fn kept_columns(names: &[String], excluded: &[&str]) -> Result<(Vec<usize>, Vec<String>), String> {
    if let Some(pattern) = excluded.iter().find(|pattern| !names.iter().any(|name| matches_pattern(pattern, name))) {
        return Err(match pattern.is_empty() {
            true => "--exclude takes statistics' names, with commas between".to_string(),
            false => format!("no statistic is called {pattern} (`analyze` lists them)"),
        });
    }
    let left_out = |name: &String| excluded.iter().any(|pattern| matches_pattern(pattern, name));
    let (out, kept): (Vec<usize>, Vec<usize>) = (0..names.len()).partition(|&column| left_out(&names[column]));
    match kept.is_empty() {
        true => Err("--exclude leaves no statistics".to_string()),
        false => Ok((kept, out.into_iter().map(|column| names[column].clone()).collect())),
    }
}

/// Each sample's song, numbered from 0 in order of appearance, and how many songs. A song is
/// its folder, and a cover (`<song> (cover)`) counts as the song it covers: same tempo and
/// length.
fn song_ids(samples: &[Sample]) -> (Vec<usize>, usize) {
    let mut ids = HashMap::new();
    let song_of = samples
        .iter()
        .map(|sample| {
            let folder = sample.simfile.parent().and_then(Path::file_name).unwrap_or_default().to_string_lossy();
            let song = folder.strip_suffix(" (cover)").unwrap_or(&folder).to_string();
            let next = ids.len();
            *ids.entry(song).or_insert(next)
        })
        .collect();
    (song_of, ids.len())
}

/// Each song's fold: the songs shuffled by `seed`, then dealt out in turn, so folds differ by a
/// song at most.
fn song_folds(num_songs: usize, folds: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..num_songs).collect();
    shuffle(&mut order, seed);
    let mut fold_of = vec![0; num_songs];
    for (dealt, &song) in order.iter().enumerate() {
        fold_of[song] = dealt % folds;
    }
    fold_of
}

/// The rows of fold `fold`'s songs to test on; of the other songs, `valid_ratio` of all songs
/// to validate on (shuffled by `seed` and the fold), and the rest to train on.
fn fold_split(song_of: &[usize], fold_of_song: &[usize], fold: usize, valid_ratio: f64, seed: u64) -> Split {
    let num_songs = fold_of_song.len();
    let mut others: Vec<usize> = (0..num_songs).filter(|&song| fold_of_song[song] != fold).collect();
    shuffle(&mut others, scramble(seed).wrapping_add(fold as u64));
    let validating: HashSet<usize> = others.iter().take((num_songs as f64 * valid_ratio).round() as usize).copied().collect();
    let (mut train, mut validation, mut test) = (Vec::new(), Vec::new(), Vec::new());
    for (row, &song) in song_of.iter().enumerate() {
        match (fold_of_song[song] == fold, validating.contains(&song)) {
            (true, _) => test.push(row),
            (false, true) => validation.push(row),
            (false, false) => train.push(row),
        }
    }
    (train, validation, test)
}

/// Sanbai's ratings of the current game's charts (3icecream.com: the official level plus a tier
/// from 0 to 0.95, from players' scores), by song title and difficulty.
struct Sanbai {
    /// `cost_fit::title_key` of each title (Japanese and romanized) to its songs
    songs: HashMap<String, HashSet<String>>,
    /// (song, difficulty) to (level, rating)
    charts: HashMap<(String, String), (f64, f64)>,
}

impl Sanbai {
    /// Columns song_id, song_name, romanized_name, difficulty, level and sanbai (others ignored).
    fn parse(text: &str) -> Result<Sanbai, String> {
        let rows = cost_fit::parse_csv(text);
        let header = rows.first().ok_or("the Sanbai file is empty")?;
        let column = |name: &str| header.iter().position(|title| title == name).ok_or_else(|| format!("the Sanbai file has no {name} column"));
        let [id, name, romanized, difficulty, level, rating] =
            ["song_id", "song_name", "romanized_name", "difficulty", "level", "sanbai"].map(column);
        let (id, name, romanized, difficulty, level, rating) = (id?, name?, romanized?, difficulty?, level?, rating?);
        let mut sanbai = Sanbai { songs: HashMap::new(), charts: HashMap::new() };
        for row in rows.iter().skip(1).filter(|row| row.len() == header.len()) {
            for title in [&row[name], &row[romanized]].into_iter().filter(|title| !title.is_empty()) {
                sanbai.songs.entry(cost_fit::title_key(title)).or_default().insert(row[id].clone());
            }
            if let (Ok(level), Ok(rating)) = (row[level].parse(), row[rating].parse()) {
                sanbai.charts.insert((row[id].clone(), row[difficulty].clone()), (level, rating));
            }
        }
        Ok(sanbai)
    }

    /// The one Sanbai song that has one of `titles`, if just one does.
    fn song(&self, titles: &[&str]) -> Option<&String> {
        let songs: HashSet<&String> = titles.iter().filter_map(|title| self.songs.get(&cost_fit::title_key(title))).flatten().collect();
        let [song] = songs.into_iter().collect::<Vec<_>>()[..] else { return None };
        Some(song)
    }

    /// (level, rating) of `song`'s chart of `difficulty`, if today's game has it.
    fn chart(&self, song: &str, difficulty: ChartDifficulty) -> Option<(f64, f64)> {
        self.charts.get(&(song.to_string(), format!("{difficulty:?}"))).copied()
    }

    /// The rating of the chart of `difficulty` in the one Sanbai song that has one of `titles`,
    /// if its level is `level`.
    fn rating(&self, titles: &[&str], difficulty: ChartDifficulty, level: f64) -> Option<f64> {
        let (sanbai_level, rating) = self.chart(self.song(titles)?, difficulty)?;
        (sanbai_level == level).then_some(rating)
    }
}

/// Each sample's Sanbai rating, matched by its folder's name and its simfile's title, or `None`
/// for all if there's no `SANBAI_FILE`.
fn sanbai_ratings(samples: &[Sample]) -> Result<Option<Vec<Option<f64>>>, Box<dyn Error>> {
    let Ok(text) = fs::read_to_string(SANBAI_FILE) else { return Ok(None) };
    let sanbai = Sanbai::parse(&text)?;
    let titles = simfile_titles(samples)?;
    let ratings = samples.iter().map(|sample| {
        let folder = folder_name(&sample.simfile);
        sanbai.rating(&[&folder, &titles[&sample.simfile]], sample.difficulty, sample.rating as f64)
    });
    Ok(Some(ratings.collect()))
}

/// Each simfile's `#TITLE`.
fn simfile_titles(samples: &[Sample]) -> Result<HashMap<PathBuf, String>, Box<dyn Error>> {
    let mut titles = HashMap::new();
    for sample in samples {
        if !titles.contains_key(&sample.simfile) {
            let title = SimFile::from_file(&sample.simfile).map_err(|e| format!("{}: {e}", sample.simfile.display()))?.title;
            titles.insert(sample.simfile.clone(), title);
        }
    }
    Ok(titles)
}

/// The name of the folder `simfile` is in: its song's.
fn folder_name(simfile: &Path) -> String {
    simfile.parent().and_then(Path::file_name).unwrap_or_default().to_string_lossy().into_owned()
}

/// Why a song from a `Songs::Current` pack isn't trained on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LeftOut {
    /// A language version (a folder ending ` (Chinese)`)
    Variant,
    /// Not one song of today's game, by its folder's name or its title
    NotInGame,
    /// One of `EXCLUDED_SONGS`
    Excluded,
    /// A chart rated otherwise than today: another song of the same name, or an older version
    RatedOtherwise,
    /// A newer pack has it
    OlderCopy,
}

/// `samples` without what isn't trained on, from packs of `Songs::Current` (`songs`, by
/// `Sample::pack`; `titles` by simfile): songs not in today's game or rated otherwise (see
/// `LeftOut`), charts today's game doesn't have, and any copy of a song a newer pack has. Also
/// how many songs were left out for each reason, and how many more charts.
fn current_charts(
    samples: Vec<Sample>,
    songs: &[Songs],
    titles: &HashMap<PathBuf, String>,
    sanbai: &Sanbai,
) -> (Vec<Sample>, BTreeMap<LeftOut, usize>, usize) {
    let excluded: HashSet<&String> = EXCLUDED_SONGS.iter().filter_map(|title| sanbai.song(&[title])).collect();
    // Each song copy (a simfile, its charts in a row): its pack, Sanbai song, and why it's left out
    let mut copies: Vec<(&Path, usize, Option<&String>, Option<LeftOut>)> = Vec::new();
    for sample in &samples {
        if copies.last().is_some_and(|copy| copy.0 == sample.simfile) {
            continue;
        }
        let folder = folder_name(&sample.simfile);
        let song = sanbai.song(&[&folder, &titles[&sample.simfile]]);
        let charts = samples.iter().filter(|other| other.simfile == sample.simfile);
        let mut rated = charts.filter_map(|chart| Some((sanbai.chart(song?, chart.difficulty)?.0, chart.rating as f64)));
        let left_out = match (songs[sample.pack], song) {
            (Songs::All, _) => None,
            _ if folder.ends_with(" (Chinese)") => Some(LeftOut::Variant),
            (_, None) => Some(LeftOut::NotInGame),
            (_, Some(song)) if excluded.contains(song) => Some(LeftOut::Excluded),
            _ if rated.any(|(today, rating)| today != rating) => Some(LeftOut::RatedOtherwise),
            _ => None,
        };
        copies.push((&sample.simfile, sample.pack, song, left_out));
    }
    // The newest pack's copy of each song, the first if a pack has two
    let mut newest: HashMap<&String, (usize, &Path)> = HashMap::new();
    for &(simfile, pack, song, left_out) in &copies {
        if let (Some(song), None) = (song, left_out)
            && newest.get(song).is_none_or(|&(kept, _)| pack > kept)
        {
            newest.insert(song, (pack, simfile));
        }
    }
    let mut verdicts: HashMap<&Path, (Option<&String>, Option<LeftOut>)> = HashMap::new();
    let mut left_out = BTreeMap::new();
    for &(simfile, pack, song, reason) in &copies {
        let older = songs[pack] == Songs::Current && reason.is_none() && song.is_some_and(|song| newest[song].1 != simfile);
        let reason = reason.or(older.then_some(LeftOut::OlderCopy));
        if let Some(reason) = reason {
            *left_out.entry(reason).or_insert(0) += 1;
        }
        verdicts.insert(simfile, (song, reason));
    }
    // Charts of kept songs that today's game doesn't have
    let verdicts: HashMap<PathBuf, (Option<String>, Option<LeftOut>)> =
        verdicts.into_iter().map(|(simfile, (song, reason))| (simfile.to_path_buf(), (song.cloned(), reason))).collect();
    let (mut kept, mut missing_charts) = (Vec::new(), 0);
    for sample in samples {
        let (song, reason) = &verdicts[&sample.simfile];
        let in_game = || song.as_ref().and_then(|song| sanbai.chart(song, sample.difficulty)).is_some();
        match (reason, songs[sample.pack]) {
            (Some(_), _) => {}
            (None, Songs::Current) if !in_game() => missing_charts += 1,
            (None, _) => kept.push(sample),
        }
    }
    (kept, left_out, missing_charts)
}

/// Correlation between the model's and Sanbai's verdicts within a level: each (official level,
/// prediction's distance from the level's middle, Sanbai's tier), both taken relative to their
/// level's mean. `None` with no spread.
fn within_level_r(points: &[(f32, f64, f64)]) -> Option<f64> {
    let mut by_level: HashMap<u32, Vec<(f64, f64)>> = HashMap::new();
    for &(level, model, sanbai) in points {
        by_level.entry(level.to_bits()).or_default().push((model, sanbai));
    }
    let (mut xy, mut xx, mut yy) = (0.0, 0.0, 0.0);
    for pairs in by_level.values() {
        let n = pairs.len() as f64;
        let (mx, my) = (pairs.iter().map(|p| p.0).sum::<f64>() / n, pairs.iter().map(|p| p.1).sum::<f64>() / n);
        for &(x, y) in pairs {
            (xy, xx, yy) = (xy + (x - mx) * (y - my), xx + (x - mx).powi(2), yy + (y - my).powi(2));
        }
    }
    (xx > 0.0 && yy > 0.0).then(|| xy / (xx * yy).sqrt())
}

/// `text` as one CSV field: in quotes if it has a comma, quote or line break.
fn csv_field(text: &str) -> String {
    match text.contains([',', '"', '\n', '\r']) {
        true => format!("\"{}\"", text.replace('"', "\"\"")),
        false => text.to_string(),
    }
}

/// Fits a PCA of the chart statistics of a folder of song folders (the argument, or the
/// `DATASET` packs) on the split `train` trains on, prints it and saves it to `PCA_FILE`.
pub fn pca(args: &[String]) -> Result<(), Box<dyn Error>> {
    let (feature_names, samples) = cached_samples(&song_packs(args)?)?;
    report_pca(&feature_names, &samples)
}

/// Rates every dance-single chart of each simfile with `--model`'s model in `ARTIFACTS`, or
/// `DEFAULT_MODEL`. Refuses if the model was trained on other statistics than charts have now.
pub fn infer(args: &[String]) -> Result<(), Box<dyn Error>> {
    let (name, simfiles) = take_option(args, "--model")?;
    if simfiles.is_empty() {
        return Err("infer needs a simfile or song folder".into());
    }
    let dir = model_dir(name.as_deref().unwrap_or(DEFAULT_MODEL))?;
    let shown = dir.display();
    let device = DevicePreference::default().device();
    let model = TrainedModel::load(&dir, &device)
        .map_err(|e| format!("couldn't load the model in {shown} ({e}): has `train` saved one there?"))?;
    let trained_on = ModelInputs::load(&dir)
        .map_err(|e| format!("the model in {shown} doesn't say which statistics it takes ({e}): train it again"))?;
    let names = ChartStatistics::default().features().into_iter().map(|(name, _)| name).collect();
    if let Some(warning) = trained_on.check(&ModelInputs::now(names))? {
        eprintln!("warning: {warning}");
    }
    // A mismatch would broadcast silently
    if trained_on.features.len() != model.model.num_inputs() {
        return Err(format!(
            "the model in {shown} takes {} statistics, but {INPUTS_FILE} lists {}: train it again",
            model.model.num_inputs(),
            trained_on.features.len()
        )
        .into());
    }
    say!("{SCALE_NOTE}\n");
    for path in &simfiles {
        let sim = SimFile::from_file(path).map_err(|e| format!("{path}: {e}"))?;
        say!("{} by {}", sim.title, sim.artist);
        say!("{:<10} {:>6} {:>9}", "chart", "rating", "predicted");
        for chart in analyze_singles(&sim) {
            let name = format!("{:?}", chart.difficulty);
            let features = match finite_features(&chart.stats) {
                Ok(features) => features,
                Err(feature) => {
                    say!("{name:<10} {:>6} {:>9}  (its {feature} isn't a finite number)", chart.rating, "-");
                    continue;
                }
            };
            let width = features.len();
            let x = Tensor::<2>::from_data(TensorData::new(features, [1, width]), &device);
            let predicted = model.predict(x).into_data().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            say!("{name:<10} {:>6} {:>9.2}", chart.rating, predicted[0]);
        }
        say!();
    }
    Ok(())
}

/// `--option value` or `--option=value`, at most once, and the other arguments in order. Any
/// other `--` argument is an error.
fn take_option(args: &[String], option: &str) -> Result<(Option<String>, Vec<String>), String> {
    let ([value], rest) = take_options(args, [option])?;
    Ok((value, rest))
}

/// `take_option` for each of `options`.
fn take_options<const N: usize>(args: &[String], options: [&str; N]) -> Result<([Option<String>; N], Vec<String>), String> {
    let (mut values, mut rest) = (std::array::from_fn(|_| None), Vec::new());
    let mut args = args.iter();
    'args: while let Some(arg) = args.next() {
        for (value, option) in values.iter_mut().zip(options) {
            let given = match arg.strip_prefix(option) {
                Some("") => args.next().ok_or_else(|| format!("{option} needs a value"))?,
                Some(after) if after.starts_with('=') => &after[1..],
                _ => continue,
            };
            if value.replace(given.to_string()).is_some() {
                return Err(format!("{option} is given twice"));
            }
            continue 'args;
        }
        if arg.starts_with("--") {
            return Err(format!("unknown option {arg}"));
        }
        rest.push(arg.clone());
    }
    Ok((values, rest))
}

/// A model's folder in `ARTIFACTS`. The name is one folder name, so the model stays in there.
fn model_dir(name: &str) -> Result<PathBuf, String> {
    match name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\']) {
        true => Err(format!("a model name is a folder name in {ARTIFACTS}/, not {name:?}")),
        false => Ok(Path::new(ARTIFACTS).join(name)),
    }
}

/// A new folder in `root` named `kind` and `time` (`model_20261126_013015Z`, UTC), with `_2`,
/// `_3` and so on if a run saved in the same second.
fn new_run_dir(root: &Path, kind: &str, time: SystemTime) -> io::Result<PathBuf> {
    fs::create_dir_all(root)?;
    let stamp = utc_stamp(time);
    let mut copy = 1;
    loop {
        let name = if copy == 1 { format!("{kind}_{stamp}") } else { format!("{kind}_{stamp}_{copy}") };
        let dir = root.join(name);
        // Fails if it's there, so two runs can't take the same folder
        match fs::create_dir(&dir) {
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => copy += 1,
            made => return made.map(|()| dir),
        }
    }
}

/// `YYYYMMDD_HHMMSSZ`, in UTC.
fn utc_stamp(time: SystemTime) -> String {
    let seconds = time.duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs());
    let (days, second) = ((seconds / 86_400) as i64, seconds % 86_400);
    // Howard Hinnant's civil_from_days: years from March, so leap days come last
    let shifted = days + 719_468;
    let (era, day_of_era) = (shifted.div_euclid(146_097), shifted.rem_euclid(146_097));
    let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 { month_from_march + 3 } else { month_from_march - 9 };
    let year = era * 400 + year_of_era + i64::from(month <= 2);
    format!("{year:04}{month:02}{day:02}_{:02}{:02}{:02}Z", second / 3600, second / 60 % 60, second % 60)
}

/// Prints the cheapest footing (see `chart_analyzer::footing`) of arrow letters as one
/// segment (at a speed, 1 if not given), or of one chart of a simfile. A simfile alone lists its
/// charts' footings added up.
pub fn feet(args: &[String]) -> Result<(), Box<dyn Error>> {
    let (input, second) = match args {
        [input] => (input, None),
        [input, second] => (input, Some(second.as_str())),
        _ => return Err("feet needs arrows (e.g. UDDRLRURDL) and maybe a speed, or a simfile and maybe a difficulty".into()),
    };
    if Path::new(input).exists() {
        return feet_of_simfile(input, second);
    }
    let speed = match second.map(str::parse::<f64>) {
        None => 1.0,
        Some(Ok(speed)) if speed > 0.0 && speed.is_finite() => speed,
        Some(_) => return Err(format!("no simfile at {input}, and {} isn't a speed (e.g. 2)", second.unwrap_or_default()).into()),
    };
    let segment = foot_arrows(input, &COSTS, speed).map_err(|e| format!("{input} is neither a simfile nor arrows: {e}"))?;
    let summary = Summary::of(std::slice::from_ref(&segment));
    // A row is a 16th
    let bpm = 15.0 * speed / COSTS.reference_gap;
    let report = format!(
        "{input} at speed {speed} (16ths at {bpm:.0} BPM): {} notes, {}\n{}\n{}\n\n{}",
        summary.notes,
        summary.footing(),
        summary.patterns(),
        summary.search(),
        footing_table(&segment, false)
    );
    print_report(&report)
}

/// Runs the whole chart analyzer on one dance-single chart and prints the result as JSON (see
/// `analysis_json`), for other programs. Takes every input the analyzer has: the simfile (`-`
/// reads its text from stdin), the difficulty, and any of the costs and settings
/// (`--jack-base 0.5` or `--jack-base=0.5`), the rest being `COSTS` and `SETTINGS`. `--footing`
/// adds the footing (see `footing_json`). With no arguments, lists them.
pub fn analyze(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.is_empty() || matches!(args, [only] if only == "--help") {
        say!("{}", analyze_usage());
        return Ok(());
    }
    let given = analyze_args(args)?;
    let sim = if given.simfile == "-" {
        SimFile::parse(&io::read_to_string(io::stdin())?).map_err(|e| format!("stdin: {e}"))?
    } else {
        SimFile::from_file(given.simfile).map_err(|e| format!("{}: {e}", given.simfile))?
    };
    print_report(&(analysis_json(&sim, given.difficulty, &given.costs, &given.settings, given.footing)? + "\n"))
}

/// A cost's or setting's option: `--jack-base` for "jack base".
fn option(name: &str) -> String {
    format!("--{}", name.replace(' ', "-"))
}

/// `analyze`'s arguments, and every cost and setting with its default.
fn analyze_usage() -> String {
    let mut usage = format!(
        "usage: analyze <simfile> <difficulty> [--footing] [--<cost or setting> <value>]...\n\n\
         A simfile is a .sm or .ssc file, its song folder, or - for its text on stdin. --footing\n\
         adds every note's footing and every technique found, by note. A cost's base or speed can\n\
         be inf, which rules its event out unless there is no other way: it counts as {MAX_COST:e}, the\n\
         most any does.\n\ncosts (default):\n",
    );
    let mut costs = COSTS;
    for (name, value) in costs.parameters() {
        let _ = writeln!(usage, "  {:<30} {value}", option(name));
    }
    usage += "\nsettings (default):\n";
    for (name, value) in SETTINGS.values() {
        let _ = writeln!(usage, "  {:<30} {value}", option(name));
    }
    usage
}

/// What `analyze` is given.
#[derive(Debug)]
struct AnalyzeArgs<'a> {
    simfile: &'a str,
    difficulty: &'a str,
    costs: Costs,
    settings: Settings,
    /// Whether to print the footing too
    footing: bool,
}

fn analyze_args(args: &[String]) -> Result<AnalyzeArgs<'_>, String> {
    let (mut costs, mut settings, mut footing) = (COSTS, SETTINGS, false);
    let (mut given, mut positional) = (Vec::new(), Vec::new());
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if !arg.starts_with("--") {
            positional.push(arg.as_str());
            continue;
        }
        let (flag, value) = match arg.split_once('=') {
            Some((flag, value)) => (flag, Some(value)),
            None => (arg.as_str(), None),
        };
        if flag == "--footing" {
            if value.is_some() {
                return Err("--footing takes no value".into());
            }
            if footing {
                return Err("--footing is given twice".into());
            }
            footing = true;
            continue;
        }
        let cost = costs.parameters().into_iter().map(|(name, _)| name).find(|name| option(name) == flag);
        let setting = settings.values().into_iter().map(|(name, _)| name).find(|name| option(name) == flag);
        if cost.is_none() && setting.is_none() {
            return Err(format!("unknown option {flag}\n\n{}", analyze_usage()));
        }
        if given.contains(&flag) {
            return Err(format!("{flag} is given twice"));
        }
        given.push(flag);
        let value = match value {
            Some(value) => value,
            None => args.next().ok_or_else(|| format!("{flag} needs a value"))?.as_str(),
        };
        let value: f64 = value.parse().map_err(|_| format!("{flag} takes a number, not {value}"))?;
        let set = match (cost, setting) {
            (Some(cost), _) => set_cost(&mut costs, cost, value),
            (_, Some(setting)) => settings.set(setting, value),
            (None, None) => unreachable!("an unknown option was refused"),
        };
        set.map_err(|must| format!("{flag} {must}, not {value}"))?;
    }
    match positional[..] {
        [simfile, difficulty] => Ok(AnalyzeArgs { simfile, difficulty, costs, settings, footing }),
        _ => Err(format!("analyze takes a simfile and a difficulty, then any costs and settings\n\n{}", analyze_usage())),
    }
}

/// Sets the cost called `name` (as in `Costs::parameters`), or says why `value` can't be it. A
/// penalty over `MAX_COST`, infinity too, is set to it.
fn set_cost(costs: &mut Costs, name: &str, value: f64) -> Result<(), String> {
    // A time, and a speed. The rest are penalties.
    let penalty = !matches!(name, "reference gap" | "max speed");
    let must = match name {
        _ if penalty && value == f64::INFINITY => None,
        _ if !value.is_finite() => Some(if penalty { "a number or inf" } else { "a finite number" }),
        "reference gap" if value <= 0.0 => Some("more than 0"),
        "max speed" if value < 0.0 => Some("0 or more"),
        _ => None,
    };
    if let Some(must) = must {
        return Err(format!("must be {must}"));
    }
    let mut parameters = costs.parameters();
    let (_, cost) = parameters.iter_mut().find(|(cost, _)| *cost == name).ok_or_else(|| format!("there is no cost called {name}"))?;
    **cost = if penalty { value.min(MAX_COST) } else { value };
    Ok(())
}

/// What `analyze` prints.
#[derive(Serialize)]
struct Analysis<'a> {
    title: &'a str,
    artist: &'a str,
    difficulty: ChartDifficulty,
    rating: f64,
    /// `analyzer_fingerprint` in hex: the same source gives the same statistics
    analyzer: String,
    /// Every cost, by name
    costs: Named,
    settings: Settings,
    /// `ChartStatistics::features`, in order
    features: Named,
    #[serde(skip_serializing_if = "Option::is_none")]
    footing: Option<FootingJson>,
}

/// `analyze --footing`'s footing. Notes are numbered over the whole chart, from 0, in time order.
#[derive(Serialize)]
struct FootingJson {
    segments: Vec<SegmentJson>,
    notes: Vec<NoteJson>,
    /// Every technique found, in order of its first note
    techniques: Vec<FoundJson>,
}

/// A run of notes footed on its own, from the feet at home (see `chart_analyzer::segments`).
#[derive(Serialize)]
struct SegmentJson {
    first_note: usize,
    notes: usize,
    /// What its footing's events cost (see `Solution::cost`)
    cost: f64,
    /// False if the search dropped branches: the footing may not be the cheapest
    exact: bool,
}

#[derive(Serialize)]
struct NoteJson {
    segment: usize,
    /// Seconds from beat 0
    time: f64,
    /// Beats from the first measure's start
    beat: f64,
    /// The arrows stepped on, left to right
    arrows: Vec<ArrowJson>,
    /// Holds a foot takes over from the other foot here, in a freeze switch
    regrabs: Vec<RegrabJson>,
    /// What the events of its step cost (see `Solution::note_costs`)
    cost: f64,
}

#[derive(Serialize)]
struct ArrowJson {
    /// left, down, up or right
    arrow: &'static str,
    /// tap, hold or roll
    note: &'static str,
    /// left or right, or null if neither foot can reach it (both are holding elsewhere)
    foot: Option<&'static str>,
    /// When a hold or roll ends, in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    end: Option<f64>,
}

#[derive(Serialize)]
struct RegrabJson {
    arrow: &'static str,
    foot: &'static str,
}

#[derive(Serialize)]
struct FoundJson {
    /// As in the statistics' names: jack, staircase, ...
    technique: String,
    /// The technique it is one kind of (full_crossover is a horizontal_crossover), if any
    #[serde(skip_serializing_if = "Option::is_none")]
    kind_of: Option<String>,
    /// notes (found in the notes alone), footing_event or footing_pattern
    source: &'static str,
    first_note: usize,
    /// How many it spans: 1 for an event
    notes: usize,
    /// Notes per second
    speed: f64,
}

const ARROW_NAMES: [&str; 4] = ["left", "down", "up", "right"];

fn foot_name(foot: Foot) -> &'static str {
    match foot {
        Foot::Left => "left",
        Foot::Right => "right",
    }
}

/// The footing of `segments` (a whole chart's), notes and techniques numbered over all of them.
fn footing_json(segments: &[AnalyzedSegment]) -> FootingJson {
    let mut json = FootingJson { segments: Vec::new(), notes: Vec::new(), techniques: Vec::new() };
    let mut first = 0;
    for (index, segment) in segments.iter().enumerate() {
        let footing = &segment.footing;
        json.segments.push(SegmentJson { first_note: first, notes: segment.notes.len(), cost: footing.cost, exact: footing.exact });
        for ((note, step), &cost) in segment.notes.iter().zip(&footing.steps).zip(&footing.note_costs) {
            let arrows = (0..4)
                .filter(|&arrow| note.steps(arrow))
                .map(|arrow| ArrowJson {
                    arrow: ARROW_NAMES[arrow],
                    note: match note.row[arrow] {
                        Note::HoldStart => "hold",
                        Note::RollStart => "roll",
                        _ => "tap",
                    },
                    foot: step.foot(arrow).map(foot_name),
                    end: note.hold_ends[arrow],
                })
                .collect();
            let regrabs = (0..4)
                .filter_map(|arrow| step.regrabber(arrow).map(|foot| RegrabJson { arrow: ARROW_NAMES[arrow], foot: foot_name(foot) }))
                .collect();
            json.notes.push(NoteJson { segment: index, time: note.time, beat: note.beat, arrows, regrabs, cost });
        }
        let mut found: Vec<_> = segment.from_notes.iter().chain(&footing.occurrences).collect();
        found.sort_by_key(|occurrence| occurrence.note);
        json.techniques.extend(found.into_iter().map(|occurrence| {
            let technique = occurrence.technique;
            FoundJson {
                technique: technique.key(),
                kind_of: Technique::ALL.into_iter().find(|parent| parent.kinds().contains(&technique)).map(Technique::key),
                source: match technique.source() {
                    Source::Notes => "notes",
                    Source::FootingEvent => "footing_event",
                    Source::FootingPattern => "footing_pattern",
                },
                first_note: first + occurrence.note,
                notes: occurrence.notes,
                speed: occurrence.speed,
            }
        }));
        first += segment.notes.len();
    }
    json
}

/// (name, value) pairs: a JSON object, in their order. A value that isn't finite is null.
struct Named(Vec<(String, f64)>);

impl Serialize for Named {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (name, value) in &self.0 {
            map.serialize_entry(name, value)?;
        }
        map.end()
    }
}

/// `sim`'s dance-single chart of `difficulty` (any case), analyzed with `costs` and `settings`, as
/// JSON. With `footing`, its footing too.
fn analysis_json(sim: &SimFile, difficulty: &str, costs: &Costs, settings: &Settings, footing: bool) -> Result<String, String> {
    let singles: Vec<&SingleChart> = sim
        .step_charts
        .iter()
        .filter_map(|chart| match chart {
            StepChart::DanceSingle(single) => Some(single),
            StepChart::DanceDouble(_) => None,
        })
        .collect();
    let name = |chart: &SingleChart| format!("{:?}", chart.difficulty);
    let matching: Vec<&SingleChart> = singles.iter().copied().filter(|chart| name(chart).eq_ignore_ascii_case(difficulty)).collect();
    let chart = match matching[..] {
        [chart] => chart,
        [] if singles.is_empty() => return Err(format!("{} has no dance-single charts", sim.title)),
        [] => {
            let names: Vec<String> = singles.iter().map(|chart| name(chart)).collect();
            return Err(format!("{} has no {difficulty} chart (it has {})", sim.title, names.join(", ")));
        }
        _ => return Err(format!("{} has {} {difficulty} charts", sim.title, matching.len())),
    };
    let analysis = analyze_chart(chart, &chart.bpms, &chart.stops, costs, settings);
    let mut costs = *costs;
    let analysis = Analysis {
        title: &sim.title,
        artist: &sim.artist,
        difficulty: chart.difficulty,
        rating: chart.rating,
        analyzer: format!("{:016x}", analyzer_fingerprint()),
        costs: Named(costs.parameters().iter().map(|(name, value)| (name.replace(' ', "_"), **value)).collect()),
        settings: *settings,
        features: Named(analysis.stats.features()),
        footing: footing.then(|| footing_json(&analysis.segments)),
    };
    serde_json::to_string_pretty(&analysis).map_err(|e| e.to_string())
}

/// The spreadsheet folder DDR A3 is in.
const SHEET_FOLDER: &str = "DDR A3 (AC) (INTL)";

/// Scores `COSTS` against a chart spreadsheet's counts for the charts of a folder of song folders
/// (see `cost_fit::score`), then again with each cost changed on its own.
pub fn scan_costs(args: &[String]) -> Result<(), Box<dyn Error>> {
    let charts = charts_in_sheet(args, "scan-costs")?;
    let threads = footing_threads();
    let evaluate = |costs: &Costs| cost_fit::score(&charts, &cost_fit::foot_all(&charts, costs, threads).counts);
    let baseline = print_scores("the costs as they are", &charts, &COSTS, threads);
    say!("{SCORE_KEY}\n");

    let scored = COLUMNS[..SCORED].iter().map(|name| format!("{name:>13}")).collect::<String>();
    say!("{:<26} {:>8} {:>7} {:>8} {scored}", "one cost changed", "to", "score", "change");
    // Per cost: (name, lowest score and its value, highest score)
    let mut ranges = Vec::new();
    for index in 0..COSTS.clone().parameters().len() {
        let mut costs = COSTS;
        let (name, value) = (costs.parameters()[index].0, *costs.parameters()[index].1);
        let (mut lowest, mut highest) = ((baseline.score, value), baseline.score);
        for changed in scan_values(name, value) {
            *costs.parameters()[index].1 = changed;
            let score = evaluate(&costs);
            let errors = score.errors[..SCORED].iter().map(|error| format!("{error:>13.3}")).collect::<String>();
            say!("{name:<26} {changed:>8.3} {:>7.4} {:>+8.4} {errors}", score.score, score.score - baseline.score);
            if score.score < lowest.0 {
                lowest = (score.score, changed);
            }
            highest = highest.max(score.score);
        }
        ranges.push((name, value, lowest, highest));
    }

    ranges.sort_by(|a, b| a.2.0.total_cmp(&b.2.0));
    say!("\n{:<26} {:>8} {:>8} {:>7} {:>8} {:>8}", "cost, best first", "now", "best at", "score", "change", "worst");
    for (name, value, (lowest, at), highest) in ranges {
        say!("{name:<26} {value:>8.3} {at:>8.3} {lowest:>7.4} {:>+8.4} {:>+8.4}", lowest - baseline.score, highest - baseline.score);
    }
    Ok(())
}

/// Scores `COSTS` against a chart spreadsheet's counts (see `cost_fit::score`): on every chart,
/// then on the charts `fit_costs` searches on and on those it holds out.
pub fn score_costs(args: &[String]) -> Result<(), Box<dyn Error>> {
    let charts = charts_in_sheet(args, "score-costs")?;
    let threads = footing_threads();
    print_scores("the costs as they are", &charts, &COSTS, threads);
    say!("{SCORE_KEY}");
    let (held_out, searched): (Vec<cost_fit::Chart>, Vec<cost_fit::Chart>) = charts.into_iter().partition(is_held_out);
    let score = |charts: &[cost_fit::Chart]| cost_fit::score(charts, &cost_fit::foot_all(charts, &COSTS, threads).counts).score;
    say!("\nscore on the {} charts fit-costs searches on: {:.4}", searched.len(), score(&searched));
    say!("score on the {} charts it holds out: {:.4}", held_out.len(), score(&held_out));
    Ok(())
}

/// Whether `fit_costs` keeps the chart out of its search.
fn is_held_out(chart: &cost_fit::Chart) -> bool {
    chart.song % HOLD_OUT_EVERY == HOLD_OUT_EVERY - 1
}

/// The costs `fit_costs` searches over: those of the events the sheet's scored columns count, and
/// the speed they are measured against. The sheet says too little about the rest.
const FITTED: [&str; 11] = [
    "jack base",
    "jack speed",
    "vertical footswitch base",
    "vertical footswitch speed",
    "sideswitch base",
    "sideswitch speed",
    "horizontal crossover base",
    "horizontal crossover speed",
    "doublestep base",
    "doublestep speed",
    "reference gap",
];
const FIT_EVALUATIONS: usize = 2000;
const FIT_SEED: u64 = 1;
/// First steps of the search, as a share of each cost (of 1, for a cost under 1).
const FIT_STEP: f64 = 0.3;
/// One song in this many is kept out of the search, to score the fitted costs on.
const HOLD_OUT_EVERY: usize = 4;
const SHORTEST_REFERENCE_GAP: f64 = 0.02;

/// Searches for the `FITTED` costs that score best against a chart spreadsheet (see
/// `cost_fit::score`) on most of the songs, and scores them on the songs held out.
pub fn fit_costs(args: &[String]) -> Result<(), Box<dyn Error>> {
    let charts = charts_in_sheet(args, "fit-costs")?;
    let threads = footing_threads();
    let (held_out, training): (Vec<cost_fit::Chart>, Vec<cost_fit::Chart>) =
        charts.into_iter().partition(is_held_out);
    say!("searching on {} charts, holding out {} (every {HOLD_OUT_EVERY}th song)", training.len(), held_out.len());

    // The search moves each cost in units of its own size
    let mut now = COSTS;
    let fitted: Vec<(usize, f64, f64)> = FITTED
        .iter()
        .map(|name| {
            let index = now.parameters().iter().position(|(parameter, _)| parameter == name).expect("a cost's name");
            let value = *now.parameters()[index].1;
            if *name == "reference gap" { (index, value, SHORTEST_REFERENCE_GAP) } else { (index, value.max(1.0), 0.0) }
        })
        .collect();
    let start: Vec<f64> = fitted.iter().map(|&(index, unit, _)| *now.parameters()[index].1 / unit).collect();
    // The costs at a point of the search, and how far the point is outside what a cost can be
    let costs_at = |point: &[f64]| {
        let (mut costs, mut outside) = (COSTS, 0.0);
        for (&(index, unit, lowest), &at) in fitted.iter().zip(point) {
            let value = (at * unit).max(lowest);
            outside += (value / unit - at).powi(2);
            *costs.parameters()[index].1 = value;
        }
        (costs, outside)
    };

    let started = Instant::now();
    say!("{:>11} {:>7} {:>7} {:>6}", "evaluations", "best", "latest", "step");
    let found = cma_es::minimize(
        &start,
        FIT_STEP,
        FIT_EVALUATIONS,
        FIT_SEED,
        |point| {
            let (costs, outside) = costs_at(point);
            cost_fit::score(&training, &cost_fit::foot_all(&training, &costs, threads).counts).score + 0.01 * outside
        },
        |generation| {
            say!("{:>11} {:>7.4} {:>7.4} {:>6.3}", generation.evaluations, generation.best, generation.generation_best, generation.step)
        },
    );
    let (best, _) = costs_at(&found.point);
    say!("\n{} evaluations in {:.0} minutes\n", found.evaluations, started.elapsed().as_secs_f64() / 60.0);

    for (charts, which) in [(&training, "searched on"), (&held_out, "held out")] {
        let before = print_scores(&format!("the costs as they are, on the charts {which}"), charts, &COSTS, threads);
        let after = print_scores(&format!("the fitted costs, on the charts {which}"), charts, &best, threads);
        say!("score {:.4} to {:.4}\n", before.score, after.score);
    }
    say!("{SCORE_KEY}\n");

    say!("{:<26} {:>8} {:>8}", "cost", "now", "fitted");
    let mut best = best;
    for &(index, ..) in &fitted {
        say!("{:<26} {:>8.3} {:>8.3}", now.parameters()[index].0, *now.parameters()[index].1, *best.parameters()[index].1);
    }
    Ok(())
}

/// The charts of a tool's songs (its second argument, or `SHEET_PACK`) that its sheet (the
/// first) has counts for.
fn charts_in_sheet(args: &[String], tool: &str) -> Result<Vec<cost_fit::Chart>, Box<dyn Error>> {
    let (sheet, songs) = match args {
        [sheet] => (sheet, pack_folder(SHEET_PACK)?),
        [sheet, songs] => (sheet, PathBuf::from(songs)),
        _ => return Err(format!("{tool} needs a chart spreadsheet (.csv), and maybe a folder of song folders").into()),
    };
    let text = fs::read_to_string(sheet).map_err(|e| format!("{sheet}: {e}"))?;
    sheet_charts(&songs, &Sheet::parse(&text, SHEET_FOLDER)?)
}

/// Half the machine's threads: its cores, with two threads a core.
fn footing_threads() -> usize {
    thread::available_parallelism().map_or(1, |threads| threads.get() / 2).max(1)
}

const SCORE_KEY: &str = "all: every event. ours: those the sheet's method would count (see `cost_fit::counts`).
error: the charts' differences from the sheet added up, over our total plus the sheet's (0 to 1).
score: the mean error of the columns scored.";

/// Foots `charts` with `costs` and prints each column against the sheet's.
fn print_scores(title: &str, charts: &[cost_fit::Chart], costs: &Costs, threads: usize) -> cost_fit::Score {
    let start = Instant::now();
    let footed = cost_fit::foot_all(charts, costs, threads);
    let score = cost_fit::score(charts, &footed.counts);
    say!("{title} ({} charts, footed in {:.1} s on {threads} threads):", charts.len(), start.elapsed().as_secs_f64());
    say!("{:<13} {:>7} {:>7} {:>7} {:>7} {:>12}", "", "all", "ours", "sheet", "error", "correlation");
    for (column, name) in COLUMNS.iter().enumerate() {
        let sheet: Vec<f64> = charts.iter().map(|chart| chart.sheet[column]).collect();
        let ours: Vec<f64> = footed.counts.iter().map(|counts| counts[column]).collect();
        say!(
            "{name:<13} {:>7} {:>7} {:>7} {:>7.3} {:>12.3}{}",
            footed.all[column],
            score.totals[column],
            sheet.iter().sum::<f64>(),
            score.errors[column],
            correlation(&ours, &sheet),
            if column < SCORED { "" } else { "  (not scored)" }
        );
    }
    let dropped = if footed.inexact > 0 { format!(" ({} segments had branches dropped)", footed.inexact) } else { String::new() };
    say!("score {:.4}{dropped}", score.score);
    score
}

/// The values to try for a cost: fractions and multiples of what it is, or small steps up from 0.
fn scan_values(name: &str, value: f64) -> Vec<f64> {
    match name {
        // Neither can be 0
        "reference gap" => [0.5, 0.75, 1.5, 2.0].map(|times| value * times).to_vec(),
        "max speed" => [0.25, 0.5, 2.0, 4.0].map(|times| value * times).to_vec(),
        _ if value == 0.0 => vec![0.5, 1.0, 2.0, 4.0],
        _ => [0.0, 0.25, 0.5, 2.0, 4.0].map(|times| value * times).to_vec(),
    }
}

/// The dance-single charts under `songs` that `sheet` has counts for.
fn sheet_charts(songs: &Path, sheet: &Sheet) -> Result<Vec<cost_fit::Chart>, Box<dyn Error>> {
    let (mut charts, mut missing) = (Vec::new(), 0);
    for song in song_folders(songs)? {
        let song_index = song.index;
        for file in song.simfiles {
            let Ok(sim) = SimFile::from_file(&file) else {
                eprintln!("skipping {}: it doesn't load", file.display());
                continue;
            };
            for chart in &sim.step_charts {
                let StepChart::DanceSingle(chart) = chart else { continue };
                match sheet.counts(&sim.title, chart.difficulty) {
                    Some(counts) => charts.push(cost_fit::Chart {
                        song: song_index,
                        segments: footing_segments(chart, &chart.bpms, &chart.stops),
                        sheet: counts,
                    }),
                    None => missing += 1,
                }
            }
        }
    }
    if charts.is_empty() {
        return Err(format!("none of the charts in {} are in the sheet's {SHEET_FOLDER} folder", songs.display()).into());
    }
    say!("{} charts from {} are in the sheet ({missing} aren't)", charts.len(), songs.display());
    Ok(charts)
}

/// A song folder with at least one simfile.
struct SongFolder {
    /// Its place among all the song folders, in name order
    index: usize,
    name: String,
    /// In name order
    simfiles: Vec<PathBuf>,
}

/// The song folders in `songs`, in name order, with their simfiles (`simfiles_in`). One with
/// none is left out with a warning, and so is their number.
fn song_folders(songs: &Path) -> io::Result<Vec<SongFolder>> {
    let (mut folders, mut skipped) = (Vec::new(), 0);
    for (index, song) in sorted_paths(songs)?.into_iter().filter(|path| path.is_dir()).enumerate() {
        let name = song.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let simfiles = simfiles_in(&song)?;
        if simfiles.is_empty() {
            eprintln!("skipping {name}: no .ssc or .sm file");
            skipped += 1;
            continue;
        }
        folders.push(SongFolder { index, name, simfiles });
    }
    if skipped > 0 {
        eprintln!("{skipped} of the {} song folders in {} have no simfile", folders.len() + skipped, songs.display());
    }
    Ok(folders)
}

/// A folder's entries in name order.
fn sorted_paths(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)?.map(|entry| entry.map(|entry| entry.path())).collect::<Result<_, _>>()?;
    paths.sort();
    Ok(paths)
}

/// Prints a long report, which is often piped to `head` or a pager that quits early.
fn print_report(report: &str) -> Result<(), Box<dyn Error>> {
    match io::stdout().write_all(report.as_bytes()) {
        Err(e) if e.kind() != io::ErrorKind::BrokenPipe => Err(e.into()),
        _ => Ok(()),
    }
}

fn feet_of_simfile(path: &str, difficulty: Option<&str>) -> Result<(), Box<dyn Error>> {
    let sim = SimFile::from_file(path).map_err(|e| format!("{path}: {e}"))?;
    let singles: Vec<&SingleChart> = sim
        .step_charts
        .iter()
        .filter_map(|chart| match chart {
            StepChart::DanceSingle(single) => Some(single),
            StepChart::DanceDouble(_) => None,
        })
        .collect();
    if singles.is_empty() {
        return Err(format!("{} has no dance-single charts", sim.title).into());
    }
    let name = |chart: &SingleChart| format!("{:?}", chart.difficulty);

    let Some(difficulty) = difficulty else {
        say!("{} by {}", sim.title, sim.artist);
        say!("{:<10} {:>6} {:>6} {:>9}  footing; patterns", "chart", "rating", "notes", "segments");
        for chart in singles {
            let footed = foot_chart(chart, &chart.bpms, &chart.stops, &COSTS);
            let summary = Summary::of(&footed);
            say!(
                "{:<10} {:>6} {:>6} {:>9}  {}; {}",
                name(chart),
                chart.rating,
                summary.notes,
                footed.len(),
                summary.footing(),
                summary.counts(Source::FootingPattern).unwrap_or("none".into())
            );
        }
        say!("\nadd a difficulty (e.g. hard) for one chart's footing");
        return Ok(());
    };

    let chart = singles.iter().find(|chart| name(chart).eq_ignore_ascii_case(difficulty)).ok_or_else(|| {
        let names: Vec<String> = singles.iter().map(|chart| name(chart)).collect();
        format!("{} has no {difficulty} chart (it has {})", sim.title, names.join(", "))
    })?;
    let footed = foot_chart(chart, &chart.bpms, &chart.stops, &COSTS);
    let summary = Summary::of(&footed);
    let mut report = format!(
        "{} [{}], rated {}: {} notes in {} segments\n{}\n{}\n{}\n",
        sim.title,
        name(chart),
        chart.rating,
        summary.notes,
        footed.len(),
        summary.footing(),
        summary.patterns(),
        summary.search()
    );
    for (i, segment) in footed.iter().enumerate() {
        let summary = Summary::of(std::slice::from_ref(segment));
        let _ = writeln!(
            report,
            "\nsegment {} of {}: {} notes, {}; {}",
            i + 1,
            footed.len(),
            summary.notes,
            summary.footing(),
            summary.patterns()
        );
        report += &footing_table(segment, true);
    }
    print_report(&report)
}

/// Segments' footings added up.
struct Summary {
    notes: usize,
    /// Per `Technique::ALL`: how many, and the notes they span
    found: [(usize, usize); Technique::ALL.len()],
    cost: f64,
    nodes: u64,
    merged: u64,
    widest: usize,
    exact: bool,
}

impl Summary {
    fn of(segments: &[FootedSegment]) -> Summary {
        let mut summary = Summary {
            notes: 0,
            found: [(0, 0); Technique::ALL.len()],
            cost: 0.0,
            nodes: 0,
            merged: 0,
            widest: 0,
            exact: true,
        };
        for segment in segments {
            summary.notes += segment.notes.len();
            for occurrence in &segment.footing.occurrences {
                let (count, notes) = &mut summary.found[occurrence.technique.index()];
                *count += 1;
                *notes += occurrence.notes;
            }
            summary.cost += segment.footing.cost;
            summary.nodes += segment.footing.nodes;
            summary.merged += segment.footing.merged;
            summary.widest = summary.widest.max(segment.footing.widest);
            summary.exact &= segment.footing.exact;
        }
        summary
    }

    /// The techniques from `source` that were found: "2 staircases, 1 drill (7 notes)". None if
    /// none were. One with kinds is left to them.
    fn counts(&self, source: Source) -> Option<String> {
        let counted: Vec<String> = Technique::ALL
            .into_iter()
            .filter(|technique| technique.source() == source && technique.kinds().is_empty())
            .filter_map(|technique| {
                let (count, notes) = self.found[technique.index()];
                let spanned = if technique.varies_in_length() { format!(" ({notes} notes)") } else { String::new() };
                (count > 0).then(|| technique.counted(count) + &spanned)
            })
            .collect();
        (!counted.is_empty()).then(|| counted.join(", "))
    }

    /// "1 jack, 2 full crossovers (cost 11.0)", or "nothing awkward". The cost depends on
    /// speed.
    fn footing(&self) -> String {
        match self.counts(Source::FootingEvent) {
            Some(events) => format!("{events} (cost {:.1})", self.cost),
            None => "nothing awkward".to_string(),
        }
    }

    fn patterns(&self) -> String {
        format!("patterns: {}", self.counts(Source::FootingPattern).unwrap_or("none".into()))
    }

    fn search(&self) -> String {
        let dropped = if self.exact { "" } else { ", some dropped: this may not be the cheapest footing" };
        format!(
            "search: {} branches tried, {} merged, at most {} kept after a note{dropped}",
            self.nodes, self.merged, self.widest
        )
    }
}

/// What a footing has on note `note`, each with its notes per second: "doublestep 8.0/s",
/// "2 jacks 4.0/s", "drill of 7 8.0/s". No speed with no note before. A technique with kinds is
/// left to them.
fn labels(segment: &FootedSegment, note: usize) -> Vec<String> {
    let found: Vec<_> = segment.footing.at(note).filter(|found| found.technique.kinds().is_empty()).collect();
    let mut labels = Vec::new();
    for (i, occurrence) in found.iter().enumerate() {
        let technique = occurrence.technique;
        if found[..i].iter().any(|earlier| earlier.technique == technique) {
            continue;
        }
        let count = found.iter().filter(|other| other.technique == technique).count();
        let mut label = match (technique.varies_in_length(), count) {
            (true, _) => format!("{} of {}", technique.name(), occurrence.notes),
            (false, 1) => technique.name().to_string(),
            (false, _) => technique.counted(count),
        };
        if occurrence.speed > 0.0 {
            let _ = write!(label, " {:.1}/s", occurrence.speed);
        }
        labels.push(label);
    }
    labels
}

/// A footing as a table: a row per note, with `L` or `R` under each arrow for the foot that steps
/// it (the same letter twice for a bracket, `x` for an arrow no foot can reach), `|` under an arrow
/// still held from earlier in the segment (`l` or `r` where that foot takes the hold over in a
/// freeze switch), then what the step is and the patterns that start there (see `labels`). Rows are labelled by
/// measure and beat if `by_measure`.
/// Else they are numbered, a row is a 16th, and the empty 16ths between notes get a row too.
fn footing_table(segment: &FootedSegment, by_measure: bool) -> String {
    let mut table = String::new();
    let _ = writeln!(table, "{}   ←  ↓  ↑  →", if by_measure { "measure  beat" } else { "  #" });
    let mut last_measure = None;
    let mut held_until = [f64::NEG_INFINITY; 4];
    for (i, note) in segment.notes.iter().enumerate() {
        if !by_measure && i > 0 {
            let before = &segment.notes[i - 1];
            let pauses = ((note.beat - before.beat) / 0.25).round() as usize - 1;
            for pause in 1..=pauses {
                let time = before.time + (note.time - before.time) * pause as f64 / (pauses + 1) as f64;
                let marks = held_until.map(|until| if until > time + 1e-9 { "|" } else { "." });
                let _ = writeln!(table, "      {}", marks.join("  "));
            }
        }
        let label = if by_measure {
            let measure = (note.beat / 4.0).floor() as usize + 1;
            let shown = if last_measure == Some(measure) { String::new() } else { measure.to_string() };
            last_measure = Some(measure);
            format!("{shown:>7}  {:>4.2}", note.beat % 4.0 + 1.0)
        } else {
            format!("{:>3}", i + 1)
        };
        let step = &segment.footing.steps[i];
        let marks: Vec<&str> = (0..4)
            .map(|arrow| match (note.steps(arrow), step.foot(arrow)) {
                // The hold's own end row isn't a note, so any note before the end is under it
                (false, _) if held_until[arrow] > note.time => match step.regrabber(arrow) {
                    Some(Foot::Left) => "l",
                    Some(Foot::Right) => "r",
                    None => "|",
                },
                (false, _) => ".",
                (true, Some(Foot::Left)) => "L",
                (true, Some(Foot::Right)) => "R",
                (true, None) => "x",
            })
            .collect();
        let events = labels(segment, i);
        for (arrow, end) in note.hold_ends.iter().enumerate() {
            if let Some(end) = end {
                held_until[arrow] = *end;
            }
        }
        let row = format!("{label}   {}   {}", marks.join("  "), events.join(", "));
        let _ = writeln!(table, "{}", row.trim_end());
    }
    table
}

/// The folders of song folders a tool reads: the one it was given, or the `DATASET` packs.
fn song_packs(args: &[String]) -> Result<Vec<Pack>, Box<dyn Error>> {
    match args {
        [] => DATASET.iter().map(|&(pack, songs)| Ok(Pack { folder: pack_folder(pack)?, songs })).collect(),
        [songs] => Ok(vec![Pack { folder: song_root(PathBuf::from(songs))?, songs: Songs::All }]),
        _ => Err("expected one folder of song folders at most".into()),
    }
}

/// A pack's folder of song folders in `PACKS_DIR` (`song_root`), if it's there.
fn pack_folder(pack: &str) -> Result<PathBuf, Box<dyn Error>> {
    let folder = Path::new(PACKS_DIR).join(pack);
    match folder.is_dir() {
        true => Ok(song_root(folder)?),
        false => Err(format!("{} not found (run from the repo, with the dataset's packs in {PACKS_DIR})", folder.display()).into()),
    }
}

/// `folder`, or the one folder in it if that's all it holds and it isn't a song's: steps-only
/// packs keep their song folders one folder deeper.
fn song_root(folder: PathBuf) -> io::Result<PathBuf> {
    let folders: Vec<PathBuf> = sorted_paths(&folder)?.into_iter().filter(|path| path.is_dir()).collect();
    match &folders[..] {
        [only] if simfiles_in(only)?.is_empty() => Ok(only.clone()),
        _ => Ok(folder),
    }
}

/// `load_samples` through the statistics cache in `CACHE_FILE`, which it updates, then
/// `current_charts` if a pack takes only those.
fn cached_samples(packs: &[Pack]) -> Result<(Vec<String>, Vec<Sample>), Box<dyn Error>> {
    let mut cache = StatsCache::load(CACHE_FILE)?;
    let folders: Vec<PathBuf> = packs.iter().map(|pack| pack.folder.clone()).collect();
    let (feature_names, mut samples) = load_samples(&folders, &mut cache)?;
    let songs = match &folders[..] {
        [folder] => folder.display().to_string(),
        _ => format!("{} packs in {PACKS_DIR}", folders.len()),
    };
    if packs.iter().any(|pack| pack.songs == Songs::Current) {
        let text = fs::read_to_string(SANBAI_FILE).map_err(|e| format!("{SANBAI_FILE}, which picks the older packs' songs: {e}"))?;
        let sanbai = Sanbai::parse(&text)?;
        let titles = simfile_titles(&samples)?;
        let kinds: Vec<Songs> = packs.iter().map(|pack| pack.songs).collect();
        let before = samples.len();
        let (kept, left_out, missing_charts) = current_charts(samples, &kinds, &titles, &sanbai);
        samples = kept;
        let reasons: Vec<String> = left_out.iter().map(|(reason, songs)| format!("{songs} {reason:?}")).collect();
        say!(
            "older packs: {} of {before} charts left out, of songs not trained on ({}), and {missing_charts} charts today's game doesn't have",
            before - samples.len(),
            reasons.join(", ")
        );
    }
    if samples.is_empty() {
        return Err(format!("no dance-single charts found in {songs}").into());
    }
    if let Some(dir) = Path::new(CACHE_FILE).parent() {
        fs::create_dir_all(dir)?;
    }
    cache.save(CACHE_FILE)?;
    say!(
        "{} charts from {songs} ({} files from the cache, {} analyzed), {} features each",
        samples.len(),
        cache.hits(),
        cache.misses(),
        samples[0].features.len()
    );
    Ok((feature_names, samples))
}

/// A chart's statistics as model inputs, or the name of one that isn't a finite number.
fn finite_features(stats: &ChartStatistics) -> Result<Vec<f32>, String> {
    let features = stats.features();
    match features.iter().find(|(_, value)| !value.is_finite()) {
        Some((feature, _)) => Err(feature.clone()),
        None => Ok(features.iter().map(|&(_, value)| value as f32).collect()),
    }
}

/// Every dance-single chart in the simfiles of the song folders in `packs`, pack by pack.
/// Folders with no simfile, files that don't load, and charts with a non-finite statistic, are
/// skipped with a warning.
///
/// Read in sorted order: listing order differs between machines, and would change the seeded
/// split.
fn load_samples(packs: &[PathBuf], cache: &mut StatsCache) -> Result<(Vec<String>, Vec<Sample>), Box<dyn Error>> {
    let mut feature_names = Vec::new();
    let mut samples = Vec::new();
    let songs = packs.iter().map(|pack| song_folders(pack)).collect::<io::Result<Vec<_>>>()?;
    let songs = songs.into_iter().enumerate().flat_map(|(pack, songs)| songs.into_iter().map(move |song| (pack, song)));
    for (pack, song) in songs {
        let song_name = song.name;
        for file in song.simfiles {
            let charts = match cache.analyze_file(&file) {
                Ok(charts) => charts,
                Err(e) => {
                    eprintln!("skipping {}: {e}", file.display());
                    continue;
                }
            };
            for chart in charts {
                let name = format!("{song_name} [{:?}]", chart.difficulty);
                feature_names = chart.stats.features().into_iter().map(|(feature, _)| feature).collect();
                match finite_features(&chart.stats) {
                    Ok(features) => {
                        let (simfile, difficulty) = (file.clone(), chart.difficulty);
                        samples.push(Sample { name, simfile, pack, difficulty, features, rating: chart.rating as f32 })
                    }
                    Err(feature) => eprintln!("skipping {name}: its {feature} isn't a finite number"),
                }
            }
        }
    }
    Ok((feature_names, samples))
}

/// Fits a PCA on the training split's features (the split `train` trains on), prints it and
/// saves it to `PCA_FILE`.
fn report_pca(feature_names: &[String], samples: &[Sample]) -> Result<(), Box<dyn Error>> {
    let feature_names: Vec<&str> = feature_names.iter().map(String::as_str).collect();
    let feature_names = &feature_names[..];
    let split = TrainingConfig::new().with_split_seed(SPLIT_SEED);
    enough_charts(samples.len(), &split, 2)?;
    let (train_rows, val_rows, test_rows) = train::split_rows(samples.len(), &split);
    let features = |rows: &[usize]| -> Vec<Vec<f32>> { rows.iter().map(|&row| samples[row].features.clone()).collect() };
    let train = features(&train_rows);
    let held_out = features(&[val_rows, test_rows].concat());

    let pca = Pca::fit(feature_names, &train);
    pca.save(PCA_FILE)?;
    let num_components = pca.components.len();
    say!(
        "\nPCA of {num_components} standardized features, fitted on the {} training charts (split seed {SPLIT_SEED}), saved to {PCA_FILE}",
        train.len()
    );
    say!("held-out: the {} validation and test charts, standardized with the training statistics\n", held_out.len());

    let train_scores = pca.transform(&train, num_components);
    let ratings: Vec<f64> = train_rows.iter().map(|&row| samples[row].rating as f64).collect();
    // Sum of squares along each component, over the held-out charts
    let held_out_scores = pca.transform(&held_out, num_components);
    let held_out_squares: Vec<f64> =
        (0..num_components).map(|k| held_out_scores.iter().map(|row| (row[k] as f64).powi(2)).sum()).collect();
    let held_out_total: f64 = held_out_squares.iter().sum();

    say!("{:>2} {:>8} {:>7} {:>7} {:>9} {:>7}  largest loadings", "pc", "variance", "share", "total", "held-out", "rating");
    let (mut total, mut held_out_so_far) = (0.0, 0.0);
    for (k, share) in pca.explained_variance_ratio().into_iter().enumerate() {
        total += share;
        held_out_so_far += held_out_squares[k] / held_out_total;
        let scores: Vec<f64> = train_scores.iter().map(|row| row[k] as f64).collect();
        let mut loadings: Vec<(f64, &str)> = pca.components[k].iter().copied().zip(feature_names.iter().copied()).collect();
        loadings.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
        let largest: Vec<String> = loadings.iter().take(4).map(|(loading, name)| format!("{loading:+.2} {name}")).collect();
        say!(
            "{:>2} {:>8.3} {:>6.1}% {:>6.1}% {:>8.1}% {:>+7.2}  {}",
            k + 1,
            pca.variances[k],
            share * 100.0,
            total * 100.0,
            held_out_so_far * 100.0,
            correlation(&scores, &ratings),
            largest.join(", ")
        );
    }
    say!("\nshare, total: of the training charts' variance. held-out: running total over the held-out charts.");
    say!("rating: correlation between the component and the official rating, over the training charts.");
    Ok(())
}

/// Pearson correlation. 0 if either side is constant.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (mean_a, mean_b) = (mean(a), mean(b));
    let covariance: f64 = a.iter().zip(b).map(|(a, b)| (a - mean_a) * (b - mean_b)).sum();
    let spread = |values: &[f64], mean: f64| values.iter().map(|value| (value - mean).powi(2)).sum::<f64>().sqrt();
    let denominator = spread(a, mean_a) * spread(b, mean_b);
    if denominator > 0.0 { covariance / denominator } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COSTS: Costs = crate::chart_analyzer::TEST_COSTS;

    #[test]
    fn refuses_too_few_charts_to_split() {
        let config = TrainingConfig::new().with_batch_size(32);
        // 46 charts are the fewest that leave a batch of 32 to train on: 0.7 of 45 rounds to 31
        assert!(enough_charts(46, &config, 32).is_ok());
        let error = enough_charts(45, &config, 32).unwrap_err();
        assert!(error.contains("45 charts are too few: they split into 31 to train on, 7 to validate and 7 to test"), "{error}");
        // A PCA takes 2 rows, and something held out
        assert!(enough_charts(7, &config, 2).is_ok());
        assert!(enough_charts(2, &config, 2).is_err());
        assert!(enough_charts(3, &config, 2).is_err());
        assert!(enough_charts(0, &config, 2).is_err());
    }

    #[test]
    fn records_the_split_a_model_trained_on() {
        let samples: Vec<Sample> = (0..20)
            .map(|i| Sample {
                name: format!("Song {} [Hard]", i),
                simfile: PathBuf::from(format!("pack/Song {i}/Song {i}.sm")),
                pack: 0,
                difficulty: ChartDifficulty::Hard,
                features: vec![0.0],
                rating: (i % 19 + 1) as f32,
            })
            .collect();
        let training = TrainingConfig::new().with_split_seed(7);
        let split = SplitRecord::new(&samples, &training);
        let rows = |charts: &[SplitChart]| charts.iter().map(|chart| chart.row).collect::<Vec<_>>();
        let (train, validation, test) = train::split_rows(samples.len(), &training);
        assert_eq!((rows(&split.train), rows(&split.validation), rows(&split.test)), (train, validation, test));
        for chart in [&split.train, &split.validation, &split.test].into_iter().flatten() {
            let sample = &samples[chart.row];
            assert_eq!((chart.chart, chart.rating, chart.simfile.as_str()), (sample.name.as_str(), sample.rating, sample.simfile.to_str().unwrap()));
        }

        let json: serde_json::Value = serde_json::to_value(&split).unwrap();
        assert_eq!((json["split_seed"].as_u64(), json["train_ratio"].as_f64(), json["valid_ratio"].as_f64()), (Some(7), Some(0.7), Some(0.15)));
        let sizes = ["train", "validation", "test"].map(|name| json[name].as_array().unwrap().len());
        assert_eq!(sizes, [14, 3, 3]);
        let first = &json["test"][0];
        let row = split.test[0].row;
        assert_eq!(first, &serde_json::json!({"chart": format!("Song {row} [Hard]"), "rating": (row % 19 + 1) as f32, "simfile": format!("pack/Song {row}/Song {row}.sm")}));
    }

    #[test]
    fn takes_one_option_out_of_the_arguments() {
        let args = |args: &[&str]| args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>();
        let take = |given: &[&str]| take_option(&args(given), "--name");
        assert_eq!(take(&[]), Ok((None, vec![])));
        assert_eq!(take(&["songs"]), Ok((None, args(&["songs"]))));
        assert_eq!(take(&["--name", "x", "songs"]), Ok((Some("x".into()), args(&["songs"]))));
        assert_eq!(take(&["songs", "--name=x"]), Ok((Some("x".into()), args(&["songs"]))));
        assert_eq!(take(&["--name="]), Ok((Some(String::new()), vec![])));
        assert_eq!(take(&["--name"]), Err("--name needs a value".into()));
        assert_eq!(take(&["--name", "x", "--name=y"]), Err("--name is given twice".into()));
        assert_eq!(take(&["--names", "x"]), Err("unknown option --names".into()));
        assert_eq!(take(&["--model", "x"]), Err("unknown option --model".into()));

        let both = |given: &[&str]| take_options(&args(given), ["--folds", "--name"]);
        let values = |folds: Option<&str>, name: Option<&str>| [folds.map(String::from), name.map(String::from)];
        assert_eq!(both(&["songs", "--name=x", "--folds", "3"]), Ok((values(Some("3"), Some("x")), args(&["songs"]))));
        assert_eq!(both(&["--folds", "3"]), Ok((values(Some("3"), None), vec![])));
        assert_eq!(both(&["--folds", "3", "--folds=4"]), Err("--folds is given twice".into()));
        assert_eq!(both(&["--name", "x", "--seed", "1"]), Err("unknown option --seed".into()));
    }

    fn sample(simfile: &str, difficulty: ChartDifficulty, rating: f32) -> Sample {
        Sample { name: String::new(), simfile: PathBuf::from(simfile), pack: 0, difficulty, features: vec![0.0], rating }
    }

    #[test]
    fn finds_song_folders_one_folder_down_in_steps_packs() {
        let root = std::env::temp_dir().join(format!("ddr-trainer-song-root-{}", std::process::id()));
        let file = |path: &str| {
            let path = root.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "").unwrap();
        };
        file("steps/Old Pack/Song A/Song A.sm");
        file("steps/Old Pack/Song B/Song B.sm");
        file("full/Song C/Song C.sm");
        file("full/Song D/Song D.sm");
        file("full/banner.png");
        file("one song/Song E/Song E.ssc");
        let found = |pack: &str| song_root(root.join(pack)).unwrap();
        let (steps, full, one) = (found("steps"), found("full"), found("one song"));
        fs::remove_dir_all(&root).unwrap();
        assert_eq!((steps, full, one), (root.join("steps/Old Pack"), root.join("full"), root.join("one song")));
    }

    #[test]
    fn keeps_only_current_songs_of_older_packs() {
        let text = "song_id,song_name,romanized_name,deleted,difficulty,level,tier,sanbai,score\n\
                    a,Alpha,,0,Easy,3,0,3,0\na,Alpha,,0,Hard,9,0,9,0\n\
                    b,ベータ,Beta,0,Hard,12,0,12,0\n\
                    c,Tohoku EVOLVED,,0,Hard,10,0,10,0\n\
                    d,Delta,,0,Hard,8,0,8,0\n";
        let sanbai = Sanbai::parse(text).unwrap();
        // Two older packs, then a newer one taken whole
        let songs = [Songs::Current, Songs::Current, Songs::All];
        let charts = [
            ("old/Alpha/Alpha.sm", 0, ChartDifficulty::Hard, 9.0),
            ("old/Beta (Chinese)/Beta.sm", 0, ChartDifficulty::Hard, 12.0),
            ("old/Gamma/Gamma.sm", 0, ChartDifficulty::Hard, 5.0),
            ("old/Tohoku EVOLVED (TYPE1)/Tohoku EVOLVED.sm", 0, ChartDifficulty::Hard, 10.0),
            ("older/Alpha/Alpha.sm", 1, ChartDifficulty::Easy, 3.0),
            ("older/Alpha/Alpha.sm", 1, ChartDifficulty::Hard, 9.0),
            ("older/Alpha/Alpha.sm", 1, ChartDifficulty::Challenge, 11.0),
            ("older/Delta/Delta.sm", 1, ChartDifficulty::Hard, 7.0),
            ("older/Beta/Beta.sm", 1, ChartDifficulty::Hard, 12.0),
            ("new/Beta/Beta.sm", 2, ChartDifficulty::Hard, 12.0),
            ("new/Removed/Removed.sm", 2, ChartDifficulty::Easy, 4.0),
        ];
        let samples: Vec<Sample> = charts
            .iter()
            .map(|&(simfile, pack, difficulty, rating)| Sample { pack, ..sample(simfile, difficulty, rating) })
            .collect();
        let titles: HashMap<PathBuf, String> = charts
            .iter()
            .map(|&(simfile, ..)| {
                let folder = folder_name(Path::new(simfile));
                let title = if folder.starts_with("Beta") { "ベータ".to_string() } else { folder.replace(" (TYPE1)", "") };
                (PathBuf::from(simfile), title)
            })
            .collect();
        let (kept, left_out, missing_charts) = current_charts(samples, &songs, &titles, &sanbai);
        let kept: Vec<(String, ChartDifficulty)> = kept.iter().map(|sample| (sample.simfile.display().to_string(), sample.difficulty)).collect();
        assert_eq!(
            kept,
            [
                ("older/Alpha/Alpha.sm".to_string(), ChartDifficulty::Easy),
                ("older/Alpha/Alpha.sm".to_string(), ChartDifficulty::Hard),
                ("new/Beta/Beta.sm".to_string(), ChartDifficulty::Hard),
                ("new/Removed/Removed.sm".to_string(), ChartDifficulty::Easy),
            ]
        );
        let expected = [
            (LeftOut::Variant, 1),
            (LeftOut::NotInGame, 1),
            (LeftOut::Excluded, 1),
            (LeftOut::RatedOtherwise, 1),
            // old Alpha (older has it) and older Beta (new has it)
            (LeftOut::OlderCopy, 2),
        ];
        assert_eq!(left_out, BTreeMap::from(expected));
        // older Alpha's Challenge
        assert_eq!(missing_charts, 1);
    }

    #[test]
    fn groups_charts_by_song_with_covers() {
        let samples = [
            sample("pack/Song B/Song B.sm", ChartDifficulty::Easy, 3.0),
            sample("pack/Song A/Song A.sm", ChartDifficulty::Easy, 3.0),
            sample("pack/Song B/Song B.sm", ChartDifficulty::Hard, 9.0),
            sample("pack/Song A (cover)/Song A (cover).ssc", ChartDifficulty::Hard, 9.0),
            sample("other/Song C/Song C.sm", ChartDifficulty::Hard, 9.0),
        ];
        assert_eq!(song_ids(&samples), (vec![0, 1, 0, 1, 2], 3));
    }

    #[test]
    fn deals_songs_into_folds_and_splits_each_by_song() {
        let fold_of_song = song_folds(23, 5, 7);
        assert_eq!(fold_of_song, song_folds(23, 5, 7), "seeded");
        let sizes = (0..5).map(|fold| fold_of_song.iter().filter(|&&of| of == fold).count()).collect::<Vec<_>>();
        assert_eq!(sizes, [5, 5, 5, 4, 4]);
        assert_ne!(fold_of_song, song_folds(23, 5, 8));

        // Two charts a song, three for every third
        let song_of: Vec<usize> = (0..23).flat_map(|song| vec![song; 2 + usize::from(song % 3 == 0)]).collect();
        for fold in 0..5 {
            let (train, validation, test) = fold_split(&song_of, &fold_of_song, fold, 0.15, 7);
            let songs = |rows: &[usize]| rows.iter().map(|&row| song_of[row]).collect::<HashSet<_>>();
            let tested: HashSet<usize> = (0..23).filter(|&song| fold_of_song[song] == fold).collect();
            assert_eq!(songs(&test), tested);
            assert_eq!(songs(&validation).len(), 3, "15% of 23 songs");
            assert!(songs(&train).is_disjoint(&songs(&validation)) && songs(&train).is_disjoint(&tested));
            let mut all = [train, validation, test].concat();
            all.sort();
            assert_eq!(all, (0..song_of.len()).collect::<Vec<_>>(), "every row once");
        }
    }

    #[test]
    fn matches_statistic_names_with_wildcards() {
        // (pattern, name, whether it matches)
        let cases = [
            ("jump_count", "jump_count", true),
            ("jump_count", "jump_count_over_2", false),
            ("jump_count", "a_jump_count", false),
            ("jump_*", "jump_count_over_2", true),
            ("jump_*", "jump_", true),
            ("jump_*", "closest_jumps_speed", false),
            ("*_compression", "step_compression", true),
            ("*_compression", "step_compression_2", false),
            ("*bpm*", "max_note_bpm", true),
            ("*bpm*", "bpm_range", true),
            ("num_*_per_second", "num_stops_per_second", true),
            ("num_*_per_second", "num_stops_per_note", false),
            // Each part after the one before
            ("a*b*a", "aba", true),
            ("a*b*a", "ab", false),
            ("*", "anything", true),
            ("", "anything", false),
        ];
        for (pattern, name, matches) in cases {
            assert_eq!(matches_pattern(pattern, name), matches, "{pattern} against {name}");
        }
    }

    #[test]
    fn leaves_out_excluded_statistics() {
        let names: Vec<String> = ["num_notes", "jump_count", "jump_per_note", "mode_bpm"].map(String::from).to_vec();
        assert_eq!(kept_columns(&names, &[]), Ok((vec![0, 1, 2, 3], vec![])));
        assert_eq!(kept_columns(&names, &["jump_*"]), Ok((vec![0, 3], vec!["jump_count".to_string(), "jump_per_note".to_string()])));
        // In the statistics' order, a name once whatever matches it
        let (kept, excluded) = kept_columns(&names, &["mode_bpm", "*_count", "jump_count"]).unwrap();
        assert_eq!((kept, excluded), (vec![0, 2], vec!["jump_count".to_string(), "mode_bpm".to_string()]));
        assert!(kept_columns(&names, &["jump_*", "jumps"]).unwrap_err().contains("no statistic is called jumps"));
        assert!(kept_columns(&names, &["jump_*", ""]).unwrap_err().contains("with commas between"));
        assert!(kept_columns(&names, &["*"]).unwrap_err().contains("leaves no statistics"));

        // The real ones: every name once, a technique's 10 by its key
        let names = statistic_names();
        assert_eq!(names.len(), ChartStatistics::default().features().len());
        let (kept, excluded) = kept_columns(&names, &["syncopation_*", "*_compression"]).unwrap();
        assert_eq!((excluded.len(), kept.len()), (13, names.len() - 13));
        assert!(excluded.iter().all(|name| name.starts_with("syncopation_") || name.ends_with("_compression")));
        assert!(kept.is_sorted() && kept.iter().all(|&column| !excluded.contains(&names[column])));
    }

    #[test]
    fn matches_charts_to_sanbai_by_title() {
        let text = "song_id,song_name,romanized_name,deleted,difficulty,level,tier,sanbai,score\n\
                    a,\"Hello, planet\",,0,Hard,12,0.4,12.40,0\n\
                    b,恋,Koi,0,Easy,5,0.9,5.90,0\n\
                    c,take me higher,,0,Hard,13,0.2,13.20,0\n\
                    d,TAKE ME HIGHER,,0,Hard,16,0.6,16.60,0\n";
        let sanbai = Sanbai::parse(text).unwrap();
        let hard = ChartDifficulty::Hard;
        // By either name, in any spacing and case
        assert_eq!(sanbai.rating(&["Hello planet", "x"], hard, 12.0), Some(12.4));
        assert_eq!(sanbai.rating(&["Koi", "恋"], ChartDifficulty::Easy, 5.0), Some(5.9));
        // At another level, another difficulty, or no title
        assert_eq!(sanbai.rating(&["Hello, planet"], hard, 13.0), None);
        assert_eq!(sanbai.rating(&["Hello, planet"], ChartDifficulty::Easy, 12.0), None);
        assert_eq!(sanbai.rating(&["Goodbye"], hard, 12.0), None);
        // Two songs with one title
        assert_eq!(sanbai.rating(&["take me higher"], hard, 13.0), None);
        assert!(Sanbai::parse("song_id,level\n").is_err());
    }

    #[test]
    fn correlates_verdicts_within_each_level() {
        // Agreeing within each level, whatever the levels' own offsets
        let agree = [(4.0, 1.0, 0.9), (4.0, 0.0, 0.1), (12.0, -2.0, 0.1), (12.0, -1.0, 0.9)];
        assert!((within_level_r(&agree).unwrap() - 1.0).abs() < 1e-12);
        let disagree = [(4.0, 1.0, 0.1), (4.0, 0.0, 0.9)];
        assert!((within_level_r(&disagree).unwrap() + 1.0).abs() < 1e-12);
        // One chart a level has no spread
        assert_eq!(within_level_r(&[(4.0, 1.0, 0.1), (5.0, 0.0, 0.9)]), None);
    }

    #[test]
    fn quotes_csv_fields_that_need_it() {
        assert_eq!(csv_field("Fleur [Hard]"), "Fleur [Hard]");
        assert_eq!(csv_field("Hello, planet [Easy]"), "\"Hello, planet [Easy]\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(cost_fit::parse_csv(&format!("{}\n", csv_field("a, \"b\""))), [vec!["a, \"b\"".to_string()]]);
    }

    #[test]
    fn keeps_models_in_artifacts() {
        assert_eq!(model_dir("a3_model3"), Ok(Path::new("artifacts/a3_model3").to_path_buf()));
        assert_eq!(model_dir("model 2026-11-25.v2"), Ok(Path::new("artifacts/model 2026-11-25.v2").to_path_buf()));
        for name in ["", ".", "..", "../elsewhere", "nested/model", "/tmp/model", "back\\slash"] {
            assert!(model_dir(name).is_err(), "{name:?}");
        }
    }

    #[test]
    fn names_new_models_by_utc_time() {
        let at = |seconds: u64| UNIX_EPOCH + std::time::Duration::from_secs(seconds);
        assert_eq!(utc_stamp(at(0)), "19700101_000000Z");
        // Leap days, the end of a day and a year, and 2100, which isn't a leap year
        assert_eq!(utc_stamp(at(951_782_400)), "20000229_000000Z");
        assert_eq!(utc_stamp(at(951_868_799)), "20000229_235959Z");
        assert_eq!(utc_stamp(at(1_735_689_599)), "20241231_235959Z");
        assert_eq!(utc_stamp(at(1_764_120_615)), "20251126_013015Z");
        assert_eq!(utc_stamp(at(4_107_542_400)), "21000301_000000Z");

        // Runs saving in the same second each get a folder
        let root = std::env::temp_dir().join(format!("ddr-trainer-models-{}", std::process::id()));
        let made: Vec<PathBuf> = (0..3).map(|_| new_run_dir(&root, "model", at(1_764_120_615)).unwrap()).collect();
        let all_there = made.iter().all(|dir| dir.is_dir());
        fs::remove_dir_all(&root).unwrap();
        let names: Vec<_> = made.iter().map(|dir| dir.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["model_20251126_013015Z", "model_20251126_013015Z_2", "model_20251126_013015Z_3"]);
        assert!(all_there);
    }

    #[test]
    fn lists_song_folders_and_skips_those_with_no_simfile() {
        let songs = std::env::temp_dir().join(format!("ddr-trainer-song-folders-{}", std::process::id()));
        let file = |path: &str| {
            let path = songs.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "").unwrap();
        };
        file("Alpha/alpha.sm");
        file("Alpha/alpha.ogg");
        file("Beta/beta.ssc");
        file("Delta/two.sm");
        file("Delta/one.SM");
        fs::create_dir_all(songs.join("Delta/folder.ssc")).unwrap();
        fs::create_dir_all(songs.join("Empty")).unwrap();
        // The .ssc over the .sm files
        file("Gamma/two.sm");
        file("Gamma/one.sm");
        file("Gamma/gamma.ssc");
        file("banner.png");

        let folders = song_folders(&songs).unwrap();
        let listed: Vec<(usize, &str, Vec<String>)> = folders
            .iter()
            .map(|folder| {
                let names = folder.simfiles.iter().map(|path| path.file_name().unwrap().to_string_lossy().into_owned());
                (folder.index, folder.name.as_str(), names.collect())
            })
            .collect();
        let names = |names: &[&str]| names.iter().map(|name| name.to_string()).collect::<Vec<_>>();
        // Empty (3) is left out, and keeps its place in the count
        assert_eq!(
            listed,
            [
                (0, "Alpha", names(&["alpha.sm"])),
                (1, "Beta", names(&["beta.ssc"])),
                (2, "Delta", names(&["one.SM", "two.sm"])),
                (4, "Gamma", names(&["gamma.ssc"]))
            ]
        );
        fs::remove_dir_all(&songs).unwrap();
        assert!(song_folders(&songs).is_err());
    }

    #[test]
    fn checks_a_models_inputs_against_the_charts() {
        let inputs = |features: &[&str], analyzer: u64| ModelInputs { features: features.iter().map(|name| name.to_string()).collect(), analyzer };
        let trained = inputs(&["num_notes", "jack_count", "chart_dur"], 7);
        assert_eq!(trained.check(&trained), Ok(None));
        // Same names, another analyzer: a warning
        let warning = trained.check(&inputs(&["num_notes", "jack_count", "chart_dur"], 8)).unwrap();
        assert!(warning.is_some_and(|warning| warning.contains("has changed")));
        // Renamed, reordered, fewer or more: refused, whatever the analyzer
        let renamed = trained.check(&inputs(&["num_notes", "jack_total", "chart_dur"], 7)).unwrap_err();
        assert!(renamed.contains("statistic 2 was jack_count, and is now jack_total"), "{renamed}");
        assert!(trained.check(&inputs(&["jack_count", "num_notes", "chart_dur"], 7)).is_err());
        let fewer = trained.check(&inputs(&["num_notes", "jack_count"], 7)).unwrap_err();
        assert!(fewer.contains("it took 3 statistics, and charts now have 2"), "{fewer}");
        assert!(trained.check(&inputs(&["num_notes", "jack_count", "chart_dur", "more"], 8)).is_err());

        // Saved beside the model and read back
        let dir = std::env::temp_dir().join(format!("ddr-trainer-model-inputs-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        trained.save(&dir).unwrap();
        assert_eq!(ModelInputs::load(&dir).unwrap(), trained);
        fs::remove_dir_all(&dir).unwrap();
        assert!(ModelInputs::load(&dir).is_err());
        // What `train` saves is what `infer` compares with
        let names: Vec<String> = ChartStatistics::default().features().into_iter().map(|(name, _)| name).collect();
        assert_eq!(ModelInputs::now(names.clone()).check(&ModelInputs::now(names)), Ok(None));
    }

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| arg.to_string()).collect()
    }

    fn cost_values(mut costs: Costs) -> [f64; 24] {
        costs.parameters().map(|(_, value)| *value)
    }

    #[test]
    fn reads_any_cost_or_setting_from_analyze_arguments() {
        let args = strings(&[
            "song.sm",
            "--jack-base",
            "0.5",
            "hard",
            "--max-speed=3",
            "--turn-back-degrees",
            "90",
            "--doublestep-speed",
            "-1",
            "--footing",
            "--window-seconds=2.5",
        ]);
        let given = analyze_args(&args).unwrap();
        assert_eq!((given.simfile, given.difficulty, given.footing), ("song.sm", "hard", true));
        let mut expected = crate::chart_analyzer::COSTS;
        (expected.jack.base, expected.max_speed, expected.doublestep.speed) = (0.5, 3.0, -1.0);
        assert_eq!(cost_values(given.costs), cost_values(expected));
        assert_eq!(given.settings, Settings { turn_back_degrees: 90, window_seconds: 2.5, ..SETTINGS });
        // None given: the defaults
        let stdin = strings(&["-", "easy"]);
        let defaults = analyze_args(&stdin).unwrap();
        assert_eq!(cost_values(defaults.costs), cost_values(crate::chart_analyzer::COSTS));
        assert_eq!(defaults.settings, SETTINGS);
        assert!(!defaults.footing);
        // A flag: it takes no value
        assert_eq!(analyze_args(&strings(&["song.sm", "--footing", "hard"])).unwrap().difficulty, "hard");
        // Every cost and setting has an option
        let mut costs = crate::chart_analyzer::COSTS;
        let names = costs.parameters().map(|(name, _)| name).into_iter().chain(SETTINGS.values().map(|(name, _)| name));
        for name in names {
            let flag = option(name);
            assert!(analyze_args(&strings(&["song.sm", "hard", &flag, "10"])).is_ok(), "{flag}");
            assert!(analyze_usage().contains(&format!("{flag} ")), "{flag}");
        }

        let refused = [
            (&["song.sm"][..], "takes a simfile and a difficulty"),
            (&["song.sm", "hard", "challenge"], "takes a simfile and a difficulty"),
            (&["song.sm", "hard", "--jack"], "unknown option --jack"),
            (&["song.sm", "hard", "--jack-base"], "--jack-base needs a value"),
            (&["song.sm", "hard", "--jack-base", "x"], "--jack-base takes a number, not x"),
            (&["song.sm", "hard", "--jack-base", "NaN"], "--jack-base must be a number or inf, not NaN"),
            (&["song.sm", "hard", "--jack-speed=-inf"], "--jack-speed must be a number or inf, not -inf"),
            (&["song.sm", "hard", "--max-speed", "inf"], "--max-speed must be a finite number, not inf"),
            (&["song.sm", "hard", "--reference-gap", "inf"], "--reference-gap must be a finite number, not inf"),
            (&["song.sm", "hard", "--jack-base", "1", "--jack-base=2"], "--jack-base is given twice"),
            (&["song.sm", "hard", "--reference-gap", "0"], "--reference-gap must be more than 0, not 0"),
            (&["song.sm", "hard", "--max-speed", "-1"], "--max-speed must be 0 or more, not -1"),
            (&["song.sm", "hard", "--max-footing-branches", "0"], "--max-footing-branches must be a whole number, 1 or more, not 0"),
            (&["song.sm", "hard", "--min-drill-notes=2.5"], "--min-drill-notes must be a whole number, 2 or more, not 2.5"),
            (&["song.sm", "hard", "--window-seconds", "0"], "--window-seconds must be more than 0, not 0"),
            (&["song.sm", "hard", "--footing", "--footing"], "--footing is given twice"),
            (&["song.sm", "hard", "--footing=yes"], "--footing takes no value"),
        ];
        for (args, error) in refused {
            let message = analyze_args(&strings(args)).unwrap_err();
            assert!(message.contains(error), "{args:?}: {message}");
        }
        assert!(analyze_args(&strings(&["song.sm", "hard", "--max-speed", "0"])).is_ok());
        // A penalty over the most one counts as, infinity too: set to that
        let capped = strings(&["song.sm", "hard", "--sideswitch-base", "inf", "--jack-speed=1e30", "--bracket-base", "-1e30"]);
        let mut expected = crate::chart_analyzer::COSTS;
        (expected.sideswitch.base, expected.jack.speed, expected.bracket.base) = (MAX_COST, MAX_COST, -1e30);
        assert_eq!(cost_values(analyze_args(&capped).unwrap().costs), cost_values(expected));
    }

    #[test]
    fn analyzes_a_chart_as_json() {
        let sim = SimFile::from_file("test_resources/footing_stats.sm").unwrap();
        // The one with the most awkward steps
        let challenge = sim
            .step_charts
            .iter()
            .find_map(|chart| match chart {
                StepChart::DanceSingle(single) if single.difficulty == ChartDifficulty::Challenge => Some(single),
                _ => None,
            })
            .unwrap();
        let doubled = {
            let mut costs = COSTS;
            for (name, value) in costs.parameters() {
                if !matches!(name, "reference gap" | "max speed") {
                    *value *= 2.0;
                }
            }
            costs
        };
        let other_settings = Settings { window_seconds: 2.0, min_stream_notes: 3, max_footing_branches: 7, ..SETTINGS };
        for (costs, settings) in [(COSTS, SETTINGS), (doubled, other_settings)] {
            let text = analysis_json(&sim, "cHaLlEnGe", &costs, &settings, false).unwrap();
            let json: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!((json["title"].as_str(), json["difficulty"].as_str()), (Some(sim.title.as_str()), Some("Challenge")));
            assert_eq!(json["rating"].as_f64(), Some(challenge.rating));
            assert_eq!(json["analyzer"].as_str(), Some(format!("{:016x}", analyzer_fingerprint()).as_str()));
            assert!(json.get("footing").is_none(), "only with --footing");
            let given: Vec<f64> = json["costs"].as_object().unwrap().values().map(|value| value.as_f64().unwrap()).collect();
            let mut sorted = costs;
            let mut by_name: Vec<(String, f64)> = sorted.parameters().iter().map(|(name, value)| (name.replace(' ', "_"), **value)).collect();
            by_name.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(given, by_name.iter().map(|(_, value)| *value).collect::<Vec<_>>());
            // Every setting, whole numbers as such
            for (name, value) in settings.values() {
                assert_eq!(json["settings"][name.replace(' ', "_")].as_f64(), Some(value), "{name}");
            }
            let branches = format!("\"max_footing_branches\": {},", settings.max_footing_branches);
            assert!(text.contains(&branches), "{branches}");

            // Every statistic, as the analyzer gives it with these costs and settings, in its order
            let expected = analyze_single_chart(challenge, &challenge.bpms, &challenge.stops, &costs, &settings).features();
            let features = &text[text.find("\"features\"").unwrap()..];
            let mut last = 0;
            for (name, value) in &expected {
                let at = features.find(&format!("\"{name}\":")).unwrap_or_else(|| panic!("{name} missing"));
                assert!(at > last, "{name} out of order");
                last = at;
                let found = &json["features"][name];
                assert_eq!(found.as_f64(), value.is_finite().then_some(*value), "{name}");
            }
            assert_eq!(json["features"].as_object().unwrap().len(), expected.len());
        }
        // The costs reach the footing: every penalty doubled, the same footing at twice the cost
        let cost_per_note = |costs: &Costs| {
            let json: serde_json::Value = serde_json::from_str(&analysis_json(&sim, "challenge", costs, &SETTINGS, false).unwrap()).unwrap();
            json["features"]["mean_cost_per_note"].as_f64().unwrap()
        };
        let (single, double) = (cost_per_note(&COSTS), cost_per_note(&doubled));
        assert!(single > 0.0 && (double - 2.0 * single).abs() < 1e-3, "{single} doubled is {double}");

        let none = analysis_json(&sim, "edit", &COSTS, &SETTINGS, false).unwrap_err();
        assert!(none.contains("has no edit chart (it has Beginner, Easy, Medium, Hard, Challenge)"), "{none}");
    }

    #[test]
    fn prints_the_footing_with_every_technique() {
        let sim = SimFile::from_file("test_resources/footing_stats.sm").unwrap();
        let text = analysis_json(&sim, "challenge", &COSTS, &SETTINGS, true).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let footing = &json["footing"];
        let number = |value: &serde_json::Value| value.as_u64().unwrap() as usize;
        // The fixture's segments: 6, 4 and 6 notes
        let segments = footing["segments"].as_array().unwrap();
        let spans: Vec<(usize, usize)> = segments.iter().map(|segment| (number(&segment["first_note"]), number(&segment["notes"]))).collect();
        assert_eq!(spans, [(0, 6), (6, 4), (10, 6)]);
        assert!(segments.iter().all(|segment| segment["exact"] == true));
        let notes = footing["notes"].as_array().unwrap();
        assert_eq!(Some(notes.len() as f64), json["features"]["num_notes"].as_f64());

        // Each note footed as the analyzer foots it
        let challenge = sim
            .step_charts
            .iter()
            .find_map(|chart| match chart {
                StepChart::DanceSingle(single) if single.difficulty == ChartDifficulty::Challenge => Some(single),
                _ => None,
            })
            .unwrap();
        let analysis = analyze_chart(challenge, &challenge.bpms, &challenge.stops, &COSTS, &SETTINGS);
        let footed = analysis
            .segments
            .iter()
            .enumerate()
            .flat_map(|(index, segment)| segment.notes.iter().zip(&segment.footing.steps).map(move |(note, step)| (index, note, step)));
        assert_eq!(footed.clone().count(), notes.len());
        for ((segment, note, step), json) in footed.zip(notes) {
            assert_eq!((number(&json["segment"]), json["time"].as_f64(), json["beat"].as_f64()), (segment, Some(note.time), Some(note.beat)));
            let arrows = json["arrows"].as_array().unwrap();
            let feet: Vec<(&str, Option<&str>)> = arrows.iter().map(|arrow| (arrow["arrow"].as_str().unwrap(), arrow["foot"].as_str())).collect();
            let expected: Vec<(&str, Option<&str>)> =
                (0..4).filter(|&arrow| note.steps(arrow)).map(|arrow| (ARROW_NAMES[arrow], step.foot(arrow).map(foot_name))).collect();
            assert_eq!(feet, expected, "note at {}", note.time);
            // A hold or roll says when it ends, a tap doesn't
            for arrow in arrows {
                let held = matches!(arrow["note"].as_str(), Some("hold" | "roll"));
                assert_eq!(arrow.get("end").is_some(), held, "{arrow}");
                assert!(!held || arrow["end"].as_f64().unwrap() > note.time, "{arrow}");
            }
        }

        // Every technique the statistics count, in order of its first note, inside the chart
        let found = footing["techniques"].as_array().unwrap();
        let count = |key: &str| found.iter().filter(|technique| technique["technique"] == key).count();
        for technique in Technique::ALL {
            let key = technique.key();
            assert_eq!(Some(count(&key) as f64), json["features"][format!("{key}_count")].as_f64(), "{key}");
        }
        let firsts: Vec<usize> = found.iter().map(|technique| number(&technique["first_note"])).collect();
        assert!(firsts.is_sorted(), "{firsts:?}");
        assert!(found.iter().all(|technique| number(&technique["first_note"]) + number(&technique["notes"]) <= notes.len()));
        // The fixture's: 2 doublesteps, a bracket and a gallop
        assert_eq!((count("doublestep"), count("bracket"), count("gallop")), (2, 1, 1));
        // A kind is found beside what it's a kind of
        for kind in found.iter().filter(|technique| technique.get("kind_of").is_some()) {
            let parent = |technique: &&serde_json::Value| technique["technique"] == kind["kind_of"] && technique["first_note"] == kind["first_note"];
            assert!(found.iter().any(|technique| parent(&technique)), "{kind}");
        }
        // The notes' costs make the segments'
        let total = |values: &[serde_json::Value]| values.iter().map(|value| value["cost"].as_f64().unwrap()).sum::<f64>();
        assert!((total(notes) - total(segments)).abs() < 1e-9);
    }

    #[test]
    fn draws_a_footing() {
        // Typed arrows are 0.125 s apart: 8 notes a second
        let crossover = foot_arrows("LDR", &COSTS, 1.0).unwrap();
        let numbered = "  #   ←  ↓  ↑  →\n  1   L  .  .  .   triple 8.0/s\n  2   .  R  .  .\n  3   .  .  .  L   full crossover 8.0/s\n";
        assert_eq!(footing_table(&crossover, false), numbered);
        // 16ths: the fifth starts the second measure
        let by_measure = footing_table(&foot_arrows("LR[UD]LR", &COSTS, 1.0).unwrap(), true);
        let rows: Vec<&str> = by_measure.lines().collect();
        assert_eq!(rows[0], "measure  beat   ←  ↓  ↑  →");
        assert_eq!(rows[1], "      1  1.00   L  .  .  .");
        assert_eq!(rows[2], "         1.25   .  .  .  R");
        assert_eq!(rows[4], "         1.75   L  .  .  .");
        assert_eq!(rows[5], "         2.00   .  .  .  R");
        assert!(rows[3].ends_with("L  R  .") || rows[3].ends_with("R  L  ."), "{}", rows[3]);

        // A hold, and the right foot taking it over
        let cheap = Costs { freeze_switch: Cost { base: 0.5, speed: 0.0 }, ..COSTS };
        let held = footing_table(&foot_arrows("D+RUD-", &cheap, 1.0).unwrap(), false);
        let rows: Vec<&str> = held.lines().collect();
        assert_eq!(rows[1], "  1   .  L  .  .");
        assert_eq!(rows[2], "  2   .  |  .  R");
        assert_eq!(rows[3], "  3   .  r  L  .   freeze switch 8.0/s");

        // Empty rows: Down is let go on the third
        let paused = footing_table(&foot_arrows("D+  D-L", &COSTS, 1.0).unwrap(), false);
        let rows: Vec<&str> = paused.lines().collect();
        assert_eq!(rows[2..5], ["      .  |  .  .", "      .  |  .  .", "      .  .  .  ."]);
        assert!(rows[5].starts_with("  2   "), "{}", rows[5]);
    }

    #[test]
    fn fits_costs_that_exist() {
        let mut costs = COSTS;
        let names: Vec<&str> = costs.parameters().iter().map(|(name, _)| *name).collect();
        for fitted in FITTED {
            assert!(names.contains(&fitted), "{fitted}");
        }
    }

    #[test]
    fn picks_values_to_scan() {
        assert_eq!(scan_values("jack speed", 3.0), [0.0, 0.75, 1.5, 6.0, 12.0]);
        assert_eq!(scan_values("jack base", 0.0), [0.5, 1.0, 2.0, 4.0]);
        // Never 0
        assert_eq!(scan_values("reference gap", 0.125), [0.0625, 0.09375, 0.1875, 0.25]);
        assert_eq!(scan_values("max speed", 4.0), [1.0, 2.0, 8.0, 16.0]);
        let mut costs = COSTS;
        for (name, value) in costs.parameters() {
            assert!(!scan_values(name, *value).contains(value), "{name}");
        }
    }

    #[test]
    fn adds_footings_up() {
        let segments = [foot_arrows("LDR", &COSTS, 1.0).unwrap(), foot_arrows("LDDR", &COSTS, 1.0).unwrap(), foot_arrows("LR", &COSTS, 1.0).unwrap()];
        let summary = Summary::of(&segments);
        assert_eq!(summary.notes, 9);
        // Typed arrows are at speed 1
        let cost = COSTS.vertical_footswitch.base
            + COSTS.vertical_footswitch.speed
            + COSTS.horizontal_crossover.base
            + COSTS.horizontal_crossover.speed;
        assert_eq!(summary.footing(), format!("1 down footswitch, 1 full crossover (cost {cost:.1})"));
        // LDR and LR
        assert_eq!(summary.patterns(), "patterns: 1 gallop, 1 triple");
        assert_eq!(Summary::of(&segments[1..2]).patterns(), "patterns: none");
        assert_eq!(Summary::of(&segments[2..]).footing(), "nothing awkward");
    }
}
