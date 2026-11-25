//! The command-line tools (see `main`). Each is a script over the rest of the crate.

use std::{
    error::Error,
    fmt::Write as _,
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
    thread,
    time::Instant,
};

use burn::optim::AdamWConfig;
use burn::prelude::*;
use serde::{Deserialize, Serialize};

use crate::chart_analyzer::{
    COSTS, ChartStatistics, Costs, Foot, FootedSegment, SETTINGS, Settings, Source, Technique, analyze_single_chart,
    analyze_singles, foot_arrows, foot_chart, footing_segments,
};
#[cfg(test)]
use crate::chart_analyzer::Cost;
use crate::cma_es;
use crate::cost_fit::{self, COLUMNS, SCORED, Sheet};
use crate::dataset::Dataset;
use crate::device::DevicePreference;
use crate::model::ModelConfig;
use crate::pca::Pca;
use crate::sim_loader::{ChartDifficulty, SimFile, SingleChart, StepChart};
use crate::stats_cache::{StatsCache, analyzer_fingerprint};
use crate::train::{self, TrainedModel, TrainingConfig};

const CACHE_FILE: &str = "artifacts/stats_cache.json";
const MODEL_DIR: &str = "artifacts/a3_model3/";
/// In `MODEL_DIR` (see `ModelInputs`).
const INPUTS_FILE: &str = "inputs.json";
const PCA_FILE: &str = "artifacts/pca.json";
/// Puts all 4 [ ]DENTITY charts in the test split.
const SPLIT_SEED: u64 = 643;
/// Test-set predictions to print at the end.
const EXAMPLES: usize = 12;
/// An official rating N spans N.0 up to N + 1 on the model's scale. A chart trains as its middle.
const LEVEL_MIDDLE: f32 = 0.5;
const SCALE_NOTE: &str = "predicted: 14.00 is the easiest 14, 14.50 a typical 14";

struct Sample {
    name: String,
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

/// Trains a model to predict DDR chart ratings from chart statistics, on every dance-single
/// chart in a folder of song folders (the argument, or DDR A3), and saves it to `MODEL_DIR`.
pub fn train(args: &[String]) -> Result<(), Box<dyn Error>> {
    let (feature_names, samples) = cached_samples(&songs_folder(args)?)?;

    // Small batches: with only a few hundred charts, they make more steps per epoch.
    let inputs: Vec<Vec<f32>> = samples.iter().map(|sample| sample.features.clone()).collect();
    let targets: Vec<Vec<f32>> = samples.iter().map(|sample| vec![sample.rating + LEVEL_MIDDLE]).collect();
    let dataset = Dataset::from_rows(&inputs, &targets, &DevicePreference::default().device());
    let model_config = ModelConfig::funnel(inputs[0].len(), 1, 2048, 1).with_dropout(0.2);
    let weight_decay = 1.0;
    let input_noise = 0.2;
    let training = TrainingConfig::new()
        .with_batch_size(32)
        .with_input_noise(input_noise)
        .with_split_seed(SPLIT_SEED)
        .with_optimizer(AdamWConfig::new().with_weight_decay(weight_decay));
    enough_charts(samples.len(), &training, training.batch_size)?;
    say!(
        "model: {} (inputs, hidden layers, output), dropout {}, weight decay {weight_decay}, input noise {input_noise}",
        model_config.shape(),
        model_config.dropout
    );
    let (trained, test) = model_config.train(&dataset, &training);
    trained.save(&model_config, MODEL_DIR)?;
    ModelInputs::now(feature_names).save(Path::new(MODEL_DIR))?;
    say!("\nsaved the model to {MODEL_DIR}/\n");

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

/// Fits a PCA of the chart statistics of a folder of song folders (the argument, or DDR A3) on
/// the split `train` trains on, prints it and saves it to `PCA_FILE`.
pub fn pca(args: &[String]) -> Result<(), Box<dyn Error>> {
    let (feature_names, samples) = cached_samples(&songs_folder(args)?)?;
    report_pca(&feature_names, &samples)
}

/// Rates every dance-single chart of each simfile with the model in `MODEL_DIR`. Refuses if
/// the model was trained on other statistics than charts have now.
pub fn infer(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.is_empty() {
        return Err("infer needs a simfile or song folder".into());
    }
    let device = DevicePreference::default().device();
    let model = TrainedModel::load(MODEL_DIR, &device)
        .map_err(|e| format!("couldn't load the model in {MODEL_DIR} ({e}): has `train` been run?"))?;
    let trained_on = ModelInputs::load(Path::new(MODEL_DIR))
        .map_err(|e| format!("the model in {MODEL_DIR} doesn't say which statistics it takes ({e}): train it again"))?;
    let names = ChartStatistics::default().features().into_iter().map(|(name, _)| name).collect();
    if let Some(warning) = trained_on.check(&ModelInputs::now(names))? {
        eprintln!("warning: {warning}");
    }
    // A mismatch would broadcast silently
    if trained_on.features.len() != model.model.num_inputs() {
        return Err(format!(
            "the model in {MODEL_DIR} takes {} statistics, but {INPUTS_FILE} lists {}: train it again",
            model.model.num_inputs(),
            trained_on.features.len()
        )
        .into());
    }
    say!("{SCALE_NOTE}\n");
    for path in args {
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
/// (`--jack-base 0.5` or `--jack-base=0.5`), the rest being `COSTS` and `SETTINGS`. With no
/// arguments, lists them.
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
    print_report(&(analysis_json(&sim, given.difficulty, &given.costs, &given.settings)? + "\n"))
}

/// A cost's or setting's option: `--jack-base` for "jack base".
fn option(name: &str) -> String {
    format!("--{}", name.replace(' ', "-"))
}

/// `analyze`'s arguments, and every cost and setting with its default.
fn analyze_usage() -> String {
    let mut usage = String::from(
        "usage: analyze <simfile> <difficulty> [--<cost or setting> <value>]...\n\n\
         A simfile is a .sm file, its song folder, or - for its text on stdin.\n\ncosts (default):\n",
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
}

fn analyze_args(args: &[String]) -> Result<AnalyzeArgs<'_>, String> {
    let (mut costs, mut settings) = (COSTS, SETTINGS);
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
        [simfile, difficulty] => Ok(AnalyzeArgs { simfile, difficulty, costs, settings }),
        _ => Err(format!("analyze takes a simfile and a difficulty, then any costs and settings\n\n{}", analyze_usage())),
    }
}

/// Sets the cost called `name` (as in `Costs::parameters`), or says why `value` can't be it.
fn set_cost(costs: &mut Costs, name: &str, value: f64) -> Result<(), String> {
    let must = match name {
        _ if !value.is_finite() => Some("a finite number"),
        // A time, and a speed
        "reference gap" if value <= 0.0 => Some("more than 0"),
        "max speed" if value < 0.0 => Some("0 or more"),
        _ => None,
    };
    if let Some(must) = must {
        return Err(format!("must be {must}"));
    }
    let mut parameters = costs.parameters();
    let (_, cost) = parameters.iter_mut().find(|(cost, _)| *cost == name).ok_or_else(|| format!("there is no cost called {name}"))?;
    **cost = value;
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
/// JSON.
fn analysis_json(sim: &SimFile, difficulty: &str, costs: &Costs, settings: &Settings) -> Result<String, String> {
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
    let stats = analyze_single_chart(chart, &sim.bpms, &sim.stops, costs, settings);
    let mut costs = *costs;
    let analysis = Analysis {
        title: &sim.title,
        artist: &sim.artist,
        difficulty: chart.difficulty,
        rating: chart.rating,
        analyzer: format!("{:016x}", analyzer_fingerprint()),
        costs: Named(costs.parameters().iter().map(|(name, value)| (name.replace(' ', "_"), **value)).collect()),
        settings: *settings,
        features: Named(stats.features()),
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

/// The charts of a tool's songs (its second argument, or DDR A3) that its sheet (the first) has
/// counts for.
fn charts_in_sheet(args: &[String], tool: &str) -> Result<Vec<cost_fit::Chart>, Box<dyn Error>> {
    let (sheet, songs) = match args {
        [sheet] => (sheet, default_songs_folder()),
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
                        segments: footing_segments(chart, &sim.bpms, &sim.stops),
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

/// A song folder with at least one `.sm` file.
struct SongFolder {
    /// Its place among all the song folders, in name order
    index: usize,
    name: String,
    /// In name order
    simfiles: Vec<PathBuf>,
}

/// The song folders in `songs`, in name order. One with no `.sm` file is left out with a
/// warning, which says if it has a `.ssc` (not read), and so is their number.
fn song_folders(songs: &Path) -> io::Result<Vec<SongFolder>> {
    let has = |path: &Path, extension: &str| path.extension().is_some_and(|ext| ext == extension);
    let (mut folders, mut skipped) = (Vec::new(), 0);
    for (index, song) in sorted_paths(songs)?.into_iter().filter(|path| path.is_dir()).enumerate() {
        let name = song.file_name().unwrap_or_default().to_string_lossy().into_owned();
        let files = sorted_paths(&song)?;
        let simfiles: Vec<PathBuf> = files.iter().filter(|path| has(path, "sm")).cloned().collect();
        if simfiles.is_empty() {
            let why = if files.iter().any(|path| has(path, "ssc")) { "only a .ssc, which isn't read" } else { "no simfile" };
            eprintln!("skipping {name}: no .sm file ({why})");
            skipped += 1;
            continue;
        }
        folders.push(SongFolder { index, name, simfiles });
    }
    if skipped > 0 {
        eprintln!("{skipped} of the {} song folders in {} have no .sm file", folders.len() + skipped, songs.display());
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
            let footed = foot_chart(chart, &sim.bpms, &sim.stops, &COSTS);
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
    let footed = foot_chart(chart, &sim.bpms, &sim.stops, &COSTS);
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

/// The folder of song folders a tool was given, or DDR A3.
fn songs_folder(args: &[String]) -> Result<PathBuf, Box<dyn Error>> {
    match args {
        [] => Ok(default_songs_folder()),
        [songs] => Ok(PathBuf::from(songs)),
        _ => Err("expected one folder of song folders at most".into()),
    }
}

/// DDR A3's songs in this machine's ITGmania install.
fn default_songs_folder() -> PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_default();
    Path::new(&home).join("Documents/ITGmania-0.9.0-Linux/itgmania/Songs/A3")
}

/// `load_samples` through the statistics cache in `CACHE_FILE`, which it updates.
fn cached_samples(songs: &Path) -> Result<(Vec<String>, Vec<Sample>), Box<dyn Error>> {
    let mut cache = StatsCache::load(CACHE_FILE)?;
    let (feature_names, samples) = load_samples(songs, &mut cache)?;
    if samples.is_empty() {
        return Err(format!("no dance-single charts found in {}", songs.display()).into());
    }
    if let Some(dir) = Path::new(CACHE_FILE).parent() {
        fs::create_dir_all(dir)?;
    }
    cache.save(CACHE_FILE)?;
    say!(
        "{} charts from {} ({} files from the cache, {} analyzed), {} features each",
        samples.len(),
        songs.display(),
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

/// Every dance-single chart in the `.sm` files of the song folders in `songs`. Folders with no
/// `.sm`, files that don't load, and charts with a non-finite statistic, are skipped with a
/// warning.
///
/// Read in sorted order: listing order differs between machines, and would change the seeded
/// split.
fn load_samples(songs: &Path, cache: &mut StatsCache) -> Result<(Vec<String>, Vec<Sample>), Box<dyn Error>> {
    let mut feature_names = Vec::new();
    let mut samples = Vec::new();
    for song in song_folders(songs)? {
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
                    Ok(features) => samples.push(Sample { name, features, rating: chart.rating as f32 }),
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
    fn lists_song_folders_and_skips_those_with_no_sm() {
        let songs = std::env::temp_dir().join(format!("ddr-trainer-song-folders-{}", std::process::id()));
        let file = |path: &str| {
            let path = songs.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "").unwrap();
        };
        file("Alpha/alpha.sm");
        file("Alpha/alpha.ogg");
        file("Beta/beta.ssc");
        fs::create_dir_all(songs.join("Empty")).unwrap();
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
        // Beta (1) and Empty (2) are left out, and keep their places in the count
        assert_eq!(listed, [(0, "Alpha", vec!["alpha.sm".to_string()]), (3, "Gamma", vec!["one.sm".to_string(), "two.sm".to_string()])]);
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
            "--window-seconds=2.5",
        ]);
        let given = analyze_args(&args).unwrap();
        assert_eq!((given.simfile, given.difficulty), ("song.sm", "hard"));
        let mut expected = crate::chart_analyzer::COSTS;
        (expected.jack.base, expected.max_speed, expected.doublestep.speed) = (0.5, 3.0, -1.0);
        assert_eq!(cost_values(given.costs), cost_values(expected));
        assert_eq!(given.settings, Settings { turn_back_degrees: 90, window_seconds: 2.5, ..SETTINGS });
        // None given: the defaults
        let stdin = strings(&["-", "easy"]);
        let defaults = analyze_args(&stdin).unwrap();
        assert_eq!(cost_values(defaults.costs), cost_values(crate::chart_analyzer::COSTS));
        assert_eq!(defaults.settings, SETTINGS);
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
            (&["song.sm", "hard", "--jack-base", "NaN"], "--jack-base must be a finite number, not NaN"),
            (&["song.sm", "hard", "--jack-base", "1", "--jack-base=2"], "--jack-base is given twice"),
            (&["song.sm", "hard", "--reference-gap", "0"], "--reference-gap must be more than 0, not 0"),
            (&["song.sm", "hard", "--max-speed", "-1"], "--max-speed must be 0 or more, not -1"),
            (&["song.sm", "hard", "--max-footing-branches", "0"], "--max-footing-branches must be a whole number, 1 or more, not 0"),
            (&["song.sm", "hard", "--min-drill-notes=2.5"], "--min-drill-notes must be a whole number, 2 or more, not 2.5"),
            (&["song.sm", "hard", "--window-seconds", "0"], "--window-seconds must be more than 0, not 0"),
        ];
        for (args, error) in refused {
            let message = analyze_args(&strings(args)).unwrap_err();
            assert!(message.contains(error), "{args:?}: {message}");
        }
        assert!(analyze_args(&strings(&["song.sm", "hard", "--max-speed", "0"])).is_ok());
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
            let text = analysis_json(&sim, "cHaLlEnGe", &costs, &settings).unwrap();
            let json: serde_json::Value = serde_json::from_str(&text).unwrap();
            assert_eq!((json["title"].as_str(), json["difficulty"].as_str()), (Some(sim.title.as_str()), Some("Challenge")));
            assert_eq!(json["rating"].as_f64(), Some(challenge.rating));
            assert_eq!(json["analyzer"].as_str(), Some(format!("{:016x}", analyzer_fingerprint()).as_str()));
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
            let expected = analyze_single_chart(challenge, &sim.bpms, &sim.stops, &costs, &settings).features();
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
            let json: serde_json::Value = serde_json::from_str(&analysis_json(&sim, "challenge", costs, &SETTINGS).unwrap()).unwrap();
            json["features"]["mean_cost_per_note"].as_f64().unwrap()
        };
        let (single, double) = (cost_per_note(&COSTS), cost_per_note(&doubled));
        assert!(single > 0.0 && (double - 2.0 * single).abs() < 1e-3, "{single} doubled is {double}");

        let none = analysis_json(&sim, "edit", &COSTS, &SETTINGS).unwrap_err();
        assert!(none.contains("has no edit chart (it has Beginner, Easy, Medium, Hard, Challenge)"), "{none}");
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
