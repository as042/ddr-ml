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

use crate::chart_analyzer::{
    COSTS, ChartStatistics, Costs, Events, Foot, FootedSegment, Patterns, analyze_singles, foot_arrows, foot_chart,
    footing_segments,
};
#[cfg(test)]
use crate::chart_analyzer::Cost;
use crate::cma_es;
use crate::cost_fit::{self, COLUMNS, SCORED, Sheet};
use crate::dataset::Dataset;
use crate::device::DevicePreference;
use crate::model::ModelConfig;
use crate::pca::Pca;
use crate::sim_loader::{SimFile, SingleChart, StepChart};
use crate::stats_cache::StatsCache;
use crate::train::{self, TrainedModel, TrainingConfig};

const CACHE_FILE: &str = "artifacts/stats_cache.json";
const MODEL_DIR: &str = "artifacts/a3_model3/";
const PCA_FILE: &str = "artifacts/pca.json";
/// Puts all 4 [ ]DENTITY charts in the test split.
const SPLIT_SEED: u64 = 643;
/// Test-set predictions to print at the end.
const EXAMPLES: usize = 12;

struct Sample {
    name: String,
    features: Vec<f32>,
    rating: f32,
}

/// Trains a model to predict DDR chart ratings from chart statistics, on every dance-single
/// chart in a folder of song folders (the argument, or DDR A3), and saves it to `MODEL_DIR`.
pub fn train(args: &[String]) -> Result<(), Box<dyn Error>> {
    let (_, samples) = cached_samples(&songs_folder(args)?)?;

    // Small batches: with only a few hundred charts, they make more steps per epoch.
    let inputs: Vec<Vec<f32>> = samples.iter().map(|sample| sample.features.clone()).collect();
    let ratings: Vec<Vec<f32>> = samples.iter().map(|sample| vec![sample.rating]).collect();
    let dataset = Dataset::from_rows(&inputs, &ratings, &DevicePreference::default().device());
    let model_config = ModelConfig::funnel(inputs[0].len(), 2, 256, 1).with_dropout(0.5);
    let weight_decay = 1.0;
    let input_noise = 0.2;
    println!(
        "model: {} (inputs, hidden layers, output), dropout {}, weight decay {weight_decay}, input noise {input_noise}",
        model_config.shape(),
        model_config.dropout
    );
    let training = TrainingConfig::new()
        .with_batch_size(32)
        .with_input_noise(input_noise)
        .with_split_seed(SPLIT_SEED)
        .with_optimizer(AdamWConfig::new().with_weight_decay(weight_decay));
    let (trained, test) = model_config.train(&dataset, &training);
    trained.save(&model_config, MODEL_DIR)?;
    println!("\nsaved the model to {MODEL_DIR}/\n");

    println!("{:<48} {:>6} {:>9}", "test chart", "rating", "predicted");
    for (&row, predicted) in test.rows.iter().zip(&test.predictions).take(EXAMPLES) {
        println!("{:<48} {:>6} {:>9.2}", samples[row].name, samples[row].rating, predicted[0]);
    }
    let accuracy = test.metrics[0];
    println!(
        "\nover all {} test charts: typically {:.2} levels off (RMS), at most {:.2}",
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

/// Rates every dance-single chart of each simfile with the model in `MODEL_DIR`.
pub fn infer(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.is_empty() {
        return Err("infer needs a simfile or song folder".into());
    }
    let device = DevicePreference::default().device();
    let model = TrainedModel::load(MODEL_DIR, &device)
        .map_err(|e| format!("couldn't load the model in {MODEL_DIR} ({e}): has `train` been run?"))?;
    for path in args {
        let sim = SimFile::from_file(path).map_err(|e| format!("{path}: {e}"))?;
        println!("{} by {}", sim.title, sim.artist);
        println!("{:<10} {:>6} {:>9}", "chart", "rating", "predicted");
        for chart in analyze_singles(&sim) {
            let name = format!("{:?}", chart.difficulty);
            let features = match finite_features(&chart.stats) {
                Ok(features) => features,
                Err(feature) => {
                    println!("{name:<10} {:>6} {:>9}  (its {feature} isn't a finite number)", chart.rating, "-");
                    continue;
                }
            };
            // A mismatch would broadcast silently
            if features.len() != model.model.num_inputs() {
                return Err(format!(
                    "the model in {MODEL_DIR} takes {} statistics, but charts now have {}: train it again",
                    model.model.num_inputs(),
                    features.len()
                )
                .into());
            }
            let width = features.len();
            let x = Tensor::<2>::from_data(TensorData::new(features, [1, width]), &device);
            let predicted = model.predict(x).into_data().try_to_vec::<f32>().map_err(|e| format!("{e:?}"))?;
            println!("{name:<10} {:>6} {:>9.2}", chart.rating, predicted[0]);
        }
        println!();
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

/// The spreadsheet folder DDR A3 is in.
const SHEET_FOLDER: &str = "DDR A3 (AC) (INTL)";

/// Scores `COSTS` against a chart spreadsheet's counts for the charts of a folder of song folders
/// (see `cost_fit::score`), then again with each cost changed on its own.
pub fn scan_costs(args: &[String]) -> Result<(), Box<dyn Error>> {
    let charts = charts_in_sheet(args, "scan-costs")?;
    let threads = footing_threads();
    let evaluate = |costs: &Costs| cost_fit::score(&charts, &cost_fit::foot_all(&charts, costs, threads).counts);
    let baseline = print_scores("the costs as they are", &charts, &COSTS, threads);
    println!("{SCORE_KEY}\n");

    let scored = COLUMNS[..SCORED].iter().map(|name| format!("{name:>13}")).collect::<String>();
    println!("{:<26} {:>8} {:>7} {:>8} {scored}", "one cost changed", "to", "score", "change");
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
            println!("{name:<26} {changed:>8.3} {:>7.4} {:>+8.4} {errors}", score.score, score.score - baseline.score);
            if score.score < lowest.0 {
                lowest = (score.score, changed);
            }
            highest = highest.max(score.score);
        }
        ranges.push((name, value, lowest, highest));
    }

    ranges.sort_by(|a, b| a.2.0.total_cmp(&b.2.0));
    println!("\n{:<26} {:>8} {:>8} {:>7} {:>8} {:>8}", "cost, best first", "now", "best at", "score", "change", "worst");
    for (name, value, (lowest, at), highest) in ranges {
        println!("{name:<26} {value:>8.3} {at:>8.3} {lowest:>7.4} {:>+8.4} {:>+8.4}", lowest - baseline.score, highest - baseline.score);
    }
    Ok(())
}

/// Scores `COSTS` against a chart spreadsheet's counts (see `cost_fit::score`): on every chart,
/// then on the charts `fit_costs` searches on and on those it holds out.
pub fn score_costs(args: &[String]) -> Result<(), Box<dyn Error>> {
    let charts = charts_in_sheet(args, "score-costs")?;
    let threads = footing_threads();
    print_scores("the costs as they are", &charts, &COSTS, threads);
    println!("{SCORE_KEY}");
    let (held_out, searched): (Vec<cost_fit::Chart>, Vec<cost_fit::Chart>) = charts.into_iter().partition(is_held_out);
    let score = |charts: &[cost_fit::Chart]| cost_fit::score(charts, &cost_fit::foot_all(charts, &COSTS, threads).counts).score;
    println!("\nscore on the {} charts fit-costs searches on: {:.4}", searched.len(), score(&searched));
    println!("score on the {} charts it holds out: {:.4}", held_out.len(), score(&held_out));
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
    println!("searching on {} charts, holding out {} (every {HOLD_OUT_EVERY}th song)", training.len(), held_out.len());

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
    println!("{:>11} {:>7} {:>7} {:>6}", "evaluations", "best", "latest", "step");
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
            println!("{:>11} {:>7.4} {:>7.4} {:>6.3}", generation.evaluations, generation.best, generation.generation_best, generation.step)
        },
    );
    let (best, _) = costs_at(&found.point);
    println!("\n{} evaluations in {:.0} minutes\n", found.evaluations, started.elapsed().as_secs_f64() / 60.0);

    for (charts, which) in [(&training, "searched on"), (&held_out, "held out")] {
        let before = print_scores(&format!("the costs as they are, on the charts {which}"), charts, &COSTS, threads);
        let after = print_scores(&format!("the fitted costs, on the charts {which}"), charts, &best, threads);
        println!("score {:.4} to {:.4}\n", before.score, after.score);
    }
    println!("{SCORE_KEY}\n");

    println!("{:<26} {:>8} {:>8}", "cost", "now", "fitted");
    let mut best = best;
    for &(index, ..) in &fitted {
        println!("{:<26} {:>8.3} {:>8.3}", now.parameters()[index].0, *now.parameters()[index].1, *best.parameters()[index].1);
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
    println!("{title} ({} charts, footed in {:.1} s on {threads} threads):", charts.len(), start.elapsed().as_secs_f64());
    println!("{:<13} {:>7} {:>7} {:>7} {:>7} {:>12}", "", "all", "ours", "sheet", "error", "correlation");
    for (column, name) in COLUMNS.iter().enumerate() {
        let sheet: Vec<f64> = charts.iter().map(|chart| chart.sheet[column]).collect();
        let ours: Vec<f64> = footed.counts.iter().map(|counts| counts[column]).collect();
        println!(
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
    println!("score {:.4}{dropped}", score.score);
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
    for (song_index, song) in sorted_paths(songs)?.into_iter().filter(|path| path.is_dir()).enumerate() {
        for file in sorted_paths(&song)?.into_iter().filter(|path| path.extension().is_some_and(|ext| ext == "sm")) {
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
    println!("{} charts from {} are in the sheet ({missing} aren't)", charts.len(), songs.display());
    Ok(charts)
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
        println!("{} by {}", sim.title, sim.artist);
        println!("{:<10} {:>6} {:>6} {:>9}  footing; patterns", "chart", "rating", "notes", "segments");
        for chart in singles {
            let footed = foot_chart(chart, &sim.bpms, &sim.stops, &COSTS);
            let summary = Summary::of(&footed);
            println!(
                "{:<10} {:>6} {:>6} {:>9}  {}; {}",
                name(chart),
                chart.rating,
                summary.notes,
                footed.len(),
                summary.footing(),
                pattern_counts(&summary.patterns)
            );
        }
        println!("\nadd a difficulty (e.g. hard) for one chart's footing");
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
    events: Events,
    patterns: Patterns,
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
            events: Events::default(),
            patterns: Patterns::default(),
            cost: 0.0,
            nodes: 0,
            merged: 0,
            widest: 0,
            exact: true,
        };
        for segment in segments {
            summary.notes += segment.notes.len();
            summary.events += segment.footing.events;
            summary.patterns += segment.footing.patterns;
            summary.cost += segment.footing.cost;
            summary.nodes += segment.footing.nodes;
            summary.merged += segment.footing.merged;
            summary.widest = summary.widest.max(segment.footing.widest);
            summary.exact &= segment.footing.exact;
        }
        summary
    }

    /// "1 jack, 2 horizontal crossovers (cost 11.0)", or "nothing awkward". The cost depends on
    /// speed.
    fn footing(&self) -> String {
        let counted: Vec<String> = event_counts(&self.events)
            .into_iter()
            .filter(|&(count, ..)| count > 0)
            .map(|(count, one, many)| format!("{count} {}", if count == 1 { one } else { many }))
            .collect();
        if counted.is_empty() { "nothing awkward".to_string() } else { format!("{} (cost {:.1})", counted.join(", "), self.cost) }
    }

    fn patterns(&self) -> String {
        format!("patterns: {}", pattern_counts(&self.patterns))
    }

    fn search(&self) -> String {
        let dropped = if self.exact { "" } else { ", some dropped: this may not be the cheapest footing" };
        format!(
            "search: {} branches tried, {} merged, at most {} kept after a note{dropped}",
            self.nodes, self.merged, self.widest
        )
    }
}

/// "2 staircases, 1 drill (7 notes), 3 gallops", or "none".
fn pattern_counts(patterns: &Patterns) -> String {
    let drills = match patterns.drills {
        1 => format!("1 drill ({} notes)", patterns.drill_notes),
        drills => format!("{drills} drills ({} notes)", patterns.drill_notes),
    };
    let counted: Vec<String> = [
        (patterns.staircases, "1 staircase".to_string(), format!("{} staircases", patterns.staircases)),
        (patterns.drills, drills.clone(), drills),
        (patterns.gallops, "1 gallop".to_string(), format!("{} gallops", patterns.gallops)),
        (patterns.triples, "1 triple".to_string(), format!("{} triples", patterns.triples)),
    ]
    .into_iter()
    .filter(|&(count, ..)| count > 0)
    .map(|(count, one, many)| if count == 1 { one } else { many })
    .collect();
    if counted.is_empty() { "none".to_string() } else { counted.join(", ") }
}

/// The patterns starting at one note: "staircase", "drill of 7", "gallop", "triple".
fn pattern_labels(patterns: &Patterns) -> Vec<String> {
    let mut labels = Vec::new();
    labels.extend((0..patterns.staircases).map(|_| "staircase".to_string()));
    if patterns.drills > 0 {
        labels.push(format!("drill of {}", patterns.drill_notes));
    }
    labels.extend((0..patterns.gallops).map(|_| "gallop".to_string()));
    labels.extend((0..patterns.triples).map(|_| "triple".to_string()));
    labels
}

/// (count, singular, plural) of each kind of event.
fn event_counts(events: &Events) -> [(usize, &'static str, &'static str); 11] {
    [
        (events.jacks, "jack", "jacks"),
        (events.vertical_crossovers, "vertical crossover", "vertical crossovers"),
        (events.vertical_footswitches, "vertical footswitch", "vertical footswitches"),
        (events.sideswitches, "sideswitch", "sideswitches"),
        (events.footing.horizontal_crossovers, "horizontal crossover", "horizontal crossovers"),
        (events.brackets, "bracket", "brackets"),
        (events.doublesteps, "doublestep", "doublesteps"),
        (events.footing.laterals, "lateral", "laterals"),
        (events.footing.half_spins, "half spin", "half spins"),
        (events.freeze_switches, "freeze switch", "freeze switches"),
        (events.footing.full_spins, "full spin", "full spins"),
    ]
}

/// A footing as a table: a row per note, with `L` or `R` under each arrow for the foot that steps
/// it (the same letter twice for a bracket, `x` for an arrow no foot can reach), `|` under an arrow
/// still held from earlier in the segment (`l` or `r` where that foot takes the hold over in a
/// freeze switch), then what the step is and the patterns that start there. Rows are labelled by
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
        let mut events: Vec<String> = event_counts(&segment.footing.step_events[i])
            .into_iter()
            .filter(|&(count, ..)| count > 0)
            .map(|(count, one, many)| if count == 1 { one.to_string() } else { format!("{count} {many}") })
            .collect();
        events.extend(pattern_labels(&segment.footing.step_patterns[i]));
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
fn cached_samples(songs: &Path) -> Result<(Vec<&'static str>, Vec<Sample>), Box<dyn Error>> {
    let mut cache = StatsCache::load(CACHE_FILE)?;
    let (feature_names, samples) = load_samples(songs, &mut cache)?;
    if samples.is_empty() {
        return Err(format!("no dance-single charts found in {}", songs.display()).into());
    }
    if let Some(dir) = Path::new(CACHE_FILE).parent() {
        fs::create_dir_all(dir)?;
    }
    cache.save(CACHE_FILE)?;
    println!(
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
fn finite_features(stats: &ChartStatistics) -> Result<Vec<f32>, &'static str> {
    match stats.features().iter().find(|(_, value)| !value.is_finite()) {
        Some(&(feature, _)) => Err(feature),
        None => Ok(stats.features().iter().map(|&(_, value)| value as f32).collect()),
    }
}

/// Every dance-single chart in the `.sm` files of the song folders in `songs`. Files that don't
/// load, and charts with a non-finite statistic, are skipped with a warning.
///
/// Read in sorted order: listing order differs between machines, and would change the seeded
/// split.
fn load_samples(songs: &Path, cache: &mut StatsCache) -> Result<(Vec<&'static str>, Vec<Sample>), Box<dyn Error>> {
    let mut feature_names = Vec::new();
    let mut samples = Vec::new();
    for song in sorted_paths(songs)?.into_iter().filter(|path| path.is_dir()) {
        let song_name = song.file_name().unwrap_or_default().to_string_lossy().into_owned();
        for file in sorted_paths(&song)?.into_iter().filter(|path| path.extension().is_some_and(|ext| ext == "sm")) {
            let charts = match cache.analyze_file(&file) {
                Ok(charts) => charts,
                Err(e) => {
                    eprintln!("skipping {}: {e}", file.display());
                    continue;
                }
            };
            for chart in charts {
                let name = format!("{song_name} [{:?}]", chart.difficulty);
                feature_names = chart.stats.features().iter().map(|&(feature, _)| feature).collect();
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
fn report_pca(feature_names: &[&str], samples: &[Sample]) -> Result<(), Box<dyn Error>> {
    let (train_rows, val_rows, test_rows) = train::split_rows(samples.len(), &TrainingConfig::new().with_split_seed(SPLIT_SEED));
    let features = |rows: &[usize]| -> Vec<Vec<f32>> { rows.iter().map(|&row| samples[row].features.clone()).collect() };
    let train = features(&train_rows);
    let held_out = features(&[val_rows, test_rows].concat());

    let pca = Pca::fit(feature_names, &train);
    pca.save(PCA_FILE)?;
    let num_components = pca.components.len();
    println!(
        "\nPCA of {num_components} standardized features, fitted on the {} training charts (split seed {SPLIT_SEED}), saved to {PCA_FILE}",
        train.len()
    );
    println!("held-out: the {} validation and test charts, standardized with the training statistics\n", held_out.len());

    let train_scores = pca.transform(&train, num_components);
    let ratings: Vec<f64> = train_rows.iter().map(|&row| samples[row].rating as f64).collect();
    // Sum of squares along each component, over the held-out charts
    let held_out_scores = pca.transform(&held_out, num_components);
    let held_out_squares: Vec<f64> =
        (0..num_components).map(|k| held_out_scores.iter().map(|row| (row[k] as f64).powi(2)).sum()).collect();
    let held_out_total: f64 = held_out_squares.iter().sum();

    println!("{:>2} {:>8} {:>7} {:>7} {:>9} {:>7}  largest loadings", "pc", "variance", "share", "total", "held-out", "rating");
    let (mut total, mut held_out_so_far) = (0.0, 0.0);
    for (k, share) in pca.explained_variance_ratio().into_iter().enumerate() {
        total += share;
        held_out_so_far += held_out_squares[k] / held_out_total;
        let scores: Vec<f64> = train_scores.iter().map(|row| row[k] as f64).collect();
        let mut loadings: Vec<(f64, &str)> = pca.components[k].iter().copied().zip(feature_names.iter().copied()).collect();
        loadings.sort_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
        let largest: Vec<String> = loadings.iter().take(4).map(|(loading, name)| format!("{loading:+.2} {name}")).collect();
        println!(
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
    println!("\nshare, total: of the training charts' variance. held-out: running total over the held-out charts.");
    println!("rating: correlation between the component and the official rating, over the training charts.");
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
    fn draws_a_footing() {
        let crossover = foot_arrows("LDR", &COSTS, 1.0).unwrap();
        let numbered =
            "  #   ←  ↓  ↑  →\n  1   L  .  .  .   triple\n  2   .  R  .  .\n  3   .  .  .  L   horizontal crossover\n";
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
        assert_eq!(rows[3], "  3   .  r  L  .   freeze switch");

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
        assert_eq!(summary.footing(), format!("1 vertical footswitch, 1 horizontal crossover (cost {cost:.1})"));
        // LDR and LR
        assert_eq!(summary.patterns(), "patterns: 1 gallop, 1 triple");
        assert_eq!(Summary::of(&segments[1..2]).patterns(), "patterns: none");
        assert_eq!(Summary::of(&segments[2..]).footing(), "nothing awkward");
    }
}
