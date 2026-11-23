mod dataset;
mod device;
mod model;
mod pca;
mod train;
mod sim_loader;
mod chart_analyzer;
mod stats_cache;

#[cfg(test)]
mod tests;

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use burn::optim::AdamWConfig;

use dataset::Dataset;
use device::DevicePreference;
use model::ModelConfig;
use pca::Pca;
use stats_cache::StatsCache;
use train::TrainingConfig;

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
/// chart in a folder of song folders (the first argument, or DDR A3), and saves it to
/// `MODEL_DIR`.
///
/// With `--pca`, fits a PCA of the chart statistics instead and saves it to `PCA_FILE`.
fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let pca_only = args.iter().any(|arg| arg == "--pca");
    args.retain(|arg| arg != "--pca");
    let songs = args.first().map(PathBuf::from).unwrap_or_else(default_songs_folder);

    let mut cache = StatsCache::load(CACHE_FILE)?;
    let (feature_names, samples) = load_samples(&songs, &mut cache)?;
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
    if pca_only {
        return report_pca(&feature_names, &samples);
    }

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

/// DDR A3's songs in this machine's ITGmania install.
fn default_songs_folder() -> PathBuf {
    let home = std::env::var_os("HOME").unwrap_or_default();
    Path::new(&home).join("Documents/ITGmania-0.9.0-Linux/itgmania/Songs/A3")
}

/// Every dance-single chart in the `.sm` files of the song folders in `songs`. Files that don't
/// load, and charts with a non-finite statistic, are skipped with a warning.
///
/// Read in sorted order: listing order differs between machines, and would change the seeded
/// split.
fn load_samples(songs: &Path, cache: &mut StatsCache) -> Result<(Vec<&'static str>, Vec<Sample>), Box<dyn Error>> {
    let sorted = |dir: &Path| -> std::io::Result<Vec<PathBuf>> {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)?.map(|entry| entry.map(|entry| entry.path())).collect::<Result<_, _>>()?;
        paths.sort();
        Ok(paths)
    };

    let mut feature_names = Vec::new();
    let mut samples = Vec::new();
    for song in sorted(songs)?.into_iter().filter(|path| path.is_dir()) {
        let song_name = song.file_name().unwrap_or_default().to_string_lossy().into_owned();
        for file in sorted(&song)?.into_iter().filter(|path| path.extension().is_some_and(|ext| ext == "sm")) {
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
                let features: Vec<f32> = chart.stats.features().iter().map(|&(_, value)| value as f32).collect();
                if let Some((feature, _)) = chart.stats.features().iter().find(|(_, value)| !value.is_finite()) {
                    eprintln!("skipping {name}: its {feature} isn't a finite number");
                    continue;
                }
                samples.push(Sample { name, features, rating: chart.rating as f32 });
            }
        }
    }
    Ok((feature_names, samples))
}

/// Fits a PCA on the training split's features (the split `main` trains on), prints it and
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
