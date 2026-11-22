mod dataset;
mod device;
mod model;
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
use stats_cache::StatsCache;
use train::TrainingConfig;

const CACHE_FILE: &str = "artifacts/stats_cache.json";
const MODEL_DIR: &str = "artifacts/a3_model2/";
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
fn main() -> Result<(), Box<dyn Error>> {
    let songs = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(default_songs_folder);

    let mut cache = StatsCache::load(CACHE_FILE)?;
    let samples = load_samples(&songs, &mut cache)?;
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

    // Small batches: with only a few hundred charts, they make more steps per epoch.
    let inputs: Vec<Vec<f32>> = samples.iter().map(|sample| sample.features.clone()).collect();
    let ratings: Vec<Vec<f32>> = samples.iter().map(|sample| vec![sample.rating]).collect();
    let dataset = Dataset::from_rows(&inputs, &ratings, &DevicePreference::default().device());
    let model_config = ModelConfig::funnel(inputs[0].len(), 2, 2048, 1).with_dropout(0.5);
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
fn load_samples(songs: &Path, cache: &mut StatsCache) -> Result<Vec<Sample>, Box<dyn Error>> {
    let sorted = |dir: &Path| -> std::io::Result<Vec<PathBuf>> {
        let mut paths: Vec<PathBuf> = fs::read_dir(dir)?.map(|entry| entry.map(|entry| entry.path())).collect::<Result<_, _>>()?;
        paths.sort();
        Ok(paths)
    };

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
                let features: Vec<f32> = chart.stats.features().iter().map(|&(_, value)| value as f32).collect();
                if let Some((feature, _)) = chart.stats.features().iter().find(|(_, value)| !value.is_finite()) {
                    eprintln!("skipping {name}: its {feature} isn't a finite number");
                    continue;
                }
                samples.push(Sample { name, features, rating: chart.rating as f32 });
            }
        }
    }
    Ok(samples)
}
