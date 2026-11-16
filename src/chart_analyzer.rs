use crate::sim_loader::{SingleChart, StepChart};

/// Difficulty statistics for a given chart. Used to train a model to predict chart rating.
struct ChartStatistics {
    // General stuff
    /// Chart length in seconds
    chart_len: f64,
    /// Percentage of chart duration with notes
    chart_active_ratio: f64,
    
    // Specials
    num_jumps: usize,
    num_holds: usize,
    num_mines: usize,
    num_rolls: usize,

    // Notes per second
    mean_notes_per_second: f64,
    median_notes_per_second: f64,
    max_notes_per_second: f64,

    // Jacks
    num_jacks: usize,
    mean_jack_interval: f64,
    median_jack_interval: f64,
    min_jack_interval: f64,

    // Streams
    num_streams: usize,
    mean_stream_len: f64,
    median_stream_len: f64,
    max_stream_len: f64,

    // Crossovers
    num_crossovers: usize,
    num_laterals: usize,
    num_half_spins: usize,
    num_full_spins: usize,
}

impl StepChart {
    pub fn analyze(&self) -> ChartStatistics {
        match self {
            StepChart::DanceSingle(single_chart) => analyze_single_chart(&single_chart),
            StepChart::DanceDouble(double_chart) => analyze_double_chart(),
        }
    }
}

fn analyze_single_chart(chart: &SingleChart) -> ChartStatistics {
    todo!()
}

/// Not yet implemented.
#[allow(dead_code)]
fn analyze_double_chart() -> ChartStatistics {
    todo!()
}