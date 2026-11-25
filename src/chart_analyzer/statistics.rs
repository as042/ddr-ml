//! A chart's difficulty statistics: the model's inputs.

use serde::{Deserialize, Serialize};

use super::technique::{Occurrence, Technique};

/// Difficulty statistics for a given chart. Used to train a model to predict chart rating.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartStatistics {
    // General stuff
    /// Seconds from the first object (note, mine or hold end) to the last
    pub(super) chart_dur: f64,
    /// Seconds from beat 0 to the first object. `#OFFSET` isn't counted.
    pub(super) chart_start: f64,
    /// Measures from the first non-empty one to the last
    pub(super) chart_num_measures: usize,
    /// Fraction of `chart_dur` spent holding, or between notes at most `Settings::active_gap_seconds` apart
    pub(super) chart_active_ratio: f64,

    // Basic counts
    /// Rows with a tap, hold head or roll head (a jump is one note)
    pub(super) num_notes: usize,
    /// Hold heads
    pub(super) num_holds: usize,
    pub(super) num_mines: usize,
    /// Roll heads (not in `num_holds`)
    pub(super) num_rolls: usize,

    // Arrows per panel
    /// Arrows stepped on, per panel in the order of `ARROW_NAMES`. A jump is two.
    pub(super) num_arrows: [usize; 4],
    /// Most on a panel less the fewest, over all
    pub(super) lopsidedness_by_max: f64,
    /// How far each panel's are from the panels' mean, added up, over all
    pub(super) lopsidedness_by_mean: f64,
    // Shares of arrows. 0.5 with none to share.
    /// Right's, of Left's and Right's
    pub(super) right_bias: f64,
    /// Up's, of Down's and Up's
    pub(super) up_bias: f64,
    /// Down's and Up's, of all
    pub(super) vertical_bias: f64,

    // Holds and rolls, from each head to its end
    /// Notes that start any
    pub(super) num_hold_notes: usize,
    pub(super) total_hold_seconds: f64,
    pub(super) mean_hold_seconds: f64,
    pub(super) min_hold_seconds: f64,
    pub(super) max_hold_seconds: f64,

    // Notes per second: notes in the window starting at each note (see `counts::windows`)
    pub(super) mean_notes_per_second: f64,
    pub(super) median_notes_per_second: f64,
    pub(super) max_notes_per_second: f64,
    pub(super) mode_notes_per_second: f64,
    pub(super) std_notes_per_second: f64,
    /// Of the windows that end by the last note. 0 with none.
    pub(super) min_notes_per_second: f64,
    /// The most notes in a window of each of `SUSTAINED_SECONDS`, per second
    pub(super) sustained_notes_per_second: [f64; SUSTAINED_SECONDS.len()],
    // The longest stretch of notes with no break (see `Settings::active_gap_seconds`)
    pub(super) longest_run_notes: usize,
    pub(super) longest_run_seconds: f64,

    // Seconds from each note to the next
    pub(super) mean_note_gap: f64,
    pub(super) median_note_gap: f64,
    pub(super) mode_note_gap: f64,
    pub(super) max_note_gap: f64,
    pub(super) std_note_gap: f64,
    /// The fastest run of each of `BURST_NOTES` notes in a row, in notes per second: a note
    /// fewer than it has, over its seconds. 0 with too few notes.
    pub(super) burst_speeds: [f64; BURST_NOTES.len()],
    /// 1 over the least seconds from a jump to the next. 0 with fewer than two.
    pub(super) closest_jumps_speed: f64,

    // Rhythms and timings. A measure's subdivision is the finest its notes need (see `row_sub_div`).
    /// Distinct subdivisions among the measures
    pub(super) num_sub_divs: usize,
    /// Most common subdivision (e.g. 8 = 8th notes). Ties go to the smaller.
    pub(super) mode_sub_div: usize,
    /// Measures with a non-power-of-2 subdivision (12, 24, 48, ...)
    pub(super) num_adv_sub_divs: usize,
    // The same of each note: its own subdivision, whatever else is in its measure
    /// Notes on each of `NOTE_SUB_DIVS`
    pub(super) sub_div_notes: [usize; NOTE_SUB_DIVS.len()],
    /// Notes on any other subdivision
    pub(super) other_sub_div_notes: usize,
    /// Distinct subdivisions among the notes
    pub(super) num_note_sub_divs: usize,
    /// Most common. Ties go to the smaller.
    pub(super) mode_note_sub_div: usize,
    pub(super) finest_note_sub_div: usize,
    pub(super) mean_note_sub_div: f64,
    /// Share of the notes on a non-power-of-2 subdivision
    pub(super) adv_sub_div_note_ratio: f64,
    /// BPM with the most time between the first and last objects. Ties go to the smaller.
    pub(super) mode_bpm: f64,
    /// BPM changes strictly between the first and last objects, not counting changes to the
    /// same BPM
    pub(super) num_bpm_changes: usize,
    // Between the first and last objects
    pub(super) min_bpm: f64,
    pub(super) max_bpm: f64,
    /// `max_bpm - min_bpm`
    pub(super) bpm_range: f64,

    // Stops from the first object's beat to before the last's
    pub(super) num_stops: usize,
    pub(super) total_stop_seconds: f64,
    pub(super) min_stop_seconds: f64,
    pub(super) max_stop_seconds: f64,
    /// Fraction of `chart_dur` stopped
    pub(super) stop_ratio: f64,

    // DDR's groove radar, estimated (see `radar`)
    pub(super) radar_stream: f64,
    pub(super) radar_voltage: f64,
    pub(super) radar_air: f64,
    pub(super) radar_freeze: f64,
    pub(super) radar_chaos: f64,

    // Each foot's steps after its first in a segment, in the foot simulation (see
    // `footing::moves`)
    // Of 1 over the seconds since the foot's step before
    pub(super) mean_foot_speed: f64,
    pub(super) median_foot_speed: f64,
    pub(super) max_foot_speed: f64,
    /// How far the feet move, added up
    pub(super) foot_travel: f64,
    // Of each move's distance over its seconds
    pub(super) mean_foot_velocity: f64,
    pub(super) median_foot_velocity: f64,
    pub(super) max_foot_velocity: f64,

    // Foot simulation's penalty per note: for what each note's step adds (see `footing::Costs`)
    pub(super) mean_cost_per_note: f64,
    pub(super) median_cost_per_note: f64,
    pub(super) max_cost_per_note: f64,
    // The same, added up over the window starting at each note (see `windows`)
    pub(super) mean_cost_per_second: f64,
    pub(super) median_cost_per_second: f64,
    pub(super) max_cost_per_second: f64,

    /// One per `Technique::ALL`, in its order
    pub(super) techniques: [TechniqueStats; Technique::ALL.len()],
}

/// How often a technique is found in a chart, and how fast. All 0 if it never is.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TechniqueStats {
    pub count: usize,
    // Of each one's speed, in notes per second (see `Occurrence`)
    pub mean_speed: f64,
    pub median_speed: f64,
    pub max_speed: f64,
    /// How many are at least as fast as each of `FAST_SPEEDS`
    pub faster: [usize; FAST_SPEEDS.len()],
    /// Notes they span, added up
    pub notes: usize,
    // Of the notes each one spans
    pub mean_length: f64,
    pub median_length: f64,
    pub max_length: f64,
}

/// The panels, in the order of a row.
const ARROW_NAMES: [&str; 4] = ["left", "down", "up", "right"];

/// The subdivisions StepMania colors notes by, short of 192nds.
pub(super) const NOTE_SUB_DIVS: [usize; 8] = [4, 8, 12, 16, 24, 32, 48, 64];

/// Notes in a row that `ChartStatistics::burst_speeds` times.
pub(super) const BURST_NOTES: [usize; 5] = [2, 3, 5, 7, 15];

/// Window lengths of `ChartStatistics::sustained_notes_per_second`.
pub(super) const SUSTAINED_SECONDS: [usize; 3] = [5, 10, 30];

/// Notes per second that `TechniqueStats::faster` counts from.
pub(super) const FAST_SPEEDS: [usize; 4] = [2, 4, 6, 10];

impl ChartStatistics {
    pub fn technique(&self, technique: Technique) -> &TechniqueStats {
        &self.techniques[technique.index()]
    }

    /// (name, value) pairs: the model's input features. A technique's lengths are only among
    /// them if it varies in length. The densities (`_per_note`, `_per_second`) are counts over
    /// `num_notes` and `chart_dur`, 0 where those are.
    pub fn features(&self) -> Vec<(String, f64)> {
        let general = [
            ("chart_dur", self.chart_dur),
            ("chart_start", self.chart_start),
            ("chart_num_measures", self.chart_num_measures as f64),
            ("chart_active_ratio", self.chart_active_ratio),
            ("num_notes", self.num_notes as f64),
            ("num_holds", self.num_holds as f64),
            ("num_mines", self.num_mines as f64),
            ("num_rolls", self.num_rolls as f64),
            ("lopsidedness_by_max", self.lopsidedness_by_max),
            ("lopsidedness_by_mean", self.lopsidedness_by_mean),
            ("right_bias", self.right_bias),
            ("up_bias", self.up_bias),
            ("vertical_bias", self.vertical_bias),
            ("num_hold_notes", self.num_hold_notes as f64),
            ("total_hold_seconds", self.total_hold_seconds),
            ("mean_hold_seconds", self.mean_hold_seconds),
            ("min_hold_seconds", self.min_hold_seconds),
            ("max_hold_seconds", self.max_hold_seconds),
            ("mean_notes_per_second", self.mean_notes_per_second),
            ("median_notes_per_second", self.median_notes_per_second),
            ("max_notes_per_second", self.max_notes_per_second),
            ("mode_notes_per_second", self.mode_notes_per_second),
            ("std_notes_per_second", self.std_notes_per_second),
            ("min_notes_per_second", self.min_notes_per_second),
            ("longest_run_notes", self.longest_run_notes as f64),
            ("longest_run_seconds", self.longest_run_seconds),
            ("mean_note_gap", self.mean_note_gap),
            ("median_note_gap", self.median_note_gap),
            ("mode_note_gap", self.mode_note_gap),
            ("max_note_gap", self.max_note_gap),
            ("std_note_gap", self.std_note_gap),
            ("closest_jumps_speed", self.closest_jumps_speed),
            ("num_sub_divs", self.num_sub_divs as f64),
            ("mode_sub_div", self.mode_sub_div as f64),
            ("num_adv_sub_divs", self.num_adv_sub_divs as f64),
            ("other_sub_div_notes", self.other_sub_div_notes as f64),
            ("num_note_sub_divs", self.num_note_sub_divs as f64),
            ("mode_note_sub_div", self.mode_note_sub_div as f64),
            ("finest_note_sub_div", self.finest_note_sub_div as f64),
            ("mean_note_sub_div", self.mean_note_sub_div),
            ("adv_sub_div_note_ratio", self.adv_sub_div_note_ratio),
            ("mode_bpm", self.mode_bpm),
            ("num_bpm_changes", self.num_bpm_changes as f64),
            ("min_bpm", self.min_bpm),
            ("max_bpm", self.max_bpm),
            ("bpm_range", self.bpm_range),
            ("num_stops", self.num_stops as f64),
            ("total_stop_seconds", self.total_stop_seconds),
            ("min_stop_seconds", self.min_stop_seconds),
            ("max_stop_seconds", self.max_stop_seconds),
            ("stop_ratio", self.stop_ratio),
            ("radar_stream", self.radar_stream),
            ("radar_voltage", self.radar_voltage),
            ("radar_air", self.radar_air),
            ("radar_freeze", self.radar_freeze),
            ("radar_chaos", self.radar_chaos),
            ("mean_foot_speed", self.mean_foot_speed),
            ("median_foot_speed", self.median_foot_speed),
            ("max_foot_speed", self.max_foot_speed),
            ("foot_travel", self.foot_travel),
            ("mean_foot_velocity", self.mean_foot_velocity),
            ("median_foot_velocity", self.median_foot_velocity),
            ("max_foot_velocity", self.max_foot_velocity),
            ("mean_cost_per_note", self.mean_cost_per_note),
            ("median_cost_per_note", self.median_cost_per_note),
            ("max_cost_per_note", self.max_cost_per_note),
            ("mean_cost_per_second", self.mean_cost_per_second),
            ("median_cost_per_second", self.median_cost_per_second),
            ("max_cost_per_second", self.max_cost_per_second),
        ];
        let mut features: Vec<(String, f64)> = general.into_iter().map(|(name, value)| (name.to_string(), value)).collect();
        for (name, count) in ARROW_NAMES.iter().zip(self.num_arrows) {
            features.push((format!("num_{name}_arrows"), count as f64));
        }
        for (sub_div, count) in NOTE_SUB_DIVS.iter().zip(self.sub_div_notes) {
            features.push((format!("sub_div_{sub_div}_notes"), count as f64));
        }
        for (notes, speed) in BURST_NOTES.iter().zip(self.burst_speeds) {
            features.push((format!("burst_{notes}_speed"), speed));
        }
        for (seconds, per_second) in SUSTAINED_SECONDS.iter().zip(self.sustained_notes_per_second) {
            features.push((format!("max_notes_per_second_over_{seconds}s"), per_second));
        }

        let over = |count: usize, whole: f64| if whole > 0.0 { count as f64 / whole } else { 0.0 };
        let (notes, seconds) = (self.num_notes as f64, self.chart_dur);
        features.push(("num_notes_per_second".to_string(), over(self.num_notes, seconds)));
        features.push(("num_holds_per_note".to_string(), over(self.num_holds, notes)));
        features.push(("num_holds_per_second".to_string(), over(self.num_holds, seconds)));
        features.push(("num_mines_per_second".to_string(), over(self.num_mines, seconds)));
        features.push(("num_bpm_changes_per_second".to_string(), over(self.num_bpm_changes, seconds)));
        features.push(("num_stops_per_second".to_string(), over(self.num_stops, seconds)));
        features.push(("foot_travel_per_note".to_string(), if notes > 0.0 { self.foot_travel / notes } else { 0.0 }));
        features.push(("foot_travel_per_second".to_string(), if seconds > 0.0 { self.foot_travel / seconds } else { 0.0 }));
        for technique in Technique::ALL {
            let stats = self.technique(technique);
            let key = technique.key();
            features.push((format!("{key}_count"), stats.count as f64));
            features.push((format!("{key}_per_note"), over(stats.count, notes)));
            features.push((format!("{key}_per_second"), over(stats.count, seconds)));
            features.push((format!("{key}_mean_speed"), stats.mean_speed));
            features.push((format!("{key}_median_speed"), stats.median_speed));
            features.push((format!("{key}_max_speed"), stats.max_speed));
            for (speed, count) in FAST_SPEEDS.iter().zip(stats.faster) {
                features.push((format!("{key}_count_over_{speed}"), count as f64));
            }
            if technique.varies_in_length() {
                features.push((format!("{key}_notes"), stats.notes as f64));
                features.push((format!("{key}_mean_length"), stats.mean_length));
                features.push((format!("{key}_median_length"), stats.median_length));
                features.push((format!("{key}_max_length"), stats.max_length));
            }
        }
        features
    }
}

/// A speed this far under one of `FAST_SPEEDS` is at it: 8th notes at 120 BPM are 4 a second,
/// give or take rounding.
const FAST_TOLERANCE: f64 = 1e-6;

/// Each technique's statistics over a chart's `occurrences`.
pub(super) fn summarize(occurrences: &[Occurrence]) -> [TechniqueStats; Technique::ALL.len()] {
    Technique::ALL.map(|technique| {
        let found: Vec<&Occurrence> = occurrences.iter().filter(|occurrence| occurrence.technique == technique).collect();
        let (mean_speed, median_speed, max_speed) = mean_median_max(found.iter().map(|occurrence| occurrence.speed).collect());
        let (mean_length, median_length, max_length) =
            mean_median_max(found.iter().map(|occurrence| occurrence.notes as f64).collect());
        TechniqueStats {
            count: found.len(),
            mean_speed,
            median_speed,
            max_speed,
            faster: FAST_SPEEDS.map(|speed| found.iter().filter(|occurrence| occurrence.speed >= speed as f64 - FAST_TOLERANCE).count()),
            notes: found.iter().map(|occurrence| occurrence.notes).sum(),
            mean_length,
            median_length,
            max_length,
        }
    })
}

/// How some values are spread. All 0 if there are none.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(super) struct Spread {
    pub total: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    /// The most common value: the mean of the most values that round to the same
    /// `1 / MODE_STEPS`. Ties go to the smallest.
    pub mode: f64,
    /// Of the values as a whole population
    pub std: f64,
}

const MODE_STEPS: f64 = 1000.0;

impl Spread {
    pub(super) fn of(mut values: Vec<f64>) -> Spread {
        if values.is_empty() {
            return Spread::default();
        }
        values.sort_by(f64::total_cmp);
        let n = values.len();
        let total: f64 = values.iter().sum();
        let mean = total / n as f64;
        let median = if n % 2 == 1 { values[n / 2] } else { (values[n / 2 - 1] + values[n / 2]) / 2.0 };
        let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / n as f64;

        // Sorted, so equal steps are in a row and the first of the longest rows is the smallest
        let step = |value: f64| (value * MODE_STEPS).round();
        // (step, values, their sum)
        let (mut most, mut row) = ((f64::NAN, 0, 0.0), (f64::NAN, 0, 0.0));
        for &value in &values {
            row = if step(value) == row.0 { (row.0, row.1 + 1, row.2 + value) } else { (step(value), 1, value) };
            if row.1 > most.1 {
                most = row;
            }
        }
        Spread { total, min: values[0], max: values[n - 1], mean, median, mode: most.2 / most.1 as f64, std: variance.sqrt() }
    }
}

/// All 0 if `values` is empty.
pub(super) fn mean_median_max(values: Vec<f64>) -> (f64, f64, f64) {
    let spread = Spread::of(values);
    (spread.mean, spread.median, spread.max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_mean_median_and_max() {
        assert_eq!(mean_median_max(vec![]), (0.0, 0.0, 0.0));
        assert_eq!(mean_median_max(vec![5.0]), (5.0, 5.0, 5.0));
        assert_eq!(mean_median_max(vec![9.0, 1.0, 2.0]), (4.0, 2.0, 9.0));
        // Even count: the median is the mean of the middle two
        assert_eq!(mean_median_max(vec![4.0, 1.0, 3.0, 10.0]), (4.5, 3.5, 10.0));
    }

    #[test]
    fn spreads_values() {
        assert_eq!(Spread::of(vec![]), Spread::default());
        let spread = Spread::of(vec![4.0, 1.0, 3.0, 10.0, 1.0, 5.0]);
        assert_eq!((spread.total, spread.min, spread.max, spread.mean, spread.median), (24.0, 1.0, 10.0, 4.0, 3.5));
        assert_eq!(spread.mode, 1.0);
        // Squares of the distances from the mean: 0, 9, 1, 36, 9, 1
        assert!((spread.std - (56.0_f64 / 6.0).sqrt()).abs() < 1e-12);
        // Ties: the smallest
        assert_eq!(Spread::of(vec![3.0, 2.0, 3.0, 2.0, 7.0]).mode, 2.0);
        // Values a rounding error apart are one value, and the mode is their mean
        let third = 1.0 / 3.0;
        let mode = Spread::of(vec![third, 0.5, third + 1e-9, 0.25, third - 1e-9]).mode;
        assert!((mode - third).abs() < 1e-12, "{mode}");
        assert_eq!(Spread::of(vec![2.5]), Spread { total: 2.5, min: 2.5, max: 2.5, mean: 2.5, median: 2.5, mode: 2.5, std: 0.0 });
    }

    #[test]
    fn summarizes_each_technique() {
        let found = |technique, notes, speed| Occurrence { technique, note: 0, notes, speed };
        let occurrences = [
            found(Technique::Drill, 5, 8.0),
            found(Technique::Jack, 1, 2.0),
            found(Technique::Drill, 9, 4.0),
            found(Technique::Drill, 6, 6.0),
            found(Technique::Jack, 1, 0.0),
        ];
        let stats = summarize(&occurrences);
        let drills = TechniqueStats {
            count: 3,
            mean_speed: 6.0,
            median_speed: 6.0,
            max_speed: 8.0,
            // At least 2, 4, 6 and 10 a second
            faster: [3, 3, 2, 0],
            notes: 20,
            mean_length: 20.0 / 3.0,
            median_length: 6.0,
            max_length: 9.0,
        };
        assert_eq!(stats[Technique::Drill.index()], drills);
        let jacks = stats[Technique::Jack.index()];
        assert_eq!((jacks.count, jacks.mean_speed, jacks.median_speed, jacks.max_speed), (2, 1.0, 1.0, 2.0));
        assert_eq!(jacks.faster, [1, 0, 0, 0]);
        // A hair under counts
        let nearly = [found(Technique::Jack, 1, 4.0 - 1e-9), found(Technique::Jack, 1, 3.9)];
        assert_eq!(summarize(&nearly)[Technique::Jack.index()].faster, [2, 1, 0, 0]);
        assert_eq!(stats[Technique::Stream.index()], TechniqueStats::default());
    }

    #[test]
    fn names_every_feature_once() {
        // A different value in every field, to catch a feature reading the wrong one
        let fields = serde_json::to_value(ChartStatistics::default()).unwrap();
        let fields = fields.as_object().unwrap();
        let mut next = 0;
        let mut distinct = |field: &serde_json::Value| -> serde_json::Value {
            let mut number = || {
                next += 1;
                serde_json::json!(next)
            };
            match field {
                serde_json::Value::Array(items) => items.iter().map(|_| number()).collect(),
                _ => number(),
            }
        };
        let mut values: serde_json::Map<String, serde_json::Value> =
            fields.iter().filter(|(name, _)| *name != "techniques").map(|(name, field)| (name.clone(), distinct(field))).collect();
        let technique_fields = serde_json::to_value(TechniqueStats::default()).unwrap();
        let techniques: Vec<serde_json::Value> = Technique::ALL
            .iter()
            .map(|_| technique_fields.as_object().unwrap().iter().map(|(name, field)| (name.clone(), distinct(field))).collect())
            .collect();
        values.insert("techniques".to_string(), serde_json::Value::Array(techniques));
        let stats: ChartStatistics = serde_json::from_value(serde_json::Value::Object(values.clone())).unwrap();

        let features = stats.features();
        let mut names: Vec<&String> = features.iter().map(|(name, _)| name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), features.len(), "a feature name is used twice");
        let feature = |name: &str| features.iter().find(|(feature, _)| feature == name).map(|(_, value)| *value);

        // A field that is one number is a feature by its own name
        let single: Vec<&String> = values.iter().filter(|(_, value)| value.is_number()).map(|(name, _)| name).collect();
        for name in &single {
            assert_eq!(feature(name), values[*name].as_f64(), "feature {name}");
        }
        // One of a list is named for what it counts
        let listed = [
            ("num_arrows", "num_down_arrows", 1),
            ("sub_div_notes", "sub_div_12_notes", 2),
            ("burst_speeds", "burst_15_speed", 4),
            ("sustained_notes_per_second", "max_notes_per_second_over_10s", 1),
        ];
        for (field, name, index) in listed {
            assert_eq!(feature(name), values[field][index].as_f64(), "feature {name}");
        }
        for technique in Technique::ALL {
            let (key, fields) = (technique.key(), &values["techniques"][technique.index()]);
            for field in ["count", "mean_speed", "median_speed", "max_speed"] {
                assert_eq!(feature(&format!("{key}_{field}")), fields[field].as_f64(), "{key} {field}");
            }
            assert_eq!(feature(&format!("{key}_count_over_6")), fields["faster"][2].as_f64(), "{key} faster");
            for field in ["notes", "mean_length", "median_length", "max_length"] {
                let expected = if technique.varies_in_length() { fields[field].as_f64() } else { None };
                assert_eq!(feature(&format!("{key}_{field}")), expected, "{key} {field}");
            }
        }

        // Densities
        let (notes, seconds) = (stats.num_notes as f64, stats.chart_dur);
        let jumps = stats.technique(Technique::Jump).count as f64;
        assert_eq!(feature("jump_per_note"), Some(jumps / notes));
        assert_eq!(feature("jump_per_second"), Some(jumps / seconds));
        assert_eq!(feature("num_notes_per_second"), Some(notes / seconds));
        assert_eq!(feature("num_holds_per_note"), Some(stats.num_holds as f64 / notes));
        let empty = ChartStatistics::default().features();
        assert!(empty.iter().all(|(_, value)| *value == 0.0), "nothing in a chart: every feature 0");

        let varying = Technique::ALL.iter().filter(|technique| technique.varies_in_length()).count();
        assert_eq!(feature("foot_travel_per_note"), Some(stats.foot_travel / notes));
        let lists = ARROW_NAMES.len() + NOTE_SUB_DIVS.len() + BURST_NOTES.len() + SUSTAINED_SECONDS.len();
        let densities = 8 + 2 * Technique::ALL.len();
        let per_technique = 4 + FAST_SPEEDS.len();
        assert_eq!(features.len(), single.len() + lists + densities + per_technique * Technique::ALL.len() + 4 * varying);
    }
}
