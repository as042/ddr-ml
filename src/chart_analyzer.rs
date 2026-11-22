mod footing;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::sim_loader::{BPMChange, ChartDifficulty, Measure, Note, SimFile, SingleChart, StepChart, Stop};
use footing::{Start, simulate_chart, wrong_foot_penalty};

/// Difficulty statistics for a given chart. Used to train a model to predict chart rating.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChartStatistics {
    // General stuff
    /// Seconds from the first object (note, mine or hold end) to the last
    chart_dur: f64,
    /// Measures from the first non-empty one to the last
    chart_num_measures: usize,
    /// Fraction of `chart_dur` spent holding, or between notes at most `ACTIVE_GAP_SECONDS` apart
    chart_active_ratio: f64,

    // Basic counts
    /// Rows with a tap, hold head or roll head (a jump is one note)
    num_notes: usize,
    /// Notes with 2+ arrows, hands included
    num_jumps: usize,
    /// Hold heads
    num_holds: usize,
    num_mines: usize,
    /// Roll heads (not in `num_holds`)
    num_rolls: usize,

    // Notes per second: notes in the window starting at each note (see `windows`)
    mean_notes_per_second: f64,
    median_notes_per_second: f64,
    max_notes_per_second: f64,

    // Jacks: notes sharing an arrow with the note before them in the same segment
    num_jacks: usize,
    // Jack speed: jacks in the window starting at each jack, within its segment
    mean_jacks_per_second: f64,
    median_jacks_per_second: f64,
    max_jacks_per_second: f64,

    // Streams: segments of at least `MIN_STREAM_NOTES` notes. Length in notes.
    num_streams: usize,
    mean_stream_len: f64,
    median_stream_len: f64,
    max_stream_len: f64,

    // Footing, from the foot simulation (see `footing::Footing`)
    num_crossovers: usize,
    num_laterals: usize,
    num_half_spins: usize,
    num_full_spins: usize,
    /// Extra crossovers, laterals and spins from starting each segment on the worse foot, per
    /// note. Uses `SEGMENT_BREAK_SECONDS` segments, not the footing ones.
    wrong_foot_penalty_per_note: f64,

    // Rhythms and timings. A measure's subdivision is the finest its notes need (see `row_sub_div`).
    /// Distinct subdivisions among the measures
    num_sub_divs: usize,
    /// Most common subdivision (e.g. 8 = 8th notes). Ties go to the smaller.
    mode_sub_div: usize,
    /// Measures with a non-power-of-2 subdivision (12, 24, 48, ...)
    num_adv_sub_divs: usize,
    /// BPM with the most time between the first and last objects. Ties go to the smaller.
    mode_bpm: f64,
    /// BPM changes strictly between the first and last objects, not counting changes to the
    /// same BPM
    num_bpm_changes: usize,
}

impl ChartStatistics {
    /// (name, value) pairs in field order: the model's input features.
    pub fn features(&self) -> [(&'static str, f64); 29] {
        [
            ("chart_dur", self.chart_dur),
            ("chart_num_measures", self.chart_num_measures as f64),
            ("chart_active_ratio", self.chart_active_ratio),
            ("num_notes", self.num_notes as f64),
            ("num_jumps", self.num_jumps as f64),
            ("num_holds", self.num_holds as f64),
            ("num_mines", self.num_mines as f64),
            ("num_rolls", self.num_rolls as f64),
            ("mean_notes_per_second", self.mean_notes_per_second),
            ("median_notes_per_second", self.median_notes_per_second),
            ("max_notes_per_second", self.max_notes_per_second),
            ("num_jacks", self.num_jacks as f64),
            ("mean_jacks_per_second", self.mean_jacks_per_second),
            ("median_jacks_per_second", self.median_jacks_per_second),
            ("max_jacks_per_second", self.max_jacks_per_second),
            ("num_streams", self.num_streams as f64),
            ("mean_stream_len", self.mean_stream_len),
            ("median_stream_len", self.median_stream_len),
            ("max_stream_len", self.max_stream_len),
            ("num_crossovers", self.num_crossovers as f64),
            ("num_laterals", self.num_laterals as f64),
            ("num_half_spins", self.num_half_spins as f64),
            ("num_full_spins", self.num_full_spins as f64),
            ("wrong_foot_penalty_per_note", self.wrong_foot_penalty_per_note),
            ("num_sub_divs", self.num_sub_divs as f64),
            ("mode_sub_div", self.mode_sub_div as f64),
            ("num_adv_sub_divs", self.num_adv_sub_divs as f64),
            ("mode_bpm", self.mode_bpm),
            ("num_bpm_changes", self.num_bpm_changes as f64),
        ]
    }
}

impl StepChart {
    /// `bpms` and `stops` are shared by all of a .sm file's charts.
    #[allow(dead_code)] // Not called yet
    pub fn analyze(&self, bpms: &[BPMChange], stops: &[Stop]) -> ChartStatistics {
        match self {
            StepChart::DanceSingle(single_chart) => analyze_single_chart(single_chart, bpms, stops),
            StepChart::DanceDouble(_double_chart) => analyze_double_chart(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalyzedChart {
    pub difficulty: ChartDifficulty,
    pub rating: f64,
    pub stats: ChartStatistics,
}

/// Dance-single charts only, in file order.
pub fn analyze_singles(sim: &SimFile) -> Vec<AnalyzedChart> {
    sim.step_charts
        .iter()
        .filter_map(|chart| match chart {
            StepChart::DanceSingle(single) => Some(AnalyzedChart {
                difficulty: single.difficulty,
                rating: single.rating,
                stats: analyze_single_chart(single, &sim.bpms, &sim.stops),
            }),
            StepChart::DanceDouble(_) => None,
        })
        .collect()
}

fn analyze_single_chart(chart: &SingleChart, bpms: &[BPMChange], stops: &[Stop]) -> ChartStatistics {
    let timing = Timing::new(bpms, stops);
    let mut stats = ChartStatistics::default();
    let span = count_notes_and_rhythms(&chart.measures, &mut stats);
    count_bpms(&timing, span, &mut stats);

    let notes = to_timed_notes(&chart.measures, &timing);
    count_notes_per_second(&notes, &mut stats);

    count_duration(&notes, &timing, span, &mut stats);

    let pattern_segments = segments(&notes, SEGMENT_BREAK_SECONDS);
    count_jacks(&pattern_segments, &mut stats);
    count_streams(&pattern_segments, &mut stats);
    count_footing(&segments(&notes, FOOTING_BREAK_SECONDS), &mut stats);
    count_wrong_foot_penalty(&pattern_segments, &mut stats);
    stats
}

/// Sliding window length (see `windows`).
const WINDOW_SECONDS: f64 = 1.0;

/// Longest gap between notes of one segment (see `segments`). 8ths at 120 BPM are exactly this
/// far apart.
const SEGMENT_BREAK_SECONDS: f64 = 0.25;

/// Minimum notes for a segment to be a stream.
const MIN_STREAM_NOTES: usize = 8;

/// Longer gaps between notes are breaks, not active time.
const ACTIVE_GAP_SECONDS: f64 = 1.0;

/// Segment break for the foot simulation. Longer than `SEGMENT_BREAK_SECONDS`: players keep
/// alternating feet through quarter notes.
const FOOTING_BREAK_SECONDS: f64 = 0.5;

/// Which foot starts each segment in the foot simulation.
const START: Start = Start::Easier;

/// Tolerance for rounding in beat-to-second conversion. Far below any real note gap (192nds at
/// 1000 BPM are 1.25 ms apart).
const TIME_EPSILON: f64 = 1e-6;

/// A song's BPM changes and stops.
struct Timing {
    /// (beat, BPM), sorted by beat
    bpms: Vec<(f64, f64)>,
    /// (beat, seconds), sorted by beat
    stops: Vec<(f64, f64)>,
}

impl Timing {
    fn new(bpms: &[BPMChange], stops: &[Stop]) -> Self {
        // Snap to StepMania's 48 rows per beat (33.333 means 33 1/3), which puts a change on the
        // same f64 as a note on that beat (see `row_beat`). The sorts are stable: of two changes
        // on one beat, the later in the file wins, as in StepMania.
        let snap = |beat: f64| (beat * 48.0).round() / 48.0;
        let mut bpms: Vec<(f64, f64)> = bpms.iter().map(|c| (snap(c.beat), c.bpm)).collect();
        bpms.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut stops: Vec<(f64, f64)> = stops.iter().map(|s| (snap(s.beat), s.seconds)).collect();
        stops.sort_by(|a, b| a.0.total_cmp(&b.0));
        Timing { bpms, stops }
    }

    /// (BPM, beats) stretches between beats `from` and `to`, in order. The first BPM also covers
    /// anything before its change.
    fn bpm_stretches(&self, from: f64, to: f64) -> impl Iterator<Item = (f64, f64)> {
        self.bpms.iter().enumerate().filter_map(move |(i, &(beat, bpm))| {
            let start = if i == 0 { from } else { beat.max(from) };
            let end = self.bpms.get(i + 1).map_or(to, |&(next, _)| next.min(to));
            (end > start).then_some((bpm, end - start))
        })
    }

    /// 0 if the song has no BPMs.
    fn bpm_at(&self, beat: f64) -> f64 {
        let at_or_before = self.bpms.iter().rev().find(|&&(change, _)| change <= beat);
        at_or_before.or(self.bpms.first()).map_or(0.0, |&(_, bpm)| bpm)
    }

    /// Seconds from beat 0 to `beat`, ignoring `#OFFSET` (a constant shift).
    fn time_at(&self, beat: f64) -> f64 {
        let moving: f64 = self.bpm_stretches(0.0, beat).map(|(bpm, beats)| beats * 60.0 / bpm).sum();
        // A stop only delays notes after its beat.
        let stopped: f64 = self.stops.iter().filter(|&&(stop, _)| stop < beat).map(|&(_, seconds)| seconds).sum();
        moving + stopped
    }
}

/// A note (see `ChartStatistics::num_notes`) and when it's hit.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TimedNote {
    /// Seconds from beat 0
    time: f64,
    /// The whole row, including mines and hold ends
    row: [Note; 4],
    /// Per arrow: when the hold or roll starting here ends
    hold_ends: [Option<f64>; 4],
}

/// The chart's notes in time order. Rows with only mines or hold ends aren't notes.
fn to_timed_notes(measures: &[Measure], timing: &Timing) -> Vec<TimedNote> {
    let mut notes: Vec<TimedNote> = Vec::new();
    // Per arrow: index of the note that started its open hold or roll
    let mut holding: [Option<usize>; 4] = [None; 4];
    let mut measure_start = 0;
    for measure in measures {
        for (r, row) in measure.rows.iter().enumerate() {
            if row.iter().all(|&note| note == Note::Empty) {
                continue;
            }
            let time = timing.time_at(row_beat(measure_start, measure, r));
            for (arrow, &note) in row.iter().enumerate() {
                if note == Note::End
                    && let Some(start) = holding[arrow].take()
                {
                    notes[start].hold_ends[arrow] = Some(time);
                }
            }
            if arrows(row) > 0 {
                for (arrow, &note) in row.iter().enumerate() {
                    if matches!(note, Note::HoldStart | Note::RollStart) {
                        holding[arrow] = Some(notes.len());
                    }
                }
                notes.push(TimedNote { time, row: *row, hold_ends: [None; 4] });
            }
        }
        measure_start += measure.num_beats;
    }
    notes
}

/// For each note, the notes from it up to (but not including) one window later. `notes` must be
/// in time order.
fn windows(notes: &[TimedNote]) -> impl Iterator<Item = &[TimedNote]> {
    let mut end = 0;
    (0..notes.len()).map(move |start| {
        let window_end = notes[start].time + WINDOW_SECONDS - TIME_EPSILON;
        end = end.max(start);
        while end < notes.len() && notes[end].time < window_end {
            end += 1;
        }
        &notes[start..end]
    })
}

fn count_notes_per_second(notes: &[TimedNote], stats: &mut ChartStatistics) {
    let per_second = windows(notes).map(|window| window.len() as f64 / WINDOW_SECONDS).collect();
    (stats.mean_notes_per_second, stats.median_notes_per_second, stats.max_notes_per_second) =
        mean_median_max(per_second);
}

/// `span`: beats of the first and last objects, from `count_notes_and_rhythms`.
fn count_duration(notes: &[TimedNote], timing: &Timing, span: Option<(f64, f64)>, stats: &mut ChartStatistics) {
    let Some((first, last)) = span else { return };
    let (start, end) = (timing.time_at(first), timing.time_at(last));
    stats.chart_dur = end - start;
    if stats.chart_dur <= 0.0 {
        return;
    }

    // Active intervals: note pairs with no break between them, and holds
    let mut active: Vec<(f64, f64)> = notes
        .windows(2)
        .filter(|pair| pair[1].time - pair[0].time <= ACTIVE_GAP_SECONDS + TIME_EPSILON)
        .map(|pair| (pair[0].time, pair[1].time))
        .collect();
    for note in notes {
        active.extend(note.hold_ends.iter().flatten().map(|&hold_end| (note.time, hold_end)));
    }
    // Length of their union
    active.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut covered = 0.0;
    let mut covered_to = f64::NEG_INFINITY;
    for (from, to) in active {
        let from = from.max(covered_to);
        if to > from {
            covered += to - from;
            covered_to = to;
        }
    }
    stats.chart_active_ratio = covered / stats.chart_dur;
}

/// All 0 if `values` is empty.
fn mean_median_max(mut values: Vec<f64>) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mean = values.iter().sum::<f64>() / n as f64;
    let median = if n % 2 == 1 { values[n / 2] } else { (values[n / 2 - 1] + values[n / 2]) / 2.0 };
    (mean, median, values[n - 1])
}

/// Runs of notes (in time order) with no gap longer than `break_seconds`. A longer gap lets the
/// player reset their feet, so patterns are only looked for inside a segment.
fn segments(notes: &[TimedNote], break_seconds: f64) -> Vec<&[TimedNote]> {
    notes.chunk_by(|a, b| b.time - a.time <= break_seconds + TIME_EPSILON).collect()
}

fn count_jacks(segments: &[&[TimedNote]], stats: &mut ChartStatistics) {
    let mut per_second = Vec::new();
    for &segment in segments {
        let is_jack: Vec<bool> = (0..segment.len())
            .map(|i| i > 0 && shares_an_arrow(&segment[i - 1], &segment[i]))
            .collect();
        stats.num_jacks += is_jack.iter().filter(|&&jack| jack).count();

        // Jack speed. Windows stop at the segment's end.
        for (start, window) in windows(segment).enumerate() {
            if is_jack[start] {
                let jacks = is_jack[start..start + window.len()].iter().filter(|&&jack| jack).count();
                per_second.push(jacks as f64 / WINDOW_SECONDS);
            }
        }
    }
    (stats.mean_jacks_per_second, stats.median_jacks_per_second, stats.max_jacks_per_second) =
        mean_median_max(per_second);
}

fn count_streams(segments: &[&[TimedNote]], stats: &mut ChartStatistics) {
    let lengths: Vec<f64> = segments
        .iter()
        .map(|segment| segment.len())
        .filter(|&notes| notes >= MIN_STREAM_NOTES)
        .map(|notes| notes as f64)
        .collect();
    stats.num_streams = lengths.len();
    (stats.mean_stream_len, stats.median_stream_len, stats.max_stream_len) = mean_median_max(lengths);
}

fn count_footing(segments: &[&[TimedNote]], stats: &mut ChartStatistics) {
    let footing = simulate_chart(segments, START);
    stats.num_crossovers = footing.crossovers;
    stats.num_laterals = footing.laterals;
    stats.num_half_spins = footing.half_spins;
    stats.num_full_spins = footing.full_spins;
}

fn count_wrong_foot_penalty(segments: &[&[TimedNote]], stats: &mut ChartStatistics) {
    let notes: usize = segments.iter().map(|segment| segment.len()).sum();
    if notes > 0 {
        stats.wrong_foot_penalty_per_note = wrong_foot_penalty(segments) as f64 / notes as f64;
    }
}

fn shares_an_arrow(a: &TimedNote, b: &TimedNote) -> bool {
    a.row.iter().zip(&b.row).any(|(&x, &y)| steps_on(x) && steps_on(y))
}

/// Arrows stepped on in a row. A row with any is a note.
fn arrows(row: &[Note; 4]) -> usize {
    row.iter().filter(|&&note| steps_on(note)).count()
}

fn steps_on(note: Note) -> bool {
    matches!(note, Note::Tap | Note::HoldStart | Note::RollStart)
}

/// Returns the beats of the first and last non-empty rows, or None for an empty chart.
fn count_notes_and_rhythms(measures: &[Measure], stats: &mut ChartStatistics) -> Option<(f64, f64)> {
    // Subdivision -> number of measures
    let mut sub_div_measures: BTreeMap<usize, usize> = BTreeMap::new();
    // (measure index, beat) of the first and last non-empty rows
    let mut first_used: Option<(usize, f64)> = None;
    let mut last_used = (0, 0.0);
    let mut measure_start = 0;

    for (m, measure) in measures.iter().enumerate() {
        let mut sub_div = None;
        for (r, row) in measure.rows.iter().enumerate() {
            if row.iter().all(|&note| note == Note::Empty) {
                continue;
            }
            let beat = row_beat(measure_start, measure, r);
            first_used.get_or_insert((m, beat));
            last_used = (m, beat);

            for note in row {
                match note {
                    Note::HoldStart => stats.num_holds += 1,
                    Note::RollStart => stats.num_rolls += 1,
                    Note::Mine => stats.num_mines += 1,
                    Note::Tap | Note::End | Note::Empty => {}
                }
            }
            let arrows = arrows(row);
            if arrows == 0 {
                continue;
            }
            stats.num_notes += 1;
            if arrows >= 2 {
                stats.num_jumps += 1;
            }
            let row_sub_div = row_sub_div(measure, r);
            sub_div = Some(sub_div.map_or(row_sub_div, |s| lcm(s, row_sub_div)));
        }
        if let Some(s) = sub_div {
            *sub_div_measures.entry(s).or_default() += 1;
        }
        measure_start += measure.num_beats;
    }

    stats.chart_num_measures = first_used.map_or(0, |(first, _)| last_used.0 - first + 1);
    stats.num_sub_divs = sub_div_measures.len();
    // `max_by_key` keeps the last maximum, so `rev` sends ties to the smallest.
    stats.mode_sub_div = sub_div_measures
        .iter()
        .rev()
        .max_by_key(|&(_, &count)| count)
        .map_or(0, |(&s, _)| s);
    stats.num_adv_sub_divs = sub_div_measures
        .iter()
        .filter(|&(&s, _)| !s.is_power_of_two())
        .map(|(_, &count)| count)
        .sum();

    first_used.map(|(_, first_beat)| (first_beat, last_used.1))
}

/// `span`: beats of the first and last objects, from `count_notes_and_rhythms`. Stops don't
/// count toward any BPM's time.
fn count_bpms(timing: &Timing, span: Option<(f64, f64)>, stats: &mut ChartStatistics) {
    let Some((start, end)) = span else { return };

    // (BPM, total seconds)
    let mut seconds_at: Vec<(f64, f64)> = Vec::new();
    let mut previous_bpm = None;
    for (bpm, beats) in timing.bpm_stretches(start, end) {
        if previous_bpm.is_some_and(|previous| previous != bpm) {
            stats.num_bpm_changes += 1;
        }
        previous_bpm = Some(bpm);

        let seconds = beats * 60.0 / bpm;
        match seconds_at.iter_mut().find(|(b, _)| *b == bpm) {
            Some((_, total)) => *total += seconds,
            None => seconds_at.push((bpm, seconds)),
        }
    }

    stats.mode_bpm = match seconds_at.iter().max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.total_cmp(&a.0))) {
        Some(&(bpm, _)) => bpm,
        // Everything is on one beat, so no time at any BPM: use the one in effect there.
        None => timing.bpm_at(start),
    };
}

fn row_beat(measure_start: usize, measure: &Measure, row: usize) -> f64 {
    // One division of integers, so equal beats give identical f64s (`Timing::new` relies on it).
    let rows = measure.rows.len();
    (measure_start * rows + row * measure.num_beats) as f64 / rows as f64
}

/// The row's subdivision, as in "16th notes": 4 on a beat, 8 halfway between two, 12 a third of
/// the way, and so on.
fn row_sub_div(measure: &Measure, row: usize) -> usize {
    // Denominator of `row * num_beats / rows` in lowest terms, times 4 (a beat is a quarter note)
    let rows = measure.rows.len();
    4 * (rows / gcd(row * measure.num_beats, rows))
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

/// Not yet implemented.
#[allow(dead_code)]
fn analyze_double_chart() -> ChartStatistics {
    todo!()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::sim_loader::{ChartDifficulty, SimFile};
    use ChartDifficulty::*;

    /// Loads a file in `test_resources/`. The `*_stats.sm` fixtures come from
    /// `make_test_simfiles.py`, and each one's header comment tabulates the expected values.
    fn load(file: &str) -> SimFile {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_resources").join(file);
        SimFile::from_file(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()))
    }

    fn single(sim: &SimFile, difficulty: ChartDifficulty) -> &SingleChart {
        sim.step_charts
            .iter()
            .find_map(|chart| match chart {
                StepChart::DanceSingle(single) if single.difficulty == difficulty => Some(single),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no dance-single {difficulty:?} chart"))
    }

    fn stats(file: &str, difficulty: ChartDifficulty) -> ChartStatistics {
        let sim = load(file);
        analyze_single_chart(single(&sim, difficulty), &sim.bpms, &sim.stops)
    }

    fn assert_close(actual: f64, expected: f64, what: &str) {
        assert!((actual - expected).abs() < 1e-9, "{what}: expected {expected}, got {actual}");
    }

    #[test]
    fn finds_each_rows_subdivision() {
        // (rows in the measure, row index, subdivision)
        let cases = [
            (4, 0, 4),
            (4, 3, 4),
            (8, 1, 8),
            (8, 2, 4),
            (12, 1, 12),
            (12, 2, 12),
            (12, 3, 4),
            (16, 2, 8),
            (16, 3, 16),
            (24, 1, 24),
            (24, 3, 8),
            (32, 25, 32),
            (48, 21, 16),
            (48, 28, 12),
            (192, 1, 192),
            (192, 48, 4),
            // Sizes StepMania doesn't write, but other editors might
            (6, 1, 12),
            (20, 1, 20),
        ];
        for (rows, row, expected) in cases {
            let measure = Measure { num_beats: 4, rows: vec![[Note::Empty; 4]; rows] };
            assert_eq!(row_sub_div(&measure, row), expected, "row {row} of a {rows}-row measure");
        }
    }

    #[test]
    fn counts_notes_and_rhythms() {
        // (difficulty, (notes, jumps, holds, rolls, mines), (measures, subdivs, mode, advanced))
        let expected = [
            (Beginner, (0, 0, 0, 0, 0), (0, 0, 0, 0)),
            (Easy, (14, 5, 4, 1, 3), (9, 1, 4, 0)),
            (Medium, (24, 0, 0, 0, 0), (2, 2, 8, 0)),
            (Hard, (83, 0, 1, 0, 3), (13, 6, 4, 3)),
            (Challenge, (55, 7, 3, 1, 2), (11, 4, 8, 1)),
        ];
        for (difficulty, counts, rhythms) in expected {
            let s = stats("chart_stats.sm", difficulty);
            assert_eq!(
                (s.num_notes, s.num_jumps, s.num_holds, s.num_rolls, s.num_mines),
                counts,
                "{difficulty:?}: (notes, jumps, holds, rolls, mines)"
            );
            assert_eq!(
                (s.chart_num_measures, s.num_sub_divs, s.mode_sub_div, s.num_adv_sub_divs),
                rhythms,
                "{difficulty:?}: (measures, subdivs, mode, advanced)"
            );
        }
    }

    #[test]
    fn counts_bpms() {
        // (file, difficulty, mode BPM, BPM changes)
        let expected = [
            ("chart_stats.sm", Beginner, 0.0, 0),
            ("chart_stats.sm", Easy, 120.0, 0),
            ("chart_stats.sm", Medium, 120.0, 0),
            ("chart_stats.sm", Hard, 120.0, 0),
            ("chart_stats.sm", Challenge, 120.0, 0),
            ("bpm_stats.sm", Beginner, 0.0, 0),
            ("bpm_stats.sm", Easy, 240.0, 0),
            ("bpm_stats.sm", Medium, 120.0, 1),
            ("bpm_stats.sm", Hard, 120.0, 3),
            ("bpm_stats.sm", Challenge, 120.0, 3),
        ];
        for (file, difficulty, mode_bpm, changes) in expected {
            let s = stats(file, difficulty);
            assert_eq!(
                (s.mode_bpm, s.num_bpm_changes),
                (mode_bpm, changes),
                "{file} {difficulty:?}: (mode BPM, BPM changes)"
            );
        }
    }

    #[test]
    fn converts_rows_to_times() {
        let sim = load("nps_stats.sm");
        let notes = to_timed_notes(&single(&sim, Easy).measures, &Timing::new(&sim.bpms, &sim.stops));
        let expected = [
            0.0,
            0.5,
            1.0,
            2.25,
            2.75,
            // Beat 5 1/3, on the second stop (written as 5.333)
            3.0 + 5.0 / 12.0,
            4.25,
            5.25,
            5.5,
            5.75,
            6.25,
            7.25,
            // Beat 17, a beat after the change to 180 BPM
            7.25 + 1.0 / 3.0,
        ];
        let times: Vec<f64> = notes.iter().map(|note| note.time).collect();
        assert_eq!(times.len(), expected.len(), "times: {times:?}");
        for (i, (&time, &expected)) in times.iter().zip(&expected).enumerate() {
            assert_close(time, expected, &format!("note {i}"));
        }
        // A hold from beat 9 to 11. Its end alone isn't a note.
        assert_eq!(notes[8].row, [Note::HoldStart, Note::Empty, Note::Empty, Note::Empty]);
        assert_eq!(notes[8].hold_ends, [Some(6.0), None, None, None]);
    }

    #[test]
    fn finds_the_mean_median_and_max() {
        assert_eq!(mean_median_max(vec![]), (0.0, 0.0, 0.0));
        assert_eq!(mean_median_max(vec![3.0]), (3.0, 3.0, 3.0));
        assert_eq!(mean_median_max(vec![5.0, 1.0, 3.0]), (3.0, 3.0, 5.0));
        // Even count: the median is the mean of the middle two
        assert_eq!(mean_median_max(vec![4.0, 1.0, 2.0, 9.0]), (4.0, 3.0, 9.0));
    }

    #[test]
    fn counts_notes_per_second() {
        // (difficulty, mean, median, max)
        let expected = [
            (Beginner, 0.0, 0.0, 0.0),
            (Easy, 24.0 / 13.0, 2.0, 3.0),
            (Medium, 1.0, 1.0, 1.0),
            (Hard, 81.0 / 16.0, 6.0, 6.0),
            (Challenge, 98.0 / 23.0, 4.0, 8.0),
        ];
        for (difficulty, mean, median, max) in expected {
            let s = stats("nps_stats.sm", difficulty);
            assert_close(s.mean_notes_per_second, mean, &format!("{difficulty:?} mean"));
            assert_close(s.median_notes_per_second, median, &format!("{difficulty:?} median"));
            assert_close(s.max_notes_per_second, max, &format!("{difficulty:?} max"));
        }
    }

    #[test]
    fn splits_notes_into_segments() {
        let sim = load("pattern_stats.sm");
        let timing = Timing::new(&sim.bpms, &sim.stops);
        let expected: [(ChartDifficulty, &[usize]); 5] = [
            (Beginner, &[]),
            (Easy, &[5, 9, 2]),
            (Medium, &[7, 8, 19, 1]),
            (Hard, &[20]),
            (Challenge, &[1, 1, 1, 1, 16, 6, 1]),
        ];
        for (difficulty, lengths) in expected {
            let notes = to_timed_notes(&single(&sim, difficulty).measures, &timing);
            let segments = segments(&notes, SEGMENT_BREAK_SECONDS);
            let actual: Vec<usize> = segments.iter().map(|segment| segment.len()).collect();
            assert_eq!(actual, lengths, "{difficulty:?}: notes in each segment");
        }
    }

    #[test]
    fn counts_jacks() {
        // (difficulty, jacks, jacks per second: mean, median, max)
        let expected = [
            (Beginner, 0, 0.0, 0.0, 0.0),
            (Easy, 8, 19.0 / 8.0, 2.0, 5.0),
            (Medium, 0, 0.0, 0.0, 0.0),
            (Hard, 10, 34.0 / 10.0, 3.0, 7.0),
            (Challenge, 4, 5.0 / 4.0, 1.0, 2.0),
        ];
        for (difficulty, jacks, mean, median, max) in expected {
            let s = stats("pattern_stats.sm", difficulty);
            assert_eq!(s.num_jacks, jacks, "{difficulty:?} jacks");
            assert_close(s.mean_jacks_per_second, mean, &format!("{difficulty:?} mean jacks per second"));
            assert_close(s.median_jacks_per_second, median, &format!("{difficulty:?} median jacks per second"));
            assert_close(s.max_jacks_per_second, max, &format!("{difficulty:?} max jacks per second"));
        }
    }

    #[test]
    fn counts_streams() {
        // (difficulty, streams, stream length: mean, median, max)
        let expected = [
            (Beginner, 0, 0.0, 0.0, 0.0),
            (Easy, 1, 9.0, 9.0, 9.0),
            (Medium, 2, 13.5, 13.5, 19.0),
            (Hard, 1, 20.0, 20.0, 20.0),
            (Challenge, 1, 16.0, 16.0, 16.0),
        ];
        for (difficulty, streams, mean, median, max) in expected {
            let s = stats("pattern_stats.sm", difficulty);
            assert_eq!(s.num_streams, streams, "{difficulty:?} streams");
            assert_close(s.mean_stream_len, mean, &format!("{difficulty:?} mean stream length"));
            assert_close(s.median_stream_len, median, &format!("{difficulty:?} median stream length"));
            assert_close(s.max_stream_len, max, &format!("{difficulty:?} max stream length"));
        }
    }

    #[test]
    fn measures_duration_and_activity() {
        // (difficulty, duration, active ratio)
        let expected = [
            (Beginner, 0.0, 0.0),
            (Easy, 7.0, 1.0 / 2.0),
            (Medium, 3.5, 4.0 / 7.0),
            (Hard, 2.75, 6.0 / 11.0),
            (Challenge, 6.0, 7.0 / 12.0),
        ];
        for (difficulty, duration, active) in expected {
            let s = stats("footing_stats.sm", difficulty);
            assert_close(s.chart_dur, duration, &format!("{difficulty:?} duration"));
            assert_close(s.chart_active_ratio, active, &format!("{difficulty:?} active ratio"));
        }
    }

    #[test]
    fn simulates_the_feet() {
        // (difficulty, crossovers, laterals, half spins, full spins)
        let expected = [
            (Beginner, 0, 0, 0, 0),
            (Easy, 0, 0, 0, 0),
            (Medium, 2, 0, 0, 0),
            (Hard, 0, 0, 0, 0),
            (Challenge, 1, 0, 0, 1),
        ];
        for (difficulty, crossovers, laterals, half_spins, full_spins) in expected {
            let s = stats("footing_stats.sm", difficulty);
            assert_eq!(
                (s.num_crossovers, s.num_laterals, s.num_half_spins, s.num_full_spins),
                (crossovers, laterals, half_spins, full_spins),
                "{difficulty:?}: (crossovers, laterals, half spins, full spins)"
            );
        }
    }

    #[test]
    fn finds_the_wrong_foot_penalty() {
        let expected = [(Beginner, 0.0), (Easy, 0.0), (Medium, 0.0), (Hard, 2.0 / 8.0), (Challenge, 4.0 / 16.0)];
        for (difficulty, penalty) in expected {
            let s = stats("footing_stats.sm", difficulty);
            assert_close(s.wrong_foot_penalty_per_note, penalty, &format!("{difficulty:?} wrong-foot penalty"));
        }
    }

    #[test]
    fn features_cover_every_statistic() {
        // A different value in every field, to catch a feature reading the wrong one
        let names: Vec<String> =
            serde_json::to_value(ChartStatistics::default()).unwrap().as_object().unwrap().keys().cloned().collect();
        let values: serde_json::Map<String, serde_json::Value> =
            names.iter().enumerate().map(|(i, name)| (name.clone(), serde_json::json!(i + 1))).collect();
        let stats: ChartStatistics = serde_json::from_value(serde_json::Value::Object(values.clone())).unwrap();

        let features = stats.features();
        let mut feature_names: Vec<String> = features.iter().map(|(name, _)| name.to_string()).collect();
        feature_names.sort();
        assert_eq!(feature_names, names, "features() and ChartStatistics have different fields");
        for (name, value) in features {
            assert_eq!(Some(value), values[name].as_f64(), "feature {name}");
        }
    }
}
