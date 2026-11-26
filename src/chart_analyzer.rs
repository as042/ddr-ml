mod counts;
mod footing;
mod radar;
mod sequences;
mod statistics;
mod technique;

use serde::{Deserialize, Serialize};

use crate::sim_loader::{BPMChange, ChartDifficulty, Measure, Note, SimFile, SingleChart, StepChart, Stop};
#[cfg(test)]
pub use footing::{Cost, TEST_COSTS};
pub use footing::{COSTS, Costs, Events, Foot, MAX_COST, Solution};
pub use statistics::ChartStatistics;
pub use technique::{Source, Technique};

/// The analyzer's thresholds, beside the foot simulation's costs. `SETTINGS` are the defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Settings {
    /// Longest gap between notes of one segment, unless a hold runs through it (see `segments`)
    pub segment_break_seconds: f64,
    /// Most branches the foot search keeps after a note (see `footing`)
    pub max_footing_branches: usize,
    /// Degrees a turn can go back from its furthest point and still be one turn (see `footing`)
    pub turn_back_degrees: i32,
    /// Sliding window for the notes and the cost per second
    pub window_seconds: f64,
    /// Longer gaps between notes are breaks, not active time
    pub active_gap_seconds: f64,
    /// Gaps are even if the longest is this fraction longer than the shortest, at most
    pub even_tolerance: f64,
    /// Fewest notes in a stream
    pub min_stream_notes: usize,
    /// Fewest notes in a drill
    pub min_drill_notes: usize,
    /// A gallop or triple is on its own if the gap on each side is this many times its own, at
    /// least
    pub isolation: f64,
}

pub const SETTINGS: Settings = Settings {
    segment_break_seconds: 2.0,
    max_footing_branches: 5000,
    turn_back_degrees: 45,
    window_seconds: 1.0,
    active_gap_seconds: 1.0,
    even_tolerance: 0.1,
    min_stream_notes: 8,
    min_drill_notes: 5,
    isolation: 2.0,
};

impl Settings {
    /// Every setting by name (its field's, with spaces), as a number.
    pub fn values(&self) -> [(&'static str, f64); 9] {
        [
            ("segment break seconds", self.segment_break_seconds),
            ("max footing branches", self.max_footing_branches as f64),
            ("turn back degrees", self.turn_back_degrees as f64),
            ("window seconds", self.window_seconds),
            ("active gap seconds", self.active_gap_seconds),
            ("even tolerance", self.even_tolerance),
            ("min stream notes", self.min_stream_notes as f64),
            ("min drill notes", self.min_drill_notes as f64),
            ("isolation", self.isolation),
        ]
    }

    /// Sets the one called `name` (as in `values`), or says why `value` can't be it.
    pub fn set(&mut self, name: &str, value: f64) -> Result<(), String> {
        let at_least = |least: f64| match value.is_finite() && value >= least {
            true => Ok(value),
            false => Err(format!("must be {least} or more")),
        };
        let positive = || match value.is_finite() && value > 0.0 {
            true => Ok(value),
            false => Err("must be more than 0".to_string()),
        };
        // Below a billion, which fits any integer type here
        let whole = |least: f64| match value.fract() == 0.0 && (least..1e9).contains(&value) {
            true => Ok(value),
            false => Err(format!("must be a whole number, {least} or more")),
        };
        match name {
            "segment break seconds" => self.segment_break_seconds = at_least(0.0)?,
            "max footing branches" => self.max_footing_branches = whole(1.0)? as usize,
            "turn back degrees" => self.turn_back_degrees = whole(0.0)? as i32,
            "window seconds" => self.window_seconds = positive()?,
            "active gap seconds" => self.active_gap_seconds = at_least(0.0)?,
            "even tolerance" => self.even_tolerance = at_least(0.0)?,
            // A run of one note has no gap
            "min stream notes" => self.min_stream_notes = whole(2.0)? as usize,
            "min drill notes" => self.min_drill_notes = whole(2.0)? as usize,
            "isolation" => self.isolation = at_least(0.0)?,
            _ => return Err(format!("there is no setting called {name}")),
        }
        Ok(())
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
                stats: analyze_single_chart(single, &single.bpms, &single.stops, &COSTS, &SETTINGS),
            }),
            StepChart::DanceDouble(_) => None,
        })
        .collect()
}

/// `costs`: the foot simulation's.
pub fn analyze_single_chart(
    chart: &SingleChart,
    bpms: &[BPMChange],
    stops: &[Stop],
    costs: &Costs,
    settings: &Settings,
) -> ChartStatistics {
    analyze_chart(chart, bpms, stops, costs, settings).stats
}

/// A chart's statistics, and the segments they were counted from.
pub struct ChartAnalysis {
    pub stats: ChartStatistics,
    /// In time order. Their notes are all the chart's.
    pub segments: Vec<AnalyzedSegment>,
}

/// A segment (see `segments`) as the analyzer sees it.
pub struct AnalyzedSegment {
    pub notes: Vec<TimedNote>,
    /// The cheapest footing, with its events and patterns
    pub footing: Solution,
    /// What the notes alone have (see `sequences`), in this segment's note indices
    pub from_notes: Vec<technique::Occurrence>,
}

/// `analyze_single_chart`, keeping the footing and the techniques found.
pub fn analyze_chart(chart: &SingleChart, bpms: &[BPMChange], stops: &[Stop], costs: &Costs, settings: &Settings) -> ChartAnalysis {
    let timing = Timing::new(bpms, stops);
    let notes = to_timed_notes(&chart.measures, &timing);
    let mut stats = ChartStatistics::default();
    counts::count(&chart.measures, &notes, &timing, settings, &mut stats);
    radar::count(&chart.measures, &timing, &mut stats);

    let segments: Vec<AnalyzedSegment> = segments(&notes, settings.segment_break_seconds)
        .into_iter()
        .map(|segment| AnalyzedSegment {
            notes: segment.to_vec(),
            footing: footing::solve(segment, costs, settings),
            from_notes: sequences::find(segment, settings),
        })
        .collect();
    let occurrences: Vec<technique::Occurrence> =
        segments.iter().flat_map(|segment| segment.from_notes.iter().chain(&segment.footing.occurrences)).copied().collect();
    stats.techniques = statistics::summarize(&occurrences);
    let note_costs = segments.iter().flat_map(|segment| segment.footing.note_costs.iter().copied()).collect();
    counts::count_costs(&notes, note_costs, settings.window_seconds, &mut stats);
    let moves: Vec<footing::moves::Move> = segments.iter().flat_map(|segment| segment.footing.moves.iter().copied()).collect();
    counts::count_feet(&moves, &mut stats);
    ChartAnalysis { stats, segments }
}

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
pub struct TimedNote {
    /// Seconds from beat 0
    pub time: f64,
    pub beat: f64,
    /// The whole row, including mines and hold ends
    pub row: [Note; 4],
    /// Per arrow: when the hold or roll starting here ends
    pub hold_ends: [Option<f64>; 4],
}

impl TimedNote {
    /// Whether the note has `arrow` (0 to 3: Left, Down, Up, Right) to step on.
    pub fn steps(&self, arrow: usize) -> bool {
        steps_on(self.row[arrow])
    }
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
            let beat = row_beat(measure_start, measure, r);
            let time = timing.time_at(beat);
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
                notes.push(TimedNote { time, beat, row: *row, hold_ends: [None; 4] });
            }
        }
        measure_start += measure.num_beats;
    }
    notes
}

/// Runs of notes (in time order) with no gap longer than `break_seconds`, unless a hold is
/// still down at the gap's end. A longer gap lets the player reset their feet, so techniques
/// are only looked for inside a segment. A foot on a hold can't reset.
fn segments(notes: &[TimedNote], break_seconds: f64) -> Vec<&[TimedNote]> {
    let mut segments = Vec::new();
    // Where the segment starts, and when the last of its holds so far ends
    let (mut start, mut held_until) = (0, f64::NEG_INFINITY);
    for (i, note) in notes.iter().enumerate() {
        let apart = i > start && note.time - notes[i - 1].time > break_seconds + TIME_EPSILON;
        if apart && held_until <= note.time + TIME_EPSILON {
            segments.push(&notes[start..i]);
            start = i;
        }
        held_until = note.hold_ends.iter().flatten().fold(held_until, |until, &end| until.max(end));
    }
    if start < notes.len() {
        segments.push(&notes[start..]);
    }
    segments
}

/// A segment and its cheapest footing.
pub struct FootedSegment {
    pub notes: Vec<TimedNote>,
    pub footing: Solution,
}

/// A chart's notes in segments, split as for the footing statistics with `SETTINGS`.
pub fn footing_segments(chart: &SingleChart, bpms: &[BPMChange], stops: &[Stop]) -> Vec<Vec<TimedNote>> {
    let notes = to_timed_notes(&chart.measures, &Timing::new(bpms, stops));
    segments(&notes, SETTINGS.segment_break_seconds).into_iter().map(<[TimedNote]>::to_vec).collect()
}

/// The cheapest footing of a segment, from home, with `SETTINGS`.
pub fn foot_segment(notes: &[TimedNote], costs: &Costs) -> Solution {
    footing::solve(notes, costs, &SETTINGS)
}

/// Each of a chart's `footing_segments`, footed.
pub fn foot_chart(chart: &SingleChart, bpms: &[BPMChange], stops: &[Stop], costs: &Costs) -> Vec<FootedSegment> {
    footing_segments(chart, bpms, stops)
        .into_iter()
        .map(|notes| FootedSegment { footing: foot_segment(&notes, costs), notes })
        .collect()
}

/// One segment from arrow letters, footed from home: `L`, `D`, `U` and `R` in either case, a
/// jump in brackets (`[LR]`). `+` after an arrow starts a hold on it, which lasts until that
/// arrow with a `-` after it (a row of its own, or in brackets with the notes it ends at), or to
/// the end. A space is an empty row: a pause. The rows are `speed` times as close as the costs'
/// reference gap.
pub fn foot_arrows(arrows: &str, costs: &Costs, speed: f64) -> Result<FootedSegment, String> {
    let mut rows: Vec<[Note; 4]> = Vec::new();
    let mut jump: Option<[Note; 4]> = None;
    // The arrow a `+` or `-` would mark
    let mut last: Option<usize> = None;
    for c in arrows.trim().chars() {
        match c {
            c if c.is_whitespace() => {
                if jump.is_none() {
                    rows.push([Note::Empty; 4]);
                    last = None;
                }
            }
            '[' if jump.is_none() => (jump, last) = (Some([Note::Empty; 4]), None),
            ']' => match (jump.take(), last.take()) {
                (Some(row), _) if row != [Note::Empty; 4] => rows.push(row),
                (Some(_), _) => return Err("`[]` has no arrows in it".into()),
                (None, _) => return Err("`]` without a `[`".into()),
            },
            '+' | '-' => {
                let arrow = last.take().ok_or_else(|| format!("`{c}` doesn't follow an arrow"))?;
                let row = match &mut jump {
                    Some(row) => row,
                    None => rows.last_mut().expect("the arrow's row"),
                };
                row[arrow] = if c == '+' { Note::HoldStart } else { Note::End };
            }
            _ => {
                let arrow = "LDUR"
                    .find(c.to_ascii_uppercase())
                    .ok_or_else(|| format!("`{c}` isn't an arrow (L, D, U or R)"))?;
                match &mut jump {
                    Some(row) => row[arrow] = Note::Tap,
                    None => {
                        let mut row = [Note::Empty; 4];
                        row[arrow] = Note::Tap;
                        rows.push(row);
                    }
                }
                last = Some(arrow);
            }
        }
    }
    if jump.is_some() {
        return Err("`[` without a `]`".into());
    }

    let gap = costs.reference_gap / speed;
    let mut notes: Vec<TimedNote> = Vec::new();
    // Per arrow: index of the note that started its open hold
    let mut holding: [Option<usize>; 4] = [None; 4];
    let letter = |arrow: usize| &"LDUR"[arrow..=arrow];
    for (index, row) in rows.iter().enumerate() {
        let time = index as f64 * gap;
        for (arrow, &note) in row.iter().enumerate() {
            match (note, holding[arrow]) {
                (Note::End, Some(start)) => (notes[start].hold_ends[arrow], holding[arrow]) = (Some(time), None),
                (Note::End, None) => return Err(format!("`{}-` without a hold on it", letter(arrow))),
                (Note::Empty, _) | (_, None) => {}
                (_, Some(_)) => return Err(format!("`{}` is stepped while it's held", letter(arrow))),
            }
        }
        if row.iter().any(|&note| steps_on(note)) {
            for (arrow, &note) in row.iter().enumerate() {
                if note == Note::HoldStart {
                    holding[arrow] = Some(notes.len());
                }
            }
            // Beats as for 16ths
            notes.push(TimedNote { time, beat: index as f64 * 0.25, row: *row, hold_ends: [None; 4] });
        }
    }
    let end = rows.len() as f64 * gap;
    for (arrow, start) in holding.into_iter().enumerate() {
        if let Some(start) = start {
            notes[start].hold_ends[arrow] = Some(end);
        }
    }
    if notes.is_empty() {
        return Err("no arrows".into());
    }
    let footing = footing::solve(&notes, costs, &SETTINGS);
    Ok(FootedSegment { notes, footing })
}

/// Arrows stepped on in a row. A row with any is a note.
fn arrows(row: &[Note; 4]) -> usize {
    row.iter().filter(|&&note| steps_on(note)).count()
}

fn steps_on(note: Note) -> bool {
    matches!(note, Note::Tap | Note::HoldStart | Note::RollStart)
}

fn row_beat(measure_start: usize, measure: &Measure, row: usize) -> f64 {
    // One division of integers, so equal beats give identical f64s (`Timing::new` relies on it).
    let rows = measure.rows.len();
    (measure_start * rows + row * measure.num_beats) as f64 / rows as f64
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

    const COSTS: Costs = TEST_COSTS;
    use ChartDifficulty::*;

    /// Loads a file in `test_resources/`. The `*_stats.sm` fixtures come from
    /// `make_test_simfiles.py`, and each one's header comment tabulates the expected values.
    pub(super) fn load(file: &str) -> SimFile {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_resources").join(file);
        SimFile::from_file(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()))
    }

    pub(super) fn single(sim: &SimFile, difficulty: ChartDifficulty) -> &SingleChart {
        sim.step_charts
            .iter()
            .find_map(|chart| match chart {
                StepChart::DanceSingle(single) if single.difficulty == difficulty => Some(single),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no dance-single {difficulty:?} chart"))
    }

    pub(super) fn stats(file: &str, difficulty: ChartDifficulty) -> ChartStatistics {
        let sim = load(file);
        let chart = single(&sim, difficulty);
        analyze_single_chart(chart, &chart.bpms, &chart.stops, &COSTS, &SETTINGS)
    }

    pub(super) fn assert_close(actual: f64, expected: f64, what: &str) {
        assert!((actual - expected).abs() < 1e-9, "{what}: expected {expected}, got {actual}");
    }

    /// A Hard chart of rows 0.125 s apart (16ths at 120 BPM): `L`, `D`, `U` and `R` for taps, `.`
    /// for an empty row.
    fn sixteenths(tokens: &str) -> SingleChart {
        let mut rows: Vec<[Note; 4]> = tokens
            .split_whitespace()
            .map(|token| {
                let mut row = [Note::Empty; 4];
                for (arrow, letter) in "LDUR".chars().enumerate() {
                    if token.contains(letter) {
                        row[arrow] = Note::Tap;
                    }
                }
                row
            })
            .collect();
        rows.resize(rows.len().next_multiple_of(16), [Note::Empty; 4]);
        let measures = rows.chunks(16).map(|rows| Measure { num_beats: 4, rows: rows.to_vec() }).collect();
        SingleChart { difficulty: Hard, rating: 10.0, measures, bpms: vec![BPMChange { beat: 0.0, bpm: 120.0 }], stops: Vec::new() }
    }

    #[test]
    fn every_setting_reaches_the_statistics() {
        // Segments: a footing a search one branch wide gets wrong; one that turns right round by way
        // of a lateral when a turn can't go back at all (each turn is then too short to be a spin);
        // a stream and a drill straight into a slower stream, then a gallop
        let gap = ". ".repeat(24);
        let chart = sixteenths(&format!(
            "L D R L {gap} U L D R U L D R U L D R {gap} L R L R L R L R L R L R . L . D . U . R . L . D . U . R . . . . U D"
        ));
        // Crossovers cheap enough to hide a spin behind
        let costs = Costs { horizontal_crossover: Cost { base: 1.5, speed: 0.2 }, ..COSTS };
        let statistic = |settings: &Settings, name: &str| {
            let features = analyze_single_chart(&chart, &chart.bpms, &chart.stops, &costs, settings).features();
            features.into_iter().find(|(feature, _)| feature == name).unwrap_or_else(|| panic!("no {name}")).1
        };
        let changes = [
            ("segment break seconds", 0.0, "stream_count"),
            ("max footing branches", 1.0, "mean_cost_per_note"),
            ("turn back degrees", 0.0, "lateral_count"),
            ("window seconds", 10.0, "max_notes_per_second"),
            ("active gap seconds", 0.0, "longest_run_notes"),
            ("even tolerance", 1.0, "stream_count"),
            ("min stream notes", 1000.0, "stream_count"),
            ("min drill notes", 1000.0, "drill_count"),
            ("isolation", 1000.0, "gallop_count"),
        ];
        let names: Vec<&str> = SETTINGS.values().iter().map(|(name, _)| *name).collect();
        assert_eq!(changes.map(|(name, ..)| name).to_vec(), names, "one change per setting");
        for (name, value, changed) in changes {
            let mut settings = SETTINGS;
            settings.set(name, value).unwrap();
            assert_ne!(statistic(&settings, changed), statistic(&SETTINGS, changed), "{name} at {value}: {changed}");
        }
    }

    #[test]
    fn sets_each_setting_by_name() {
        // `values` names the fields
        let fields = serde_json::to_value(SETTINGS).unwrap();
        let fields = fields.as_object().unwrap();
        assert_eq!(fields.len(), SETTINGS.values().len());
        for (name, value) in SETTINGS.values() {
            assert_eq!(fields[&name.replace(' ', "_")].as_f64(), Some(value), "{name}");
        }

        let mut settings = SETTINGS;
        for (name, value) in [("turn back degrees", 90.0), ("max footing branches", 7.0), ("even tolerance", 0.0)] {
            settings.set(name, value).unwrap();
        }
        assert_eq!((settings.turn_back_degrees, settings.max_footing_branches, settings.even_tolerance), (90, 7, 0.0));
        let refused = [
            ("max footing branches", 0.0, "a whole number, 1 or more"),
            ("max footing branches", 2.5, "a whole number, 1 or more"),
            ("min stream notes", 1.0, "a whole number, 2 or more"),
            ("turn back degrees", -45.0, "a whole number, 0 or more"),
            ("window seconds", 0.0, "more than 0"),
            ("segment break seconds", -1.0, "0 or more"),
            ("isolation", f64::NAN, "0 or more"),
            ("active gap seconds", f64::INFINITY, "0 or more"),
            ("segment length", 1.0, "no setting called segment length"),
        ];
        for (name, value, error) in refused {
            let mut settings = SETTINGS;
            let message = settings.set(name, value).unwrap_err();
            assert!(message.contains(error), "{name} {value}: {message}");
        }
    }

    #[test]
    fn converts_rows_to_times() {
        let sim = load("nps_stats.sm");
        let chart = single(&sim, Easy);
        let notes = to_timed_notes(&chart.measures, &Timing::new(&chart.bpms, &chart.stops));
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
    fn foots_arrow_letters() {
        let footed = foot_arrows("ld[U D]r", &COSTS, 1.0).unwrap();
        let rows: Vec<[bool; 4]> = footed.notes.iter().map(|note| [0, 1, 2, 3].map(|arrow| note.steps(arrow))).collect();
        assert_eq!(
            rows,
            [[true, false, false, false], [false, true, false, false], [false, true, true, false], [false, false, false, true]]
        );
        assert_eq!(footed.footing.steps.len(), 4);
        // Evenly spaced, at speed 1
        assert!(footed.notes.windows(2).all(|pair| pair[1].time - pair[0].time == COSTS.reference_gap));
        let crossover = COSTS.horizontal_crossover.base + COSTS.horizontal_crossover.speed;
        assert_eq!(foot_arrows("LDRDL", &COSTS, 1.0).unwrap().footing.cost, crossover);

        // A space is a pause of one row. Twice the speed, half the gaps.
        let times = |arrows: &str, speed: f64| -> Vec<f64> {
            foot_arrows(arrows, &COSTS, speed).unwrap().notes.iter().map(|note| note.time).collect()
        };
        let gap = COSTS.reference_gap;
        assert_eq!(times(" LD R  U ", 1.0), [0.0, gap, 3.0 * gap, 6.0 * gap]);
        assert_eq!(times("LD R", 2.0), [0.0, gap / 2.0, 3.0 * gap / 2.0]);
        let fast = foot_arrows("LDRDL", &COSTS, 2.0).unwrap().footing.cost;
        assert_eq!(fast, COSTS.horizontal_crossover.base + COSTS.horizontal_crossover.speed * 2.0);
        // Up's hold ends in the pause
        let held = foot_arrows("U+L U-R", &COSTS, 1.0).unwrap();
        assert_eq!(held.notes[0].hold_ends[2], Some(3.0 * gap));

        // Down held under two notes, Left to the end, Up until the last jump
        let held = foot_arrows("d+L+rD-[U+R][U-D]", &COSTS, 1.0);
        let held = held.as_ref().map(|footed| footed.notes.iter().map(|note| note.hold_ends).collect::<Vec<_>>());
        let gap = COSTS.reference_gap;
        let expected = [
            [None, Some(3.0 * gap), None, None],
            [Some(6.0 * gap), None, None, None],
            [None; 4],
            [None, None, Some(5.0 * gap), None],
            [None; 4],
        ];
        assert_eq!(held.as_deref(), Ok(&expected[..]));

        for bad in ["", "  ", "LXR", "L[UD", "LD]", "L[]R", "[L[D]]", "+L", "L-", "L+L", "[LR]+", "L+-", "D+D-D-", "L +", "L+ L"] {
            assert!(foot_arrows(bad, &COSTS, 1.0).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn foots_each_segment_of_a_chart() {
        let sim = load("footing_stats.sm");
        for chart in &sim.step_charts {
            let StepChart::DanceSingle(chart) = chart else { continue };
            let footed = foot_chart(chart, &chart.bpms, &chart.stops, &COSTS);
            let notes: usize = footed.iter().map(|segment| segment.notes.len()).sum();
            assert_eq!(notes, stats("footing_stats.sm", chart.difficulty).num_notes, "{:?}", chart.difficulty);
            for segment in &footed {
                assert_eq!(segment.footing.steps.len(), segment.notes.len());
                assert!(segment.footing.exact);
            }
        }
    }

    #[test]
    fn keeps_a_hold_and_the_notes_after_it_in_one_segment() {
        // A tap on `arrow`, held until `held_until` if it's a hold
        let at = |time: f64, arrow: usize, held_until: Option<f64>| {
            let (mut row, mut hold_ends) = ([Note::Empty; 4], [None; 4]);
            row[arrow] = if held_until.is_some() { Note::HoldStart } else { Note::Tap };
            hold_ends[arrow] = held_until;
            TimedNote { time, beat: time, row, hold_ends }
        };
        let lengths = |notes: &[TimedNote]| -> Vec<usize> {
            segments(notes, SETTINGS.segment_break_seconds).iter().map(|segment| segment.len()).collect()
        };
        // Taps: a gap of 2.5 s is a break, one of 2 s isn't
        assert_eq!(lengths(&[at(0.0, 0, None), at(1.0, 1, None), at(3.5, 2, None), at(4.0, 3, None)]), [2, 2]);
        assert_eq!(lengths(&[at(0.0, 0, None), at(2.0, 1, None)]), [2]);
        // Left held to 4 s: the taps from 2.5 s are under it
        let held = [at(0.0, 0, Some(4.0)), at(2.5, 1, None), at(3.0, 2, None), at(3.5, 1, None)];
        assert_eq!(lengths(&held), [4]);
        // Once the hold is over, or as it ends, the gap since the note before decides again
        assert_eq!(lengths(&[at(0.0, 0, Some(2.4)), at(2.5, 1, None)]), [1, 1]);
        assert_eq!(lengths(&[at(0.0, 0, Some(2.5)), at(2.5, 1, None)]), [1, 1]);
        assert_eq!(lengths(&[at(0.0, 0, Some(2.6)), at(2.5, 1, None)]), [2]);
        // The longest hold so far counts, not just the last note's
        assert_eq!(lengths(&[at(0.0, 0, Some(6.0)), at(1.0, 1, Some(1.5)), at(5.0, 2, None)]), [3]);
        assert_eq!(lengths(&[at(0.0, 0, Some(6.0)), at(1.0, 1, Some(1.5)), at(7.0, 2, None)]), [2, 1]);
        assert_eq!(lengths(&[]), Vec::<usize>::new());

        // The foot search then knows the left foot is down: Down, Up, Down under the hold
        // takes a bracket or doublesteps, where on their own they would be free
        let footing = foot_segment(&held, &COSTS);
        assert!(footing.cost > 0.0);
        assert_eq!(foot_segment(&held[1..], &COSTS).cost, 0.0);
        let found = sequences::find(&held, &SETTINGS);
        assert_eq!(found.iter().filter(|found| found.technique == Technique::HeldStep).count(), 3);
    }

    #[test]
    fn splits_notes_into_segments() {
        let sim = load("footing_stats.sm");
        let expected: [(ChartDifficulty, &[usize]); 5] =
            [(Beginner, &[]), (Easy, &[5, 1]), (Medium, &[3, 3]), (Hard, &[4, 4]), (Challenge, &[6, 4, 6])];
        for (difficulty, lengths) in expected {
            let chart = single(&sim, difficulty);
            let notes = to_timed_notes(&chart.measures, &Timing::new(&chart.bpms, &chart.stops));
            let segments = segments(&notes, SETTINGS.segment_break_seconds);
            let actual: Vec<usize> = segments.iter().map(|segment| segment.len()).collect();
            assert_eq!(actual, lengths, "{difficulty:?}: notes in each segment");
        }
    }

    /// Checks a technique's count and its speeds' (mean, median, max).
    fn assert_speeds(s: &ChartStatistics, technique: Technique, count: usize, speeds: (f64, f64, f64), chart: ChartDifficulty) {
        let found = s.technique(technique);
        let what = format!("{chart:?} {}", technique.plural());
        assert_eq!(found.count, count, "{what}");
        assert_close(found.mean_speed, speeds.0, &format!("{what}: mean speed"));
        assert_close(found.median_speed, speeds.1, &format!("{what}: median speed"));
        assert_close(found.max_speed, speeds.2, &format!("{what}: max speed"));
    }

    #[test]
    fn finds_repeated_arrows_and_jumps() {
        // (difficulty, repeated arrows, jumps), each (count, speeds: (mean, median, max))
        let none = (0, (0.0, 0.0, 0.0));
        let expected = [
            (Beginner, none, none),
            (Easy, (10, (91.0 / 15.0, 8.0, 8.0)), (2, (6.0, 6.0, 8.0))),
            (Medium, none, none),
            (Hard, (10, (6.8, 8.0, 8.0)), (4, (5.0, 4.0, 8.0))),
            (Challenge, (9, (296.0 / 81.0, 2.0, 8.0)), (4, (85.0 / 18.0, 5.0, 8.0))),
        ];
        for (difficulty, repeats, jumps) in expected {
            let s = stats("pattern_stats.sm", difficulty);
            assert_speeds(&s, Technique::RepeatedArrow, repeats.0, repeats.1, difficulty);
            assert_speeds(&s, Technique::Jump, jumps.0, jumps.1, difficulty);
        }
    }

    #[test]
    fn finds_streams() {
        // (difficulty, streams, their notes, lengths: (mean, median, max), speeds: (mean, median, max))
        let expected = [
            (Beginner, 0, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (Easy, 0, 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            (Medium, 2, 24, (12.0, 12.0, 16.0), (6.0, 6.0, 8.0)),
            (Hard, 1, 17, (17.0, 17.0, 17.0), (8.0, 8.0, 8.0)),
            (Challenge, 1, 16, (16.0, 16.0, 16.0), (4.0, 4.0, 4.0)),
        ];
        for (difficulty, streams, notes, lengths, speeds) in expected {
            let s = stats("pattern_stats.sm", difficulty);
            assert_speeds(&s, Technique::Stream, streams, speeds, difficulty);
            let found = s.technique(Technique::Stream);
            assert_eq!(found.notes, notes, "{difficulty:?}: notes in streams");
            assert_close(found.mean_length, lengths.0, &format!("{difficulty:?} mean stream length"));
            assert_close(found.median_length, lengths.1, &format!("{difficulty:?} median stream length"));
            assert_close(found.max_length, lengths.2, &format!("{difficulty:?} max stream length"));
        }
    }

    #[test]
    fn finds_the_footings_events_and_patterns() {
        // (difficulty, the techniques found on its footing: (technique, count, each one's speed))
        let expected: [(ChartDifficulty, &[(Technique, usize, f64)]); 5] = [
            (Beginner, &[]),
            (Easy, &[(Technique::Staircase, 1, 2.0)]),
            (Medium, &[(Technique::Doublestep, 2, 2.0)]),
            (Hard, &[]),
            (Challenge, &[(Technique::Doublestep, 2, 4.0), (Technique::Bracket, 1, 4.0), (Technique::Gallop, 1, 4.0)]),
        ];
        for (difficulty, found) in expected {
            let s = stats("footing_stats.sm", difficulty);
            for technique in Technique::ALL.into_iter().filter(|technique| technique.source() != Source::Notes) {
                let (count, speed) = found
                    .iter()
                    .find(|(expected, ..)| *expected == technique)
                    .map_or((0, 0.0), |&(_, count, speed)| (count, speed));
                assert_speeds(&s, technique, count, (speed, speed, speed), difficulty);
            }
        }
    }
}
