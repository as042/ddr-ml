//! DDR's groove radar, estimated as the arcade game is thought to work it out (DDR A, single
//! play). Each value is a rate of the chart on a scale where 100 is a lot and the hardest chart
//! of its day is 200. A row of four mines is a shock arrow.

use super::counts::row_sub_div;
use super::statistics::ChartStatistics;
use super::{Timing, arrows, row_beat};
use crate::sim_loader::{Measure, Note};

/// A radar value from its rate: `rate * low.0 / low.1` below `knee`, and from there
/// `(rate + high.0) * high.1 / high.2`. Whole, rounded down.
struct Scale {
    knee: f64,
    low: (f64, f64),
    high: (f64, f64, f64),
}

impl Scale {
    fn value(&self, rate: f64) -> f64 {
        let value = if rate < self.knee { rate * self.low.0 / self.low.1 } else { (rate + self.high.0) * self.high.1 / self.high.2 };
        round_down(value)
    }
}

/// To a whole number. One that an f64 puts a hair under a whole number is that number.
fn round_down(value: f64) -> f64 {
    (value + 1e-9).floor()
}

/// Of notes and shock arrows a minute.
const STREAM: Scale = Scale { knee: 300.0, low: (1.0, 3.0), high: (-139.0, 100.0, 161.0) };
/// Of the most notes and mines in any `VOLTAGE_BEATS`, as notes a minute at the average BPM.
const VOLTAGE: Scale = Scale { knee: 600.0, low: (1.0, 6.0), high: (594.0, 100.0, 1194.0) };
/// Of jumps and shock arrows a minute.
const AIR: Scale = Scale { knee: 55.0, low: (20.0, 11.0), high: (36.0, 100.0, 91.0) };
/// Of the beats held, in ten-thousandths of the song's.
const FREEZE: Scale = Scale { knee: 3500.0, low: (1.0, 35.0), high: (2484.0, 100.0, 5984.0) };
/// Of the irregularity a second, times 100 (see `irregularity`).
const CHAOS: Scale = Scale { knee: 2000.0, low: (1.0, 20.0), high: (21605.0, 100.0, 23605.0) };

const VOLTAGE_BEATS: f64 = 4.0;

/// Irregularity is this many times as much for each BPM changed by a minute.
const CHAOS_PER_BPM_CHANGE: f64 = 1.0 / 1500.0;

/// A row with an arrow to step on or a mine.
struct Row {
    beat: f64,
    arrows: usize,
    mines: usize,
    sub_div: usize,
}

impl Row {
    fn is_shock(&self) -> bool {
        self.arrows == 0 && self.mines == 4
    }
}

/// The song runs from beat 0 to the end of the last measure with anything in it.
pub(super) fn count(measures: &[Measure], timing: &Timing, stats: &mut ChartStatistics) {
    let mut rows: Vec<Row> = Vec::new();
    // (start, end) beats of each hold and roll
    let mut holds: Vec<(f64, f64)> = Vec::new();
    let mut open: [Option<f64>; 4] = [None; 4];
    let (mut measure_start, mut song_beats, mut last_beat) = (0, 0.0, 0.0);
    for measure in measures {
        let measure_end = measure_start + measure.num_beats;
        for (r, row) in measure.rows.iter().enumerate() {
            if row.iter().all(|&note| note == Note::Empty) {
                continue;
            }
            let beat = row_beat(measure_start, measure, r);
            (song_beats, last_beat) = (measure_end as f64, beat);
            for (arrow, &note) in row.iter().enumerate() {
                match (note, open[arrow]) {
                    (Note::End, Some(start)) => {
                        holds.push((start, beat));
                        open[arrow] = None;
                    }
                    (Note::HoldStart | Note::RollStart, _) => open[arrow] = Some(beat),
                    _ => {}
                }
            }
            let (arrows, mines) = (arrows(row), row.iter().filter(|&&note| note == Note::Mine).count());
            if arrows + mines > 0 {
                rows.push(Row { beat, arrows, mines, sub_div: row_sub_div(measure, r) });
            }
        }
        measure_start = measure_end;
    }
    let seconds = timing.time_at(song_beats);
    if seconds <= 0.0 {
        return;
    }
    let per_minute = |count: usize| round_down(60.0 * count as f64 / seconds);
    let shocks = rows.iter().filter(|row| row.is_shock()).count();

    stats.radar_stream = STREAM.value(per_minute(rows.iter().filter(|row| row.arrows > 0).count() + shocks));
    stats.radar_air = AIR.value(per_minute(rows.iter().filter(|row| row.arrows >= 2).count() + shocks));

    let average_bpm = 60.0 * song_beats / seconds;
    stats.radar_voltage = VOLTAGE.value(round_down(average_bpm * most_in_a_window(&rows) as f64 / VOLTAGE_BEATS));

    stats.radar_freeze = FREEZE.value(round_down(10000.0 * held_beats(holds) / song_beats));

    let bpm_change_per_minute = 60.0 * bpm_change(timing, last_beat) / seconds;
    let irregularity = irregularity(&rows) * (1.0 + bpm_change_per_minute * CHAOS_PER_BPM_CHANGE);
    stats.radar_chaos = CHAOS.value(100.0 * irregularity / seconds);
}

/// Most rows in any `VOLTAGE_BEATS` beats: from a row up to (but not including) that many later.
fn most_in_a_window(rows: &[Row]) -> usize {
    let (mut most, mut end) = (0, 0);
    for start in 0..rows.len() {
        while end < rows.len() && rows[end].beat < rows[start].beat + VOLTAGE_BEATS {
            end += 1;
        }
        most = most.max(end - start);
    }
    most
}

/// Beats the holds last, added up. Those that start together count once, as the longest.
fn held_beats(mut holds: Vec<(f64, f64)>) -> f64 {
    holds.sort_by(|a, b| a.0.total_cmp(&b.0));
    holds.chunk_by(|a, b| a.0 == b.0).map(|together| together.iter().map(|&(start, end)| end - start).fold(0.0, f64::max)).sum()
}

/// Each row's arrows and mines, times a weight for its subdivision, over the beats since the
/// row before, added up. Nothing for a row on the beat, nor for the first.
fn irregularity(rows: &[Row]) -> f64 {
    let weight = |sub_div: usize| match sub_div {
        4 => 0.0,
        8 => 2.0,
        16 => 4.0,
        _ => 5.0,
    };
    rows.windows(2).map(|pair| (pair[1].arrows + pair[1].mines) as f64 * weight(pair[1].sub_div) / (pair[1].beat - pair[0].beat)).sum()
}

/// BPM changed up to beat `until`, added up: the difference at each change, and the BPM a stop
/// starts again at (a change on the stop's beat is in that).
fn bpm_change(timing: &Timing, until: f64) -> f64 {
    let changes = timing.bpms.iter().skip(1).map(|&(beat, _)| beat);
    let mut beats: Vec<f64> = changes.chain(timing.stops.iter().map(|&(beat, _)| beat)).filter(|&beat| beat <= until).collect();
    beats.sort_by(f64::total_cmp);
    beats.dedup();
    let before = |beat: f64| timing.bpms.iter().rev().find(|&&(change, _)| change < beat).or(timing.bpms.first()).map_or(0.0, |&(_, bpm)| bpm);
    let stops_at = |beat: f64| timing.stops.iter().any(|&(stop, _)| stop == beat);
    beats.into_iter().map(|beat| if stops_at(beat) { timing.bpm_at(beat) } else { (timing.bpm_at(beat) - before(beat)).abs() }).sum()
}

#[cfg(test)]
mod tests {
    use super::super::tests::stats;
    use super::*;
    use crate::sim_loader::BPMChange;
    use crate::sim_loader::ChartDifficulty::*;
    use crate::sim_loader::Stop;

    /// A measure of 4 beats from rows like "1000".
    fn measure(rows: &[&str]) -> Measure {
        let note = |c: char| match c {
            '1' => Note::Tap,
            '2' => Note::HoldStart,
            '3' => Note::End,
            'M' => Note::Mine,
            _ => Note::Empty,
        };
        let rows = rows.iter().map(|row| {
            let notes: Vec<Note> = row.chars().map(note).collect();
            [notes[0], notes[1], notes[2], notes[3]]
        });
        Measure { num_beats: 4, rows: rows.collect() }
    }

    fn radar(measures: &[Measure], bpms: &[(f64, f64)], stops: &[(f64, f64)]) -> [f64; 5] {
        let bpms: Vec<BPMChange> = bpms.iter().map(|&(beat, bpm)| BPMChange { beat, bpm }).collect();
        let stops: Vec<Stop> = stops.iter().map(|&(beat, seconds)| Stop { beat, seconds }).collect();
        let mut s = ChartStatistics::default();
        count(measures, &Timing::new(&bpms, &stops), &mut s);
        [s.radar_stream, s.radar_voltage, s.radar_air, s.radar_freeze, s.radar_chaos]
    }

    #[test]
    fn scales_each_rate() {
        // (scale, the rate that gives 100, the rate that gives 200)
        for (scale, hundred, two_hundred) in [(STREAM, 300.0, 461.0), (VOLTAGE, 600.0, 1794.0), (AIR, 55.0, 146.0), (FREEZE, 3500.0, 9484.0), (CHAOS, 2000.0, 25605.0)] {
            assert_eq!((scale.value(0.0), scale.value(hundred), scale.value(two_hundred)), (0.0, 100.0, 200.0));
            // The two lines meet at the knee
            assert_eq!(scale.value(hundred - 1e-6), 99.0);
        }
        // Rounded down
        assert_eq!((STREAM.value(299.0), AIR.value(11.0), AIR.value(15.0)), (99.0, 20.0, 27.0));
    }

    #[test]
    fn estimates_a_charts_radar() {
        // (difficulty, stream, voltage, air, freeze, chaos)
        let expected = [(Beginner, [0.0; 5]), (Easy, [14.0, 20.0, 27.0, 64.0, 0.0])];
        for (difficulty, radar) in expected {
            let s = stats("chart_stats.sm", difficulty);
            assert_eq!([s.radar_stream, s.radar_voltage, s.radar_air, s.radar_freeze, s.radar_chaos], radar, "{difficulty:?}");
        }
    }

    #[test]
    fn weighs_notes_off_the_beat_for_chaos() {
        // 8ths at 120 BPM: an 8th (2 / 0.5 beats), a quarter (0), and a jump on an 8th
        // (2 arrows x 2 / 0.5). 12 over the measure's 2 s: 600 a second, in hundredths.
        let eighths = || [measure(&["1000", "0100", "0010", "1001", "0000", "0000", "0000", "0000"])];
        assert_eq!(radar(&eighths(), &[(0.0, 120.0)], &[])[4], 30.0);
        // To 240 BPM at beat 1: 120 changed in 1.25 s, 5760 a minute, so 4.84 times the
        // irregularity. 12 x 4.84 / 1.25 s = 4646.4, past the knee.
        assert_eq!(radar(&eighths(), &[(0.0, 120.0), (1.0, 240.0)], &[])[4], 111.0);
        // A change after the last note doesn't count
        assert_eq!(radar(&eighths(), &[(0.0, 120.0), (2.0, 120.5)], &[])[4], 30.0);
        // A stop of 1 s on beat 1 counts as the BPM: 120 in 3 s, 2400 a minute, 2.6 times.
        // 12 x 2.6 / 3 s = 1040.
        assert_eq!(radar(&eighths(), &[(0.0, 120.0)], &[(1.0, 1.0)])[4], 52.0);
        // A 16th weighs 4 and anything else 5: 4 / 0.25 + 5 / (1/3 - 1/4) = 76, over 2 s
        let mut rows = vec!["0000"; 48];
        (rows[0], rows[3], rows[4]) = ("1000", "0100", "0010");
        assert_eq!(radar(&[measure(&rows)], &[(0.0, 120.0)], &[])[4], ((3800.0_f64 + 21605.0) * 100.0 / 23605.0).floor());
    }

    #[test]
    fn counts_shock_arrows_and_holds() {
        // A measure at 120 BPM: 2 s. A shock arrow is a note (4 in all: 120 a minute), a jump
        // (30 a minute) and, for chaos, 4 arrows.
        let shock = [measure(&["1000", "MMMM", "0010", "0001"])];
        assert_eq!(&radar(&shock, &[(0.0, 120.0)], &[])[..3], [40.0, 20.0, 54.0]);
        let off_beat = [measure(&["1000", "MMMM", "0000", "0000", "0000", "0000", "0000", "0000"])];
        assert_eq!(radar(&off_beat, &[(0.0, 120.0)], &[])[4], (4.0_f64 * 2.0 / 0.5 * 100.0 / 2.0 / 20.0).floor());
        // The song ends with the last measure used, so the empty one after doesn't dilute it:
        // 4 notes in 2 s
        let ended = [measure(&["1000", "0100", "0010", "0001"]), measure(&["0000"; 4])];
        assert_eq!(radar(&ended, &[(0.0, 120.0)], &[])[0], 40.0);

        // Holds of 2 and 3 beats that start together count 3, and one of 1 beat after them adds
        // to it: 4 of 8 beats, 5000 ten-thousandths
        let holds = [measure(&["2200", "0000", "3000", "0300"]), measure(&["0020", "0030", "0000", "0000"])];
        assert_eq!(radar(&holds, &[(0.0, 120.0)], &[])[3], ((5000.0_f64 + 2484.0) * 100.0 / 5984.0).floor());
    }
}
