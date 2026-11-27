//! The statistics that aren't a technique's: counts, rhythm, tempo, density, and what the feet
//! do between steps.

use std::collections::BTreeMap;
use std::ops::Range;

use super::footing::moves::Move;
use super::statistics::{BURST_NOTES, ChartStatistics, NOTE_SUB_DIVS, SUSTAINED_SECONDS, Spread};
use super::technique::speed;
use super::{FINEST_PER_BEAT, Settings, TIME_EPSILON, TimedNote, Timing, arrows, grid_of, row_beat};
use crate::sim_loader::{Measure, Note};

/// Every statistic but the techniques' and the footing's costs. `notes`: the measures', timed.
/// `motions`: the same, with each flam as one note (see `Settings::flam_seconds`), for the
/// density and gap statistics.
pub(super) fn count(
    measures: &[Measure],
    notes: &[TimedNote],
    motions: &[TimedNote],
    timing: &Timing,
    settings: &Settings,
    stats: &mut ChartStatistics,
) {
    let span = count_notes_and_rhythms(measures, notes, stats);
    count_bpms(timing, span, stats);
    count_duration(notes, timing, span, settings.active_gap_seconds, stats);
    count_stops(timing, span, stats);
    count_note_bpms(notes, timing, stats);
    count_run_stops(notes, timing, settings.active_gap_seconds, stats);
    count_arrows(notes, stats);
    count_holds(notes, stats);
    count_notes_per_second(motions, settings, stats);
    count_gaps(motions, stats);
}

/// For each note, the notes from it up to (but not including) `seconds` later, by index.
/// `notes` must be in time order.
fn windows(notes: &[TimedNote], seconds: f64) -> impl Iterator<Item = Range<usize>> {
    let mut end = 0;
    (0..notes.len()).map(move |start| {
        let window_end = notes[start].time + seconds - TIME_EPSILON;
        end = end.max(start);
        while end < notes.len() && notes[end].time < window_end {
            end += 1;
        }
        start..end
    })
}

/// Notes per second in each `Settings::window_seconds`, and the longest run with no gap over
/// `Settings::active_gap_seconds`.
fn count_notes_per_second(notes: &[TimedNote], settings: &Settings, stats: &mut ChartStatistics) {
    let window_seconds = settings.window_seconds;
    let per_second: Vec<f64> = windows(notes, window_seconds).map(|window| window.len() as f64 / window_seconds).collect();
    // Windows with all their second inside the chart
    let last = notes.last().map_or(0.0, |note| note.time);
    let whole = notes.iter().take_while(|note| note.time + window_seconds <= last + TIME_EPSILON).count();
    stats.min_notes_per_second = Spread::of(per_second[..whole].to_vec()).min;

    let spread = Spread::of(per_second);
    stats.mean_notes_per_second = spread.mean;
    stats.median_notes_per_second = spread.median;
    stats.max_notes_per_second = spread.max;
    stats.mode_notes_per_second = spread.mode;
    stats.std_notes_per_second = spread.std;

    for (seconds, sustained) in SUSTAINED_SECONDS.iter().zip(&mut stats.sustained_notes_per_second) {
        let seconds = *seconds as f64;
        *sustained = windows(notes, seconds).map(|window| window.len()).max().unwrap_or(0) as f64 / seconds;
    }
    for run in notes.chunk_by(|a, b| b.time - a.time <= settings.active_gap_seconds + TIME_EPSILON) {
        stats.longest_run_notes = stats.longest_run_notes.max(run.len());
        stats.longest_run_seconds = stats.longest_run_seconds.max(run[run.len() - 1].time - run[0].time);
    }
}

/// `note_costs`: one per note (see `footing::Solution::note_costs`). Per second is in each
/// `window_seconds`.
pub(super) fn count_costs(notes: &[TimedNote], note_costs: Vec<f64>, window_seconds: f64, stats: &mut ChartStatistics) {
    let windows = windows(notes, window_seconds);
    let per_second = Spread::of(windows.map(|window| note_costs[window].iter().sum::<f64>() / window_seconds).collect());
    (stats.mean_cost_per_second, stats.median_cost_per_second, stats.max_cost_per_second) =
        (per_second.mean, per_second.median, per_second.max);
    let per_note = Spread::of(note_costs);
    (stats.mean_cost_per_note, stats.median_cost_per_note, stats.max_cost_per_note) = (per_note.mean, per_note.median, per_note.max);
}

/// `moves`: every segment's (see `footing::Solution::moves`).
pub(super) fn count_feet(moves: &[Move], stats: &mut ChartStatistics) {
    let speeds = Spread::of(moves.iter().map(|m| speed(m.seconds)).collect());
    (stats.mean_foot_speed, stats.median_foot_speed, stats.max_foot_speed) = (speeds.mean, speeds.median, speeds.max);
    stats.foot_travel = moves.iter().map(|m| m.distance).sum();
    let velocities = Spread::of(moves.iter().map(|m| m.distance * speed(m.seconds)).collect());
    (stats.mean_foot_velocity, stats.median_foot_velocity, stats.max_foot_velocity) =
        (velocities.mean, velocities.median, velocities.max);
}

/// The gaps between notes, and the fastest runs of them.
fn count_gaps(notes: &[TimedNote], stats: &mut ChartStatistics) {
    let gaps = Spread::of(notes.windows(2).map(|pair| pair[1].time - pair[0].time).collect());
    stats.mean_note_gap = gaps.mean;
    stats.median_note_gap = gaps.median;
    stats.mode_note_gap = gaps.mode;
    stats.max_note_gap = gaps.max;
    stats.std_note_gap = gaps.std;

    let shortest = |seconds: &mut dyn Iterator<Item = f64>| seconds.fold(f64::INFINITY, f64::min);
    for (burst, fastest) in BURST_NOTES.iter().zip(&mut stats.burst_speeds) {
        let seconds = shortest(&mut notes.windows(*burst).map(|run| run[burst - 1].time - run[0].time));
        *fastest = (burst - 1) as f64 * speed(seconds);
    }
    let jumps: Vec<f64> = notes.iter().filter(|note| arrows(&note.row) >= 2).map(|note| note.time).collect();
    stats.closest_jumps_speed = speed(shortest(&mut jumps.windows(2).map(|pair| pair[1] - pair[0])));
}

fn count_arrows(notes: &[TimedNote], stats: &mut ChartStatistics) {
    let per_panel = [0, 1, 2, 3].map(|arrow| notes.iter().filter(|note| note.steps(arrow)).count());
    let [left, down, up, right] = per_panel;
    let all = left + down + up + right;
    let share = |part: usize, whole: usize| if whole > 0 { part as f64 / whole as f64 } else { 0.5 };
    stats.num_arrows = per_panel;
    stats.right_bias = share(right, left + right);
    stats.up_bias = share(up, down + up);
    stats.vertical_bias = share(down + up, all);
    if all > 0 {
        let (most, fewest) = (per_panel.into_iter().max().unwrap_or(0), per_panel.into_iter().min().unwrap_or(0));
        stats.lopsidedness_by_max = (most - fewest) as f64 / all as f64;
        let mean = all as f64 / 4.0;
        stats.lopsidedness_by_mean = per_panel.iter().map(|&count| (count as f64 - mean).abs()).sum::<f64>() / all as f64;
    }
}

fn count_holds(notes: &[TimedNote], stats: &mut ChartStatistics) {
    let starts_one = |note: &&TimedNote| note.row.iter().any(|&arrow| matches!(arrow, Note::HoldStart | Note::RollStart));
    stats.num_hold_notes = notes.iter().filter(starts_one).count();
    let seconds = notes.iter().flat_map(|note| note.hold_ends.iter().flatten().map(|&end| end - note.time));
    let holds = Spread::of(seconds.collect());
    stats.total_hold_seconds = holds.total;
    stats.mean_hold_seconds = holds.mean;
    stats.min_hold_seconds = holds.min;
    stats.max_hold_seconds = holds.max;
}

/// `span`: beats of the first and last objects, from `count_notes_and_rhythms`.
/// `active_gap_seconds`: longer gaps between notes are breaks, not active time.
fn count_duration(notes: &[TimedNote], timing: &Timing, span: Option<(f64, f64)>, active_gap_seconds: f64, stats: &mut ChartStatistics) {
    let Some((first, last)) = span else { return };
    let (start, end) = (timing.time_at(first), timing.time_at(last));
    stats.chart_start = start;
    stats.chart_dur = end - start;
    if stats.chart_dur <= 0.0 {
        return;
    }

    // Active intervals: note pairs with no break between them, and holds
    let mut active: Vec<(f64, f64)> = notes
        .windows(2)
        .filter(|pair| pair[1].time - pair[0].time <= active_gap_seconds + TIME_EPSILON)
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

/// `span` as for `count_duration`, which must come first. A stop on the last object's beat
/// only delays what's after it, so it isn't counted.
fn count_stops(timing: &Timing, span: Option<(f64, f64)>, stats: &mut ChartStatistics) {
    let Some((first, last)) = span else { return };
    let inside: Vec<f64> = timing.stops.iter().filter(|&&(beat, _)| first <= beat && beat < last).map(|&(_, seconds)| seconds).collect();
    stats.num_stops = inside.len();
    let stops = Spread::of(inside);
    stats.total_stop_seconds = stops.total;
    stats.min_stop_seconds = stops.min;
    stats.max_stop_seconds = stops.max;
    if stats.chart_dur > 0.0 {
        stats.stop_ratio = stops.total / stats.chart_dur;
    }
}

/// The stops between two notes `active_gap_seconds` apart at most, once the stops between
/// them are taken out. A stop on a note's beat comes after the note.
fn count_run_stops(notes: &[TimedNote], timing: &Timing, active_gap_seconds: f64, stats: &mut ChartStatistics) {
    for pair in notes.windows(2) {
        let between: Vec<f64> =
            timing.stops.iter().filter(|&&(beat, _)| pair[0].beat <= beat && beat < pair[1].beat).map(|&(_, seconds)| seconds).collect();
        let stopped: f64 = between.iter().sum();
        if pair[1].time - pair[0].time - stopped <= active_gap_seconds + TIME_EPSILON {
            stats.num_run_stops += between.len();
            stats.run_stop_seconds += stopped;
        }
    }
}

/// The BPM at each note, and how it changes from one note to the next.
fn count_note_bpms(notes: &[TimedNote], timing: &Timing, stats: &mut ChartStatistics) {
    let bpms: Vec<f64> = notes.iter().map(|note| timing.bpm_at(note.beat)).collect();
    for pair in bpms.windows(2).filter(|pair| pair[0] > 0.0 && pair[1] > 0.0 && pair[0] != pair[1]) {
        let octaves = (pair[1] / pair[0]).log2();
        stats.note_bpm_travel += octaves.abs();
        if octaves > 0.0 {
            stats.num_note_speedups += 1;
            stats.max_note_speedup = stats.max_note_speedup.max(octaves);
        } else {
            stats.num_note_slowdowns += 1;
            stats.max_note_slowdown = stats.max_note_slowdown.max(-octaves);
        }
    }
    let bpms = Spread::of(bpms);
    (stats.mean_note_bpm, stats.median_note_bpm, stats.min_note_bpm, stats.max_note_bpm) = (bpms.mean, bpms.median, bpms.min, bpms.max);
}

/// `segments`: each one's notes (see `segments`). Gaps between segments aren't counted.
pub(super) fn count_rhythms(segments: &[&[TimedNote]], stats: &mut ChartStatistics) {
    // Beats from a note to the next, in millionths, to how many times
    let mut gaps: BTreeMap<i64, usize> = BTreeMap::new();
    for pair in segments.iter().flat_map(|notes| notes.windows(2)) {
        *gaps.entry(((pair[1].beat - pair[0].beat) * 1e6).round() as i64).or_default() += 1;
    }
    let all = gaps.values().sum::<usize>() as f64;
    let entropy = |&count: &usize| {
        let share = count as f64 / all;
        -share * share.log2()
    };
    stats.rhythm_entropy = gaps.values().map(entropy).sum();
}

/// The key with the highest count. Ties go to the smallest. 0 if there are none.
fn most_common(counts: &BTreeMap<usize, usize>) -> usize {
    // `max_by_key` keeps the last maximum, so `rev` sends ties to the smallest.
    counts.iter().rev().max_by_key(|&(_, &count)| count).map_or(0, |(&key, _)| key)
}

/// `notes`: the measures', in order. Returns the beats of the first and last non-empty rows, or
/// None for an empty chart.
fn count_notes_and_rhythms(measures: &[Measure], notes: &[TimedNote], stats: &mut ChartStatistics) -> Option<(f64, f64)> {
    // Subdivision -> number of measures, and of notes
    let mut sub_div_measures: BTreeMap<usize, usize> = BTreeMap::new();
    let mut sub_div_notes: BTreeMap<usize, usize> = BTreeMap::new();
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
            let note_sub_div = note_sub_div(&notes[stats.num_notes]);
            stats.num_notes += 1;
            *sub_div_notes.entry(note_sub_div).or_default() += 1;
            sub_div = Some(sub_div.map_or(note_sub_div, |s| lcm(s, note_sub_div)));
        }
        if let Some(s) = sub_div {
            *sub_div_measures.entry(s).or_default() += 1;
        }
        measure_start += measure.num_beats;
    }

    stats.chart_num_measures = first_used.map_or(0, |(first, _)| last_used.0 - first + 1);
    stats.num_sub_divs = sub_div_measures.len();
    stats.mode_sub_div = most_common(&sub_div_measures);
    stats.num_adv_sub_divs = sub_div_measures
        .iter()
        .filter(|&(&s, _)| !s.is_power_of_two())
        .map(|(_, &count)| count)
        .sum();

    for (&sub_div, &notes) in &sub_div_notes {
        match NOTE_SUB_DIVS.iter().position(|&listed| listed == sub_div) {
            Some(index) => stats.sub_div_notes[index] = notes,
            None => stats.other_sub_div_notes += notes,
        }
    }
    stats.num_note_sub_divs = sub_div_notes.len();
    stats.mode_note_sub_div = most_common(&sub_div_notes);
    stats.finest_note_sub_div = sub_div_notes.keys().next_back().copied().unwrap_or(0);
    if stats.num_notes > 0 {
        let notes = stats.num_notes as f64;
        stats.mean_note_sub_div = sub_div_notes.iter().map(|(&s, &count)| (s * count) as f64).sum::<f64>() / notes;
        let advanced: usize = sub_div_notes.iter().filter(|&(&s, _)| !s.is_power_of_two()).map(|(_, &count)| count).sum();
        stats.adv_sub_div_note_ratio = advanced as f64 / notes;
        stats.offbeat_note_ratio = 1.0 - sub_div_notes.get(&4).copied().unwrap_or(0) as f64 / notes;
    }

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
    let bpms = Spread::of(seconds_at.iter().map(|&(bpm, _)| bpm).collect());
    (stats.min_bpm, stats.max_bpm) = if seconds_at.is_empty() { (stats.mode_bpm, stats.mode_bpm) } else { (bpms.min, bpms.max) };
    stats.bpm_range = stats.max_bpm - stats.min_bpm;
}

/// The note's subdivision, as in "16th notes": 4 on a beat, 8 halfway between two, 12 a third
/// of the way, and so on, by its `heard_beat`. `row_sub_div` unless a stop moved the beat.
fn note_sub_div(note: &TimedNote) -> usize {
    4 * grid_of(note.heard_beat).unwrap_or(FINEST_PER_BEAT)
}

/// The row's subdivision as written (see `note_sub_div`).
pub(super) fn row_sub_div(measure: &Measure, row: usize) -> usize {
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

#[cfg(test)]
mod tests {
    use super::super::tests::{assert_close, stats};
    use super::*;
    use crate::chart_analyzer::Technique;
    use crate::sim_loader::ChartDifficulty::*;

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
            let mut measure = Measure { num_beats: 4, rows: vec![[Note::Empty; 4]; rows] };
            assert_eq!(row_sub_div(&measure, row), expected, "row {row} of a {rows}-row measure");
            // A note there, in the second measure, has the same
            measure.rows[row][0] = Note::Tap;
            let empty = Measure { num_beats: 4, rows: vec![[Note::Empty; 4]; 4] };
            let notes = super::super::to_timed_notes(&[empty, measure], &Timing::new(&[], &[]));
            assert_eq!(note_sub_div(&notes[0]), expected, "a note on row {row} of a {rows}-row measure");
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
                (s.num_notes, s.technique(Technique::Jump).count, s.num_holds, s.num_rolls, s.num_mines),
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
    fn costs_the_footing_per_note_and_per_second() {
        // (difficulty, (mean, median, max) per note, the same per second)
        let nothing = (0.0, 0.0, 0.0);
        let expected = [
            (Beginner, nothing, nothing),
            (Easy, nothing, nothing),
            (Medium, (5.5 / 6.0, 0.0, 2.75), (11.0 / 6.0, 2.75, 2.75)),
            (Hard, nothing, nothing),
            (Challenge, (15.5 / 16.0, 0.0, 6.5), (51.0 / 16.0, 2.25, 9.0)),
        ];
        for (difficulty, per_note, per_second) in expected {
            let s = stats("footing_stats.sm", difficulty);
            let costs = [
                ("mean cost per note", s.mean_cost_per_note, per_note.0),
                ("median cost per note", s.median_cost_per_note, per_note.1),
                ("max cost per note", s.max_cost_per_note, per_note.2),
                ("mean cost per second", s.mean_cost_per_second, per_second.0),
                ("median cost per second", s.median_cost_per_second, per_second.1),
                ("max cost per second", s.max_cost_per_second, per_second.2),
            ];
            for (what, actual, expected) in costs {
                assert_close(actual, expected, &format!("{difficulty:?} {what}"));
            }
        }
    }

    #[test]
    fn measures_duration_and_activity() {
        // (difficulty, duration, active ratio)
        let expected = [
            (Beginner, 0.0, 0.0),
            (Easy, 7.0, 1.0 / 2.0),
            (Medium, 5.5, 4.0 / 11.0),
            (Hard, 4.75, 6.0 / 19.0),
            (Challenge, 10.0, 7.0 / 20.0),
        ];
        for (difficulty, duration, active) in expected {
            let s = stats("footing_stats.sm", difficulty);
            assert_close(s.chart_dur, duration, &format!("{difficulty:?} duration"));
            assert_close(s.chart_active_ratio, active, &format!("{difficulty:?} active ratio"));
        }
    }

    #[test]
    fn counts_arrows_per_panel() {
        // (difficulty, arrows on Left, Down, Up and Right, lopsidedness by max and by mean,
        // right, up and vertical bias)
        let expected = [
            (Beginner, [0, 0, 0, 0], (0.0, 0.0), (0.5, 0.5, 0.5)),
            (Easy, [8, 5, 3, 4], (0.25, 0.3), (1.0 / 3.0, 3.0 / 8.0, 0.4)),
            (Medium, [6, 6, 6, 6], (0.0, 0.0), (0.5, 0.5, 0.5)),
        ];
        for (difficulty, arrows, lopsidedness, biases) in expected {
            let s = stats("chart_stats.sm", difficulty);
            assert_eq!(s.num_arrows, arrows, "{difficulty:?}");
            assert_close(s.lopsidedness_by_max, lopsidedness.0, &format!("{difficulty:?} lopsidedness by max"));
            assert_close(s.lopsidedness_by_mean, lopsidedness.1, &format!("{difficulty:?} lopsidedness by mean"));
            assert_close(s.right_bias, biases.0, &format!("{difficulty:?} right bias"));
            assert_close(s.up_bias, biases.1, &format!("{difficulty:?} up bias"));
            assert_close(s.vertical_bias, biases.2, &format!("{difficulty:?} vertical bias"));
        }
    }

    #[test]
    fn counts_each_notes_subdivision() {
        // (difficulty, notes on 4ths, 8ths, 12ths, 16ths, 24ths, 32nds, 48ths and 64ths,
        // (distinct, mode, finest), mean, advanced share)
        let expected = [
            (Beginner, [0; 8], (0, 0, 0), 0.0, 0.0),
            (Easy, [14, 0, 0, 0, 0, 0, 0, 0], (1, 4, 4), 4.0, 0.0),
            (Medium, [8, 8, 0, 8, 0, 0, 0, 0], (3, 4, 16), 224.0 / 24.0, 0.0),
            (Hard, [43, 15, 14, 8, 2, 1, 0, 0], (6, 4, 32), 668.0 / 83.0, 16.0 / 83.0),
        ];
        for (difficulty, notes, (distinct, mode, finest), mean, advanced) in expected {
            let s = stats("chart_stats.sm", difficulty);
            assert_eq!((s.sub_div_notes, s.other_sub_div_notes), (notes, 0), "{difficulty:?}");
            assert_eq!(
                (s.num_note_sub_divs, s.mode_note_sub_div, s.finest_note_sub_div),
                (distinct, mode, finest),
                "{difficulty:?}: (distinct, mode, finest)"
            );
            assert_close(s.mean_note_sub_div, mean, &format!("{difficulty:?} mean subdivision"));
            assert_close(s.adv_sub_div_note_ratio, advanced, &format!("{difficulty:?} advanced share"));
        }

        // 20ths aren't listed: a tap on the beat and two off it in a 20-row measure
        let mut rows = vec![[Note::Empty; 4]; 20];
        for row in [0, 1, 3] {
            rows[row][0] = Note::Tap;
        }
        let mut s = ChartStatistics::default();
        let measures = [Measure { num_beats: 4, rows }];
        count_notes_and_rhythms(&measures, &super::super::to_timed_notes(&measures, &Timing::new(&[], &[])), &mut s);
        assert_eq!((s.sub_div_notes[0], s.other_sub_div_notes), (1, 2));
        assert_eq!((s.num_note_sub_divs, s.mode_note_sub_div, s.finest_note_sub_div), (2, 20, 20));
    }

    #[test]
    fn measures_holds() {
        // (difficulty, notes starting a hold or roll, their seconds: (total, mean, min, max))
        let expected = [(Beginner, 0, (0.0, 0.0, 0.0, 0.0)), (Easy, 4, (6.0, 1.2, 1.0, 1.5)), (Medium, 0, (0.0, 0.0, 0.0, 0.0))];
        for (difficulty, notes, (total, mean, min, max)) in expected {
            let s = stats("chart_stats.sm", difficulty);
            assert_eq!(s.num_hold_notes, notes, "{difficulty:?}");
            assert_close(s.total_hold_seconds, total, &format!("{difficulty:?} total hold"));
            assert_close(s.mean_hold_seconds, mean, &format!("{difficulty:?} mean hold"));
            assert_close(s.min_hold_seconds, min, &format!("{difficulty:?} shortest hold"));
            assert_close(s.max_hold_seconds, max, &format!("{difficulty:?} longest hold"));
        }
    }

    #[test]
    fn finds_the_bpm_range_and_the_start() {
        // (difficulty, min BPM, max BPM, seconds to the first object)
        let expected = [
            (Beginner, 0.0, 0.0, 0.0),
            (Easy, 240.0, 240.0, 4.6),
            (Medium, 120.0, 240.0, 16.0 + 13.0 / 30.0),
            (Hard, 60.0, 240.0, 2.1),
            (Challenge, 60.0, 240.0, 1.6),
        ];
        for (difficulty, min, max, start) in expected {
            let s = stats("bpm_stats.sm", difficulty);
            assert_eq!((s.min_bpm, s.max_bpm, s.bpm_range), (min, max, max - min), "{difficulty:?}: (min, max, range)");
            assert_close(s.chart_start, start, &format!("{difficulty:?} start"));
        }
    }

    #[test]
    fn counts_stops() {
        // (difficulty, stops, total, shortest, longest, share of the chart)
        let expected = [
            (Beginner, 0, 0.0, 0.0, 0.0, 0.0),
            (Easy, 2, 1.25, 0.5, 0.75, 15.0 / 91.0),
            (Hard, 0, 0.0, 0.0, 0.0, 0.0),
        ];
        for (difficulty, stops, total, min, max, ratio) in expected {
            let s = stats("nps_stats.sm", difficulty);
            assert_eq!(s.num_stops, stops, "{difficulty:?}");
            assert_close(s.total_stop_seconds, total, &format!("{difficulty:?} total stop"));
            assert_close(s.min_stop_seconds, min, &format!("{difficulty:?} shortest stop"));
            assert_close(s.max_stop_seconds, max, &format!("{difficulty:?} longest stop"));
            assert_close(s.stop_ratio, ratio, &format!("{difficulty:?} stop ratio"));
        }

        // On the first object's beat: counted. On the last's: not.
        let stop = |beat: f64, seconds: f64| crate::sim_loader::Stop { beat, seconds };
        let timing = Timing::new(&[], &[stop(4.0, 1.0), stop(6.0, 3.0), stop(8.0, 2.0)]);
        let mut s = ChartStatistics { chart_dur: 16.0, ..Default::default() };
        count_stops(&timing, Some((4.0, 8.0)), &mut s);
        assert_eq!((s.num_stops, s.total_stop_seconds, s.min_stop_seconds, s.max_stop_seconds, s.stop_ratio), (2, 4.0, 1.0, 3.0, 0.25));
    }

    #[test]
    fn measures_rhythm() {
        let entropy = |gaps: &[usize]| {
            let all = gaps.iter().sum::<usize>() as f64;
            gaps.iter().map(|&count| -(count as f64 / all) * (count as f64 / all).log2()).sum::<f64>()
        };
        // (difficulty, off-beat share, how many gaps of each length)
        let expected: [(_, f64, &[usize]); 4] = [
            (Beginner, 0.0, &[]),
            (Easy, 11.0 / 23.0, &[8, 5, 4, 3, 1, 1]),
            (Medium, 1.0 / 3.0, &[8, 3]),
            (Hard, 0.0, &[13, 1]),
        ];
        for (difficulty, offbeat, gaps) in expected {
            let s = stats("rhythm_stats.sm", difficulty);
            assert_close(s.offbeat_note_ratio, offbeat, &format!("{difficulty:?} off-beat share"));
            assert_close(s.rhythm_entropy, entropy(gaps), &format!("{difficulty:?} rhythm entropy"));
        }
        // As the fixture's header has them
        assert!((stats("rhythm_stats.sm", Easy).rhythm_entropy - 2.2610).abs() < 5e-5);
        assert!((stats("rhythm_stats.sm", Medium).rhythm_entropy - 0.8454).abs() < 5e-5);

        // Two gaps of a beat and one of a half. The 4 beats between the segments aren't a gap.
        let tap = |beat: f64| TimedNote {
            time: beat,
            beat,
            heard_beat: beat,
            row: [Note::Tap, Note::Empty, Note::Empty, Note::Empty],
            hold_ends: [None; 4],
        };
        let (first, second) = ([tap(0.0), tap(1.0), tap(2.0)], [tap(6.0), tap(6.5)]);
        let mut s = ChartStatistics::default();
        count_rhythms(&[&first, &second], &mut s);
        assert_close(s.rhythm_entropy, entropy(&[2, 1]), "two segments");
        let mut s = ChartStatistics::default();
        count_rhythms(&[&first[..1], &second[..1]], &mut s);
        assert_eq!(s.rhythm_entropy, 0.0, "no gaps");
    }

    #[test]
    fn finds_the_tempo_at_the_notes() {
        // (difficulty, BPM at the notes: (mean, median, min, max), (speedups, slowdowns),
        // octaves: (travel, largest speedup, largest slowdown))
        let steady = |bpm: f64| ((bpm, bpm, bpm, bpm), (0, 0), (0.0, 0.0, 0.0));
        let expected = [
            (Beginner, steady(0.0)),
            (Easy, steady(120.0)),
            (Hard, ((168.0, 120.0, 60.0, 480.0), (3, 1), (6.0, 2.0, 2.0))),
            (Challenge, steady(120.0)),
        ];
        for (difficulty, (bpms, changes, octaves)) in expected {
            let s = stats("rhythm_stats.sm", difficulty);
            assert_eq!((s.mean_note_bpm, s.median_note_bpm, s.min_note_bpm, s.max_note_bpm), bpms, "{difficulty:?}: (mean, median, min, max)");
            assert_eq!((s.num_note_speedups, s.num_note_slowdowns), changes, "{difficulty:?}: (speedups, slowdowns)");
            assert_close(s.note_bpm_travel, octaves.0, &format!("{difficulty:?} travel"));
            assert_close(s.max_note_speedup, octaves.1, &format!("{difficulty:?} largest speedup"));
            assert_close(s.max_note_slowdown, octaves.2, &format!("{difficulty:?} largest slowdown"));
        }
        // The notes meet 4 of Hard's changes: one has no note in it, one is on the last note
        assert_eq!(stats("rhythm_stats.sm", Hard).num_bpm_changes, 5);
    }

    #[test]
    fn counts_stops_in_runs() {
        // (difficulty, stops, their seconds, stops in a run, their seconds)
        let expected = [(Beginner, 0, 0.0, 0, 0.0), (Easy, 0, 0.0, 0, 0.0), (Hard, 0, 0.0, 0, 0.0), (Challenge, 6, 4.25, 4, 1.75)];
        for (difficulty, stops, seconds, in_runs, run_seconds) in expected {
            let s = stats("rhythm_stats.sm", difficulty);
            assert_eq!((s.num_stops, s.num_run_stops), (stops, in_runs), "{difficulty:?}: (stops, in a run)");
            assert_close(s.total_stop_seconds, seconds, &format!("{difficulty:?} stopped"));
            assert_close(s.run_stop_seconds, run_seconds, &format!("{difficulty:?} stopped in runs"));
        }
    }

    #[test]
    fn spreads_notes_per_second() {
        // (difficulty, mode, std, min)
        let expected = [
            (Beginner, 0.0, 0.0, 0.0),
            (Easy, 2.0, 74.0_f64.sqrt() / 13.0, 1.0),
            (Medium, 1.0, 0.0, 0.0),
            (Hard, 6.0, 655.0_f64.sqrt() / 16.0, 6.0),
            (Challenge, 4.0, 2080.0_f64.sqrt() / 23.0, 2.0),
        ];
        for (difficulty, mode, std, min) in expected {
            let s = stats("nps_stats.sm", difficulty);
            assert_close(s.mode_notes_per_second, mode, &format!("{difficulty:?} mode"));
            assert_close(s.std_notes_per_second, std, &format!("{difficulty:?} std"));
            assert_close(s.min_notes_per_second, min, &format!("{difficulty:?} min"));
        }
    }

    #[test]
    fn finds_sustained_density_and_the_longest_run() {
        // (difficulty, most notes in 5, 10 and 30 s, the longest run's notes and seconds)
        let expected = [
            (Beginner, [0, 0, 0], 0, 0.0),
            (Easy, [9, 13, 13], 10, 16.0 / 3.0),
            (Medium, [1, 1, 1], 1, 0.0),
            (Hard, [16, 16, 16], 16, 2.5),
        ];
        for (difficulty, most, run_notes, run_seconds) in expected {
            let s = stats("nps_stats.sm", difficulty);
            for ((seconds, actual), most) in SUSTAINED_SECONDS.iter().zip(s.sustained_notes_per_second).zip(most) {
                assert_close(actual, most as f64 / *seconds as f64, &format!("{difficulty:?} notes per second over {seconds} s"));
            }
            assert_eq!(s.longest_run_notes, run_notes, "{difficulty:?}");
            assert_close(s.longest_run_seconds, run_seconds, &format!("{difficulty:?} longest run"));
        }
    }

    #[test]
    fn measures_the_feets_moves() {
        // (difficulty, each move's speed, the feet's travel, each move's velocity)
        let diagonal = 2.0_f64.sqrt();
        let expected = [
            (Beginner, 0.0, 0.0, 0.0),
            (Easy, 1.0, 3.0 * diagonal, diagonal),
            (Medium, 2.0, 2.0 * diagonal, 2.0 * diagonal),
        ];
        for (difficulty, speed, travel, velocity) in expected {
            let s = stats("footing_stats.sm", difficulty);
            let speeds = (s.mean_foot_speed, s.median_foot_speed, s.max_foot_speed);
            let velocities = (s.mean_foot_velocity, s.median_foot_velocity, s.max_foot_velocity);
            for (what, actual, expected) in [("speed", speeds, speed), ("velocity", velocities, velocity)] {
                assert_close(actual.0, expected, &format!("{difficulty:?} mean foot {what}"));
                assert_close(actual.1, expected, &format!("{difficulty:?} median foot {what}"));
                assert_close(actual.2, expected, &format!("{difficulty:?} max foot {what}"));
            }
            assert_close(s.foot_travel, travel, &format!("{difficulty:?} foot travel"));
        }
    }

    #[test]
    fn counts_held_steps_and_fast_techniques() {
        // One note under a hold, 0.5 s after the hold's head
        let held = stats("chart_stats.sm", Easy);
        let held = held.technique(Technique::HeldStep);
        assert_eq!((held.count, held.max_speed, held.faster), (1, 2.0, [1, 0, 0, 0]));
        assert_eq!(stats("chart_stats.sm", Medium).technique(Technique::HeldStep).count, 0);
        // Doublesteps at 2 a second, and at 4: at least 2, 4, 6 and 10 a second
        assert_eq!(stats("footing_stats.sm", Medium).technique(Technique::Doublestep).faster, [2, 0, 0, 0]);
        assert_eq!(stats("footing_stats.sm", Challenge).technique(Technique::Doublestep).faster, [2, 2, 0, 0]);
    }

    #[test]
    fn times_gaps_and_bursts() {
        // (difficulty, gaps: (mean, median, mode, max, std), fastest 2, 3, 5, 7 and 15 notes,
        // closest jumps)
        let sixth = 1.0 / 6.0;
        let easy_std = (5.9375_f64 / 12.0 - (91.0_f64 / 144.0).powi(2)).sqrt();
        let challenge_std = (2.125_f64 / 22.0 - (3.0_f64 / 11.0).powi(2)).sqrt();
        let expected = [
            (Beginner, (0.0, 0.0, 0.0, 0.0, 0.0), [0.0; 5], 0.0),
            (Easy, (91.0 / 144.0, 0.5, 0.5, 1.25, easy_std), [4.0, 4.0, 2.0, 1.8, 0.0], 0.0),
            (Medium, (0.0, 0.0, 0.0, 0.0, 0.0), [0.0; 5], 0.0),
            (Hard, (sixth, sixth, sixth, sixth, 0.0), [6.0; 5], 3.0),
            (Challenge, (3.0 / 11.0, 0.25, 0.125, 0.5, challenge_std), [8.0, 8.0, 8.0, 8.0, 5.6], 2.0),
        ];
        for (difficulty, (mean, median, mode, max, std), bursts, jumps) in expected {
            let s = stats("nps_stats.sm", difficulty);
            assert_close(s.mean_note_gap, mean, &format!("{difficulty:?} mean gap"));
            assert_close(s.median_note_gap, median, &format!("{difficulty:?} median gap"));
            assert_close(s.mode_note_gap, mode, &format!("{difficulty:?} mode gap"));
            assert_close(s.max_note_gap, max, &format!("{difficulty:?} max gap"));
            assert_close(s.std_note_gap, std, &format!("{difficulty:?} std of the gaps"));
            for ((notes, actual), expected) in BURST_NOTES.iter().zip(s.burst_speeds).zip(bursts) {
                assert_close(actual, expected, &format!("{difficulty:?} fastest {notes} notes"));
            }
            assert_close(s.closest_jumps_speed, jumps, &format!("{difficulty:?} closest jumps"));
        }
    }
}
