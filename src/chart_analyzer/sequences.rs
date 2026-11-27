//! Techniques found from the notes alone, whatever the feet do.

use super::technique::{Occurrence, Technique, even, even_runs, gaps};
use super::{BEAT_EPSILON, Settings, TIME_EPSILON, TimedNote, arrows, grid_of, steps_on};

/// The repeated arrows, jumps, held steps, syncopations and rhythm changes of `segment`, each
/// on its note, and its streams, each on its first: a stream is a run of evenly spaced notes as
/// long as it goes, so a change of rhythm ends it, and the next may start on its last note (see
/// `Settings::min_stream_notes` and `Settings::even_tolerance`). A held step is under a hold
/// from earlier in the segment. A syncopation is a note off the beat that the segment doesn't
/// follow with a note on the next stronger position (see `stronger_grid`), by the notes'
/// `heard_beat`. The rhythm changes on a note whose gaps before and after aren't even.
pub fn find(segment: &[TimedNote], settings: &Settings) -> Vec<Occurrence> {
    let mut found = Vec::new();
    // When the last of the holds so far ends
    let mut held_until = f64::NEG_INFINITY;
    for i in 0..segment.len() {
        if i > 0 && shares_an_arrow(&segment[i - 1], &segment[i]) {
            found.push(Occurrence::at(Technique::RepeatedArrow, segment, i));
        }
        if arrows(&segment[i].row) >= 2 {
            found.push(Occurrence::at(Technique::Jump, segment, i));
        }
        if held_until > segment[i].time + TIME_EPSILON {
            found.push(Occurrence::at(Technique::HeldStep, segment, i));
        }
        held_until = segment[i].hold_ends.iter().flatten().fold(held_until, |until, &end| until.max(end));
        if syncopated(segment, i) {
            found.push(Occurrence::at(Technique::Syncopation, segment, i));
        }
        if changes_rhythm(segment, i, settings.even_tolerance) {
            found.push(Occurrence::at(Technique::RhythmChange, segment, i));
        }
    }
    let apart = |i: usize| segment[i].time - segment[i - 1].time > TIME_EPSILON;
    for (first, notes) in even_runs(segment, settings.min_stream_notes, settings.even_tolerance, apart, |_, _| true) {
        found.push(Occurrence::over(Technique::Stream, segment, first, notes));
    }
    found
}

fn shares_an_arrow(a: &TimedNote, b: &TimedNote) -> bool {
    a.row.iter().zip(&b.row).any(|(&x, &y)| steps_on(x) && steps_on(y))
}

/// Positions per beat of the grid one step stronger than `beat`'s own: half as many, or the
/// beats alone for a grid that doesn't halve. So a 16th leads to an 8th, a 24th to a 12th, and
/// an 8th or a 12th to a beat. None on a beat, or off every grid (see `grid_of`).
fn stronger_grid(beat: f64) -> Option<usize> {
    match grid_of(beat)? {
        1 => None,
        per_beat if per_beat % 2 == 0 => Some(per_beat / 2),
        _ => Some(1),
    }
}

/// Whether note `i` is off the beat and no later note of `segment` is on the next position of
/// its `stronger_grid`.
fn syncopated(segment: &[TimedNote], i: usize) -> bool {
    let Some(per_beat) = stronger_grid(segment[i].heard_beat) else { return false };
    let per_beat = per_beat as f64;
    let resolution = ((segment[i].heard_beat * per_beat).floor() + 1.0) / per_beat;
    let later = segment[i + 1..].iter().take_while(|note| note.heard_beat < resolution + BEAT_EPSILON);
    !later.last().is_some_and(|note| note.heard_beat > resolution - BEAT_EPSILON)
}

/// Whether the gaps before and after note `i` aren't even (see `even`). Not on the first or
/// last note of `segment`, with one gap.
fn changes_rhythm(segment: &[TimedNote], i: usize, tolerance: f64) -> bool {
    let (before, after) = gaps(segment, i);
    before.is_finite() && after.is_finite() && !even(before.min(after), before.max(after), tolerance)
}

#[cfg(test)]
mod tests {
    use super::super::footing::tests::segment;
    use super::super::{SETTINGS, TEST_COSTS, foot_arrows};
    use super::*;
    use crate::sim_loader::Note;

    /// (note, notes, speed) of each `technique` found.
    fn found(tokens: &str, technique: Technique) -> Vec<(usize, usize, f64)> {
        found_in(&segment(tokens), technique)
    }

    fn found_in(notes: &[TimedNote], technique: Technique) -> Vec<(usize, usize, f64)> {
        find(notes, &SETTINGS)
            .into_iter()
            .filter(|occurrence| occurrence.technique == technique)
            .map(|occurrence| (occurrence.note, occurrence.notes, occurrence.speed))
            .collect()
    }

    #[test]
    fn finds_repeated_arrows() {
        // 4 on one arrow are 3, at 8 a second
        assert_eq!(found("L L L L", Technique::RepeatedArrow), [(1, 1, 8.0), (2, 1, 8.0), (3, 1, 8.0)]);
        // Slow ones count, at their speed
        assert_eq!(found("L . . . L", Technique::RepeatedArrow), [(1, 1, 2.0)]);
        // A jump repeated is one, and so is one arrow of a jump
        assert_eq!(found("LR LR", Technique::RepeatedArrow), [(1, 1, 8.0)]);
        assert_eq!(found("LR L U", Technique::RepeatedArrow), [(1, 1, 8.0)]);
        // A hold head counts, its end doesn't
        assert_eq!(found("D+ D- D", Technique::RepeatedArrow), [(1, 1, 4.0)]);
        assert_eq!(found("L D L D", Technique::RepeatedArrow), []);
    }

    #[test]
    fn finds_jumps() {
        // The first has no note before it
        assert_eq!(found("LR U . DU", Technique::Jump), [(0, 1, 0.0), (2, 1, 4.0)]);
        assert_eq!(found("L D U R", Technique::Jump), []);
    }

    #[test]
    fn finds_held_steps() {
        // Left held from the first note to 0.375 s: the two notes under it
        assert_eq!(found("L+ D R L- D", Technique::HeldStep), [(1, 1, 8.0), (2, 1, 8.0)]);
        // Two holds: under the longer after the shorter ends
        assert_eq!(found("L+ U+ . U- D . L- R", Technique::HeldStep), [(1, 1, 8.0), (2, 1, 1.0 / 0.375)]);
        // A hold with no end written isn't one
        assert_eq!(found("L+ D U R", Technique::HeldStep), []);
        // Left held under Up. Not under the Right it starts with, nor the Down it ends with.
        let mixed = foot_arrows("[L+R]U[L-D]R", &TEST_COSTS, 1.0).unwrap().notes;
        let rows: Vec<[Note; 4]> = mixed.iter().map(|note| note.row).collect();
        assert_eq!(rows[0], [Note::HoldStart, Note::Empty, Note::Empty, Note::Tap]);
        assert_eq!(rows[2], [Note::End, Note::Tap, Note::Empty, Note::Empty]);
        assert_eq!(found_in(&mixed, Technique::HeldStep), [(1, 1, 8.0)]);
    }

    /// Taps on `beats`, at 120 BPM.
    fn on_beats(beats: &[f64]) -> Vec<TimedNote> {
        let tap = [Note::Tap, Note::Empty, Note::Empty, Note::Empty];
        beats.iter().map(|&beat| TimedNote { time: beat * 0.5, beat, heard_beat: beat, row: tap, hold_ends: [None; 4] }).collect()
    }

    /// Checks the (note, speed) of each `technique` found in taps on `beats`.
    fn assert_found_on(beats: &[f64], technique: Technique, expected: &[(usize, f64)]) {
        let found = found_in(&on_beats(beats), technique);
        let notes = |found: &mut dyn Iterator<Item = usize>| found.collect::<Vec<usize>>();
        assert_eq!(notes(&mut found.iter().map(|found| found.0)), notes(&mut expected.iter().map(|expected| expected.0)), "{beats:?}");
        for (found, expected) in found.iter().zip(expected) {
            assert!((found.2 - expected.1).abs() < 1e-9, "{beats:?}: note {} at {}, not {}", found.0, found.2, expected.1);
        }
    }

    #[test]
    fn finds_the_next_stronger_grid() {
        let third = 1.0 / 3.0;
        // (beat, positions per beat of the grid its note leads to)
        let cases = [
            (0.0, None),
            (7.0, None),
            (0.5, Some(1)),
            (2.25, Some(2)),
            (2.75, Some(2)),
            (third, Some(1)),
            (5.0 + 2.0 * third, Some(1)),
            // 24ths to 12ths, 32nds to 16ths, 48ths to 24ths, 64ths to 32nds
            (1.0 / 6.0, Some(3)),
            (0.125, Some(4)),
            (1.0 / 12.0, Some(6)),
            (0.0625, Some(8)),
            // A 192nd of a measure, the finest StepMania writes
            (1.0 / 48.0, Some(24)),
            // 20ths don't halve
            (0.2, Some(1)),
            (0.5 + 1e-4, None),
        ];
        for (beat, grid) in cases {
            assert_eq!(stronger_grid(beat), grid, "beat {beat}");
        }
    }

    #[test]
    fn finds_syncopations() {
        let syncopations = |beats: &[f64], expected: &[(usize, f64)]| assert_found_on(beats, Technique::Syncopation, expected);
        let third = 1.0 / 3.0;
        // On the beat, or 8ths that reach the next beat: none
        syncopations(&[0.0, 1.0, 2.0], &[]);
        syncopations(&[0.0, 0.5, 1.0, 1.5, 2.0], &[]);
        // An 8th with no note on the next beat, at the speed of the gap before it
        syncopations(&[0.0, 0.5, 1.5, 2.0], &[(1, 4.0)]);
        // Only off the beat: all of them, and the first has no note before
        syncopations(&[0.5, 1.5, 2.5], &[(0, 0.0), (1, 2.0), (2, 2.0)]);
        // A 16th leads to the next 8th, not the next beat
        syncopations(&[0.0, 0.25, 0.5, 0.75, 1.0], &[]);
        syncopations(&[0.0, 0.25, 1.0], &[(1, 8.0)]);
        // A gallop's 16th reaches its beat. Dotted 8ths don't reach theirs.
        syncopations(&[0.0, 0.75, 1.0], &[]);
        syncopations(&[0.0, 0.75, 1.5, 2.25, 3.0], &[(1, 8.0 / 3.0), (2, 8.0 / 3.0), (3, 8.0 / 3.0)]);
        // Triplets lead to the next beat, a 24th to the next 12th
        syncopations(&[0.0, third, 2.0 * third, 1.0], &[]);
        syncopations(&[0.0, third, 2.0 * third, 2.0], &[(1, 6.0), (2, 6.0)]);
        syncopations(&[0.0, 0.5 * third, third, 1.0], &[]);
        syncopations(&[0.0, 0.5 * third, 1.0], &[(1, 12.0)]);
        // Jumps and hold heads are notes
        let mut notes = on_beats(&[0.0, 0.5, 1.0]);
        notes[1].row = [Note::HoldStart, Note::Empty, Note::Empty, Note::Tap];
        assert_eq!(found_in(&notes, Technique::Syncopation), []);
        assert_eq!(found_in(&notes[..2], Technique::Syncopation), [(1, 1, 4.0)]);
    }

    #[test]
    fn finds_rhythm_changes() {
        let changes = |beats: &[f64], expected: &[(usize, f64)]| assert_found_on(beats, Technique::RhythmChange, expected);
        // Even gaps, or too few notes to have two
        changes(&[0.0, 1.0, 2.0, 3.0], &[]);
        changes(&[0.0, 1.0], &[]);
        changes(&[0.0], &[]);
        // Quarters into 8ths and back: on the notes between two gaps, at the shorter's speed
        changes(&[0.0, 1.0, 2.0, 2.5, 3.0, 4.0], &[(2, 4.0), (4, 4.0)]);
        // A 16th among quarters: on both its notes
        changes(&[0.0, 1.0, 2.0, 2.25, 3.25, 4.25], &[(2, 8.0), (3, 8.0)]);
        // Dotted 8th, 16th, quarter: three gaps, two changes
        changes(&[0.0, 0.75, 1.0, 2.0], &[(1, 8.0), (2, 8.0)]);
        // Within `Settings::even_tolerance` is even
        changes(&[0.0, 1.0, 2.0625, 3.0625], &[]);
        changes(&[0.0, 1.0, 2.25, 3.25], &[(1, 2.0), (2, 2.0)]);
        // In seconds: twice the BPM halves the gap
        let mut faster = on_beats(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        (faster[3].time, faster[4].time) = (1.25, 1.5);
        assert_eq!(found_in(&faster, Technique::RhythmChange), [(2, 1, 4.0)]);
    }

    #[test]
    fn finds_streams() {
        assert_eq!(found("L D U R L D U R", Technique::Stream), [(0, 8, 8.0)]);
        assert_eq!(found("L D U R L D U", Technique::Stream), []);
        // Slow ones count
        assert_eq!(found("L . D . U . R . L . D . U . R", Technique::Stream), [(0, 8, 4.0)]);
        // A change of rhythm ends one. The next starts on its last note.
        let two = found("L D U R L D U R L . D . U . R . L . D . U . R", Technique::Stream);
        assert_eq!(two, [(0, 9, 8.0), (8, 8, 4.0)]);
        // Too few at either rhythm
        assert_eq!(found("L D U R L . D . U . R . L", Technique::Stream), []);
        // Jumps are notes too
        assert_eq!(found("L D UR R L D U R", Technique::Stream), [(0, 8, 8.0)]);
    }
}
