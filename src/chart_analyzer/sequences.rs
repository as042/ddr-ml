//! Techniques found from the notes alone, whatever the feet do.

use super::technique::{Occurrence, Technique, even_runs};
use super::{Settings, TIME_EPSILON, TimedNote, arrows, steps_on};

/// The repeated arrows, jumps and held steps of `segment`, each on its note, and its streams,
/// each on its first: a stream is a run of evenly spaced notes as long as it goes, so a change
/// of rhythm ends it, and the next may start on its last note (see `Settings::min_stream_notes`
/// and `Settings::even_tolerance`). A held step is under a hold from earlier in the segment.
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
