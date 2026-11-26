//! Named patterns of several notes, found on a footing. None has a cost.
//!
//! A pattern's notes each have one arrow, stepped without a bracket or a freeze switch.

use super::super::Settings;
use super::super::technique::{Occurrence, Technique, even, even_runs, speed};
use super::{DOWN_ARROW, Events, Foot, LEFT_ARROW, RIGHT_ARROW, Step, TIME_EPSILON, TimedNote, UP_ARROW, steps_on};

/// The patterns of `segment`, footed by `steps`, each on its first note:
/// - staircase: each arrow once from one side to the other, feet alternating, evenly spaced
///   (LDUR, LUDR, RUDL or RDUL). Two may share an end note.
/// - drill: `Settings::min_drill_notes` or more notes back and forth between two arrows, a foot
///   on each, evenly spaced. The next may start on its last note.
/// - gallop: 2 notes on their own (see `Settings::isolation`), on different arrows and feet. A
///   flam (`flam`) is timed by the gap before it.
/// - triple: 3 evenly spaced notes on their own, feet alternating
///
/// Evenly spaced is by `Settings::even_tolerance`. None has a speed limit: each one's speed is in
/// its `Occurrence`.
pub fn find(segment: &[TimedNote], steps: &[Step], step_events: &[Events], settings: &Settings) -> Vec<Occurrence> {
    let notes = segment.len();
    let mut found = Vec::new();

    // (arrow, foot) of each note that can be part of a pattern
    let plain: Vec<Option<(usize, Foot)>> = (0..notes)
        .map(|i| {
            let mut arrows = (0..4).filter(|&arrow| steps_on(segment[i].row[arrow]));
            let (Some(arrow), None) = (arrows.next(), arrows.next()) else { return None };
            // A tap under the foot's own hold steps one arrow too
            if step_events[i].brackets > 0 || step_events[i].freeze_switches > 0 {
                return None;
            }
            steps[i].foot(arrow).map(|foot| (arrow, foot))
        })
        .collect();
    let arrow = |i: usize| plain[i].map(|(arrow, _)| arrow);
    // Before note i
    let gap = |i: usize| segment[i].time - segment[i - 1].time;
    let before = |i: usize| if i == 0 { f64::INFINITY } else { gap(i) };
    let after = |i: usize| if i + 1 == notes { f64::INFINITY } else { gap(i + 1) };
    // Note i and the one before: another arrow, the other foot
    let alternates = |i: usize| {
        matches!((plain[i - 1], plain[i]), (Some((a, left)), Some((b, right))) if a != b && left != right)
            && gap(i) > TIME_EPSILON
    };
    let alone = |first: usize, last: usize, gap: f64| {
        let least = settings.isolation * gap - TIME_EPSILON;
        before(first) >= least && after(last) >= least
    };
    for i in 1..notes {
        if alternates(i) && alone(i - 1, i, gap(i)) {
            let mut gallop = Occurrence::over(Technique::Gallop, segment, i - 1, 2);
            if flam(segment, i - 1, settings) {
                gallop.speed = speed(before(i - 1));
            }
            found.push(gallop);
        }
    }
    let mut add = |technique: Technique, first: usize, notes: usize| {
        found.push(Occurrence::over(technique, segment, first, notes));
    };
    for i in 2..notes {
        let (shortest, longest) = (gap(i - 1).min(gap(i)), gap(i - 1).max(gap(i)));
        if alternates(i - 1) && alternates(i) && even(shortest, longest, settings.even_tolerance) && alone(i - 2, i, longest) {
            add(Technique::Triple, i - 2, 3);
        }
    }
    for i in 3..notes {
        let gaps = [gap(i - 2), gap(i - 1), gap(i)];
        let (shortest, longest) = (gaps.into_iter().fold(f64::INFINITY, f64::min), gaps.into_iter().fold(0.0, f64::max));
        let ends = matches!(
            (arrow(i - 3), arrow(i)),
            (Some(LEFT_ARROW), Some(RIGHT_ARROW)) | (Some(RIGHT_ARROW), Some(LEFT_ARROW))
        );
        let middle = matches!(
            (arrow(i - 2), arrow(i - 1)),
            (Some(DOWN_ARROW), Some(UP_ARROW)) | (Some(UP_ARROW), Some(DOWN_ARROW))
        );
        if ends && middle && (i - 2..=i).all(alternates) && even(shortest, longest, settings.even_tolerance) {
            add(Technique::Staircase, i - 3, 4);
        }
    }
    let back_and_forth = |first: usize, i: usize| i - first < 2 || arrow(i) == arrow(i - 2);
    for (first, notes) in even_runs(segment, settings.min_drill_notes, settings.even_tolerance, alternates, back_and_forth) {
        add(Technique::Drill, first, notes);
    }
    found
}

/// The gallop from note `first` is a flam: close enough to hit as one motion.
pub fn flam(segment: &[TimedNote], first: usize, settings: &Settings) -> bool {
    segment[first + 1].time - segment[first].time <= settings.flam_seconds + TIME_EPSILON
}

#[cfg(test)]
mod tests {
    use super::super::tests::{WIDE, segment};
    use super::super::{Cost, Costs, TEST_COSTS as COSTS, solve};
    use super::*;

    /// A footing's patterns, counted.
    #[derive(Debug, Default, PartialEq)]
    struct Patterns {
        staircases: usize,
        drills: usize,
        drill_notes: usize,
        gallops: usize,
        triples: usize,
    }

    fn patterns(staircases: usize, drills: usize, drill_notes: usize, gallops: usize, triples: usize) -> Patterns {
        Patterns { staircases, drills, drill_notes, gallops, triples }
    }

    fn found_in(notes: &[TimedNote]) -> Patterns {
        let solution = solve(notes, &COSTS, &WIDE);
        Patterns {
            staircases: solution.count(Technique::Staircase),
            drills: solution.count(Technique::Drill),
            drill_notes: solution.of(Technique::Drill).map(|drill| drill.notes).sum(),
            gallops: solution.count(Technique::Gallop),
            triples: solution.count(Technique::Triple),
        }
    }

    fn found(tokens: &str) -> Patterns {
        found_in(&segment(tokens))
    }

    /// The notes (0-based) a pattern starts at.
    fn starts(tokens: &str, technique: Technique) -> Vec<usize> {
        solve(&segment(tokens), &COSTS, &WIDE).of(technique).map(|pattern| pattern.note).collect()
    }

    #[test]
    fn finds_staircases() {
        for stairs in ["L D U R", "L U D R", "R U D L", "R D U L"] {
            assert_eq!(found(stairs).staircases, 1, "{stairs}");
        }
        // Back down again, sharing Right
        assert_eq!(found("L D U R U D L").staircases, 2);
        assert_eq!(starts("L D U R U D L", Technique::Staircase), [0, 3]);
        // Not each arrow in turn
        assert_eq!(found("L D R U").staircases, 0);
        assert_eq!(found("D L R U").staircases, 0);
        // Not evenly spaced: two gallops
        assert_eq!(found("L D . U R"), patterns(0, 0, 0, 2, 0));
        // The left foot taps Up while it holds Left
        assert_eq!(found("L+ D U R L-").staircases, 0);
        // Slow ones count
        assert_eq!(found("L . D . U . R").staircases, 1);
    }

    #[test]
    fn finds_drills() {
        assert_eq!(found("L R L R L"), patterns(0, 1, 5, 0, 0));
        assert_eq!(found("U D U D U D U"), patterns(0, 1, 7, 0, 0));
        assert_eq!(found("L R L R"), patterns(0, 0, 0, 0, 0));
        // Broken by a pause: two triples
        assert_eq!(found("L R L . R L R"), patterns(0, 0, 0, 0, 2));
        // Two, sharing a Left
        assert_eq!(found("L D L D L U L U L"), patterns(0, 2, 10, 0, 0));
        assert_eq!(starts("L D L D L U L U L", Technique::Drill), [0, 4]);
        // After a jump
        assert_eq!(starts("LR U D U D U", Technique::Drill), [1]);
        // One foot for both arrows isn't a drill
        let jacks = Costs { vertical_footswitch: Cost { base: 100.0, speed: 0.0 }, ..COSTS };
        assert_eq!(solve(&segment("U U U U U"), &jacks, &WIDE).count(Technique::Drill), 0);
    }

    #[test]
    fn finds_gallops() {
        assert_eq!(found("L D"), patterns(0, 0, 0, 1, 0));
        // The gaps between them: exactly twice the gaps within
        assert_eq!(found("L D . L D . L D"), patterns(0, 0, 0, 3, 0));
        assert_eq!(starts("L D . L D . L D", Technique::Gallop), [0, 2, 4]);
        // Between jumps
        assert_eq!(found("UD . L R . UD").gallops, 1);
        // Too close to a jump
        assert_eq!(found("UD L R").gallops, 0);
        // A slow one counts
        assert_eq!(found("L . D").gallops, 1);
        // One arrow
        assert_eq!(found("L L").gallops, 0);

        // 0.1 s apart, then a note 0.15 s later: neither on its own nor evenly spaced
        let mut near = segment("L D U");
        (near[1].time, near[2].time) = (0.1, 0.25);
        assert_eq!(found_in(&near), patterns(0, 0, 0, 0, 0));
        (near[1].time, near[2].time) = (0.1, 0.3);
        assert_eq!(found_in(&near), patterns(0, 0, 0, 1, 0));
    }

    #[test]
    fn finds_triples() {
        assert_eq!(found("L D U"), patterns(0, 0, 0, 0, 1));
        assert_eq!(found("L D L . . R U R"), patterns(0, 0, 0, 0, 2));
        assert_eq!(starts("L D L . . R U R", Technique::Triple), [0, 3]);
        assert_eq!(found("L D U . R"), patterns(0, 0, 0, 0, 1));
        // Four in a row
        assert_eq!(found("L D U R").triples, 0);
        // A jump in the middle
        assert_eq!(found("L UD R"), patterns(0, 0, 0, 0, 0));
        // A slow one counts
        assert_eq!(found("L . D . U").triples, 1);
    }

    #[test]
    fn times_each_pattern() {
        // (first note, notes, notes per second) of each pattern of a kind
        let timed = |tokens: &str, technique: Technique| -> Vec<(usize, usize, f64)> {
            let solution = solve(&segment(tokens), &COSTS, &WIDE);
            solution.of(technique).map(|pattern| (pattern.note, pattern.notes, pattern.speed)).collect()
        };
        // Notes 0.125 s apart: 8 a second. A row between: 4.
        assert_eq!(timed("L D", Technique::Gallop), [(0, 2, 8.0)]);
        assert_eq!(timed("L . D", Technique::Gallop), [(0, 2, 4.0)]);
        // The second is 0.5 s after the first, twice its own gap
        assert_eq!(timed("L D . . . L . D", Technique::Gallop), [(0, 2, 8.0), (2, 2, 4.0)]);
        // As flams: the gap before, 0 for none
        let flams = Settings { flam_seconds: 0.25, ..WIDE };
        let timed_flams = |tokens: &str| -> Vec<(usize, usize, f64)> {
            let solution = solve(&segment(tokens), &COSTS, &flams);
            solution.of(Technique::Gallop).map(|pattern| (pattern.note, pattern.notes, pattern.speed)).collect()
        };
        assert_eq!(timed_flams("L D . . . L . D"), [(0, 2, 0.0), (2, 2, 2.0)]);
        assert_eq!(timed_flams("U . . . L R"), [(1, 2, 2.0)]);
        assert_eq!(timed("L . D . U", Technique::Triple), [(0, 3, 4.0)]);
        assert_eq!(timed("L D U R", Technique::Staircase), [(0, 4, 8.0)]);
        assert_eq!(timed("L . . . D . . . U . . . R", Technique::Staircase), [(0, 4, 2.0)]);
        assert_eq!(timed("L R L R L R L", Technique::Drill), [(0, 7, 8.0)]);
        // Two drills sharing a Left, the second half as fast
        assert_eq!(timed("L D L D L . U . L . U . L", Technique::Drill), [(0, 5, 8.0), (4, 5, 4.0)]);
    }

    #[test]
    fn finds_nothing_in_too_few_notes() {
        assert_eq!(found("L"), Patterns::default());
        assert_eq!(found("LR"), Patterns::default());
    }
}
