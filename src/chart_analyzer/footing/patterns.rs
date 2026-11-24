//! Named patterns of several notes, found on a footing. None has a cost.
//!
//! A pattern's notes each have one arrow, stepped without a bracket or a freeze switch.

use std::ops::AddAssign;

use super::{DOWN_ARROW, Events, Foot, LEFT_ARROW, RIGHT_ARROW, Step, TIME_EPSILON, TimedNote, UP_ARROW, steps_on};

/// Longest gap between the notes of a gallop or triple.
pub const BURST_GAP_SECONDS: f64 = 0.15;
/// Notes are evenly spaced if their longest gap is this fraction longer than their shortest, at most.
const EVEN_TOLERANCE: f64 = 0.1;
/// A gallop or triple is on its own if the gap on each side is this many times its own, at least.
const ISOLATION: f64 = 2.0;
const MIN_DRILL_NOTES: usize = 5;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Patterns {
    /// Each arrow once from one side to the other, feet alternating, evenly spaced: LDUR, LUDR,
    /// RUDL or RDUL. Two may share an end note.
    pub staircases: usize,
    /// `MIN_DRILL_NOTES` or more notes back and forth between two arrows, a foot on each, evenly
    /// spaced
    pub drills: usize,
    pub drill_notes: usize,
    /// 2 notes on their own, `BURST_GAP_SECONDS` apart at most, on different arrows and feet
    pub gallops: usize,
    /// 3 evenly spaced notes on their own, `BURST_GAP_SECONDS` apart at most, feet alternating
    pub triples: usize,
}

impl AddAssign for Patterns {
    fn add_assign(&mut self, other: Patterns) {
        let Patterns { staircases, drills, drill_notes, gallops, triples } = other;
        self.staircases += staircases;
        self.drills += drills;
        self.drill_notes += drill_notes;
        self.gallops += gallops;
        self.triples += triples;
    }
}

/// The patterns starting at each note of `segment`, footed by `steps`.
pub fn find(segment: &[TimedNote], steps: &[Step], step_events: &[Events]) -> Vec<Patterns> {
    let notes = segment.len();
    let mut found = vec![Patterns::default(); notes];

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
    let even = |shortest: f64, longest: f64| longest <= shortest * (1.0 + EVEN_TOLERANCE) + TIME_EPSILON;
    let quick = |gap: f64| gap <= BURST_GAP_SECONDS + TIME_EPSILON;
    let alone = |first: usize, last: usize, gap: f64| {
        before(first) >= ISOLATION * gap - TIME_EPSILON && after(last) >= ISOLATION * gap - TIME_EPSILON
    };

    for i in 1..notes {
        if alternates(i) && quick(gap(i)) && alone(i - 1, i, gap(i)) {
            found[i - 1].gallops += 1;
        }
    }
    for i in 2..notes {
        let (shortest, longest) = (gap(i - 1).min(gap(i)), gap(i - 1).max(gap(i)));
        if alternates(i - 1) && alternates(i) && quick(longest) && even(shortest, longest) && alone(i - 2, i, longest) {
            found[i - 2].triples += 1;
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
        if ends && middle && (i - 2..=i).all(alternates) && even(shortest, longest) {
            found[i - 3].staircases += 1;
        }
    }

    // Drills: each run as long as it goes. The next may start on its last note.
    let mut start = 0;
    let (mut shortest, mut longest) = (f64::INFINITY, 0.0_f64);
    for i in 1..=notes {
        let joins = i < notes && alternates(i);
        if joins {
            let (shorter, longer) = (shortest.min(gap(i)), longest.max(gap(i)));
            let back = i - start < 2 || arrow(i) == arrow(i - 2);
            if back && even(shorter, longer) {
                (shortest, longest) = (shorter, longer);
                continue;
            }
        }
        if i - start >= MIN_DRILL_NOTES {
            found[start].drills += 1;
            found[start].drill_notes += i - start;
        }
        (start, shortest, longest) = if joins { (i - 1, gap(i), gap(i)) } else { (i, f64::INFINITY, 0.0) };
    }
    found
}

#[cfg(test)]
mod tests {
    use super::super::tests::segment;
    use super::super::{Cost, Costs, TEST_COSTS as COSTS, solve};
    use super::*;

    const WIDTH: usize = 1_000_000;

    fn patterns(staircases: usize, drills: usize, drill_notes: usize, gallops: usize, triples: usize) -> Patterns {
        Patterns { staircases, drills, drill_notes, gallops, triples }
    }

    fn found_in(notes: &[TimedNote]) -> Patterns {
        let solution = solve(notes, &COSTS, WIDTH);
        let mut added = Patterns::default();
        for &patterns in &solution.step_patterns {
            added += patterns;
        }
        assert_eq!(added, solution.patterns);
        solution.patterns
    }

    fn found(tokens: &str) -> Patterns {
        found_in(&segment(tokens))
    }

    /// The notes (0-based) a pattern starts at.
    fn starts(tokens: &str, count: fn(&Patterns) -> usize) -> Vec<usize> {
        let solution = solve(&segment(tokens), &COSTS, WIDTH);
        (0..solution.step_patterns.len()).filter(|&i| count(&solution.step_patterns[i]) > 0).collect()
    }

    #[test]
    fn finds_staircases() {
        for stairs in ["L D U R", "L U D R", "R U D L", "R D U L"] {
            assert_eq!(found(stairs).staircases, 1, "{stairs}");
        }
        // Back down again, sharing Right
        assert_eq!(found("L D U R U D L").staircases, 2);
        assert_eq!(starts("L D U R U D L", |patterns| patterns.staircases), [0, 3]);
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
        assert_eq!(starts("L D L D L U L U L", |patterns| patterns.drills), [0, 4]);
        // After a jump
        assert_eq!(starts("LR U D U D U", |patterns| patterns.drills), [1]);
        // One foot for both arrows isn't a drill
        let jacks = Costs { vertical_footswitch: Cost { base: 100.0, speed: 0.0 }, ..COSTS };
        assert_eq!(solve(&segment("U U U U U"), &jacks, WIDTH).patterns.drills, 0);
    }

    #[test]
    fn finds_gallops() {
        assert_eq!(found("L D"), patterns(0, 0, 0, 1, 0));
        // The gaps between them: exactly twice the gaps within
        assert_eq!(found("L D . L D . L D"), patterns(0, 0, 0, 3, 0));
        assert_eq!(starts("L D . L D . L D", |patterns| patterns.gallops), [0, 2, 4]);
        // Between jumps
        assert_eq!(found("UD . L R . UD").gallops, 1);
        // Too close to a jump
        assert_eq!(found("UD L R").gallops, 0);
        // Too slow
        assert_eq!(found("L . D").gallops, 0);
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
        assert_eq!(starts("L D L . . R U R", |patterns| patterns.triples), [0, 3]);
        assert_eq!(found("L D U . R"), patterns(0, 0, 0, 0, 1));
        // Four in a row
        assert_eq!(found("L D U R").triples, 0);
        // A jump in the middle
        assert_eq!(found("L UD R"), patterns(0, 0, 0, 0, 0));
        // Too slow
        assert_eq!(found("L . D . U").triples, 0);
    }

    #[test]
    fn finds_nothing_in_too_few_notes() {
        assert_eq!(found("L"), Patterns::default());
        assert_eq!(found("LR"), Patterns::default());
    }
}
