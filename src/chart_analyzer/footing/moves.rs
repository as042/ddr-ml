//! How each foot gets from one of its steps to its next, on a footing.

use super::{ARROW_POSITIONS, Arrows, Step, TimedNote, has};

/// A foot's step, after an earlier one of its own in the segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Move {
    /// The note it steps
    pub note: usize,
    /// Since the foot's step before
    pub seconds: f64,
    /// How far the foot comes: 0 onto the same arrow, 2 across the pad (Left to Right). A foot on
    /// two arrows is midway between them.
    pub distance: f64,
}

/// Each foot's moves in `segment`, footed by `steps`, in note order.
pub fn find(segment: &[TimedNote], steps: &[Step]) -> Vec<Move> {
    // Per foot: when and where it last stepped
    let mut last: [Option<(f64, (f64, f64))>; 2] = [None; 2];
    let mut moves = Vec::new();
    for (note, step) in steps.iter().enumerate() {
        for foot in 0..2 {
            if step.feet[foot] == 0 {
                continue;
            }
            let (time, to) = (segment[note].time, position(step.feet[foot]));
            if let Some((before, from)) = last[foot] {
                moves.push(Move { note, seconds: time - before, distance: (to.0 - from.0).hypot(to.1 - from.1) });
            }
            last[foot] = Some((time, to));
        }
    }
    moves
}

/// Where a foot on `arrows` is on the pad: the middle of them.
fn position(arrows: Arrows) -> (f64, f64) {
    let on: Vec<(i32, i32)> = (0..4).filter(|&arrow| has(arrows, arrow)).map(|arrow| ARROW_POSITIONS[arrow]).collect();
    let middle = |along: fn(&(i32, i32)) -> i32| on.iter().map(along).sum::<i32>() as f64 / on.len() as f64;
    (middle(|position| position.0), middle(|position| position.1))
}

#[cfg(test)]
mod tests {
    use super::super::tests::segment;
    use super::super::{TEST_COSTS, solve};
    use super::*;

    /// (note, seconds, distance) of each move in the cheapest footing.
    fn moves(tokens: &str) -> Vec<(usize, f64, f64)> {
        solve(&segment(tokens), &TEST_COSTS, 1_000_000).moves.iter().map(|m| (m.note, m.seconds, m.distance)).collect()
    }

    #[test]
    fn measures_each_foots_moves() {
        let diagonal = 2.0_f64.sqrt();
        // Each foot back onto its own arrow
        assert_eq!(moves("L R L R"), [(2, 0.25, 0.0), (3, 0.25, 0.0)]);
        // A staircase: the left foot from Left to Up, the right from Down to Right
        assert_eq!(moves("L D U R"), [(2, 0.25, diagonal), (3, 0.25, diagonal)]);
        // Across the pad and back in a crossover, after a pause
        assert_eq!(moves("L D R . D L"), [(2, 0.25, 2.0), (3, 0.375, 0.0), (4, 0.375, 2.0)]);
        // A jump moves both feet: here neither goes anywhere
        assert_eq!(moves("LR LR"), [(1, 0.125, 0.0), (1, 0.125, 0.0)]);
        // Nothing to come from
        assert_eq!(moves("L"), []);
        assert_eq!(moves("L R"), []);
    }

    #[test]
    fn puts_a_bracketing_foot_between_its_arrows() {
        assert_eq!(position(0b0001), (-1.0, 0.0));
        assert_eq!(position(0b1000), (1.0, 0.0));
        // Left and Down
        assert_eq!(position(0b0011), (-0.5, -0.5));
    }
}
