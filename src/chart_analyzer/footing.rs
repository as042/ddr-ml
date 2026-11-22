//! Foot simulation, for the patterns that depend on which foot steps where: crossovers, laterals
//! and spins.
//!
//! Each segment (see `segments`) starts at home, left foot on Left and right foot on Right, with
//! `Start` choosing which foot steps first. The feet then alternate, except:
//! - a jack is stepped by the foot already on its arrow
//! - a foot holding a hold or roll can't step
//! - after a jump, the next note goes to the foot that leaves the player facing the screen more
//!   squarely
//!
//! Positions are judged by which way the player faces (see `facing`).

use std::ops::AddAssign;

use super::{TIME_EPSILON, TimedNote, shares_an_arrow, steps_on};

const LEFT_ARROW: usize = 0;
const DOWN_ARROW: usize = 1;
const UP_ARROW: usize = 2;
const RIGHT_ARROW: usize = 3;

/// (x, y) of each arrow on the pad: x to the right, y toward the screen.
const ARROW_POSITIONS: [(i32, i32); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Foot {
    Left = 0,
    Right = 1,
}

impl Foot {
    fn other(self) -> Foot {
        match self {
            Foot::Left => Foot::Right,
            Foot::Right => Foot::Left,
        }
    }
}

/// Which foot steps the first note of each segment. `START` picks one; the rest are kept for
/// comparison.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Start {
    Left,
    Right,
    /// The less awkward foot (see `Footing::awkwardness`). Left on a tie.
    Easier,
    /// The more awkward foot. Left on a tie.
    Harder,
    /// The foot that steps the segment's first Left or Right arrow from its own side. Left if
    /// neither does.
    Natural,
}

/// Awkward positions counted by a simulation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Footing {
    /// Steps into a crossover from an uncrossed position: one foot on the far side's arrow, the
    /// other on Up or Down (facing 135 degrees from the screen)
    pub crossovers: usize,
    /// Steps into a lateral: left foot on Right, right foot on Left (facing away)
    pub laterals: usize,
    /// Turns of 180 to 359 degrees in one direction that pass through facing away (facing left
    /// round to facing right doesn't)
    pub half_spins: usize,
    /// Full 360 degree turns in one direction
    pub full_spins: usize,
}

impl Footing {
    /// Total count, for comparing simulations.
    pub fn awkwardness(&self) -> usize {
        self.crossovers + self.laterals + self.half_spins + self.full_spins
    }
}

impl AddAssign for Footing {
    fn add_assign(&mut self, other: Footing) {
        self.crossovers += other.crossovers;
        self.laterals += other.laterals;
        self.half_spins += other.half_spins;
        self.full_spins += other.full_spins;
    }
}

pub(super) fn simulate_chart(segments: &[&[TimedNote]], start: Start) -> Footing {
    let mut total = Footing::default();
    for &segment in segments {
        total += simulate_segment(segment, start);
    }
    total
}

pub(super) fn simulate_segment(segment: &[TimedNote], start: Start) -> Footing {
    let left = simulate(segment, Foot::Left);
    let right = simulate(segment, Foot::Right);
    let use_right = match start {
        Start::Left => false,
        Start::Right => true,
        Start::Easier => right.footing.awkwardness() < left.footing.awkwardness(),
        Start::Harder => right.footing.awkwardness() > left.footing.awkwardness(),
        Start::Natural => !left.natural && right.natural,
    };
    if use_right { right.footing } else { left.footing }
}

/// Extra awkwardness from starting each segment on the worse foot rather than the better.
pub(super) fn wrong_foot_penalty(segments: &[&[TimedNote]]) -> usize {
    segments
        .iter()
        .map(|segment| {
            let left = simulate(segment, Foot::Left).footing.awkwardness();
            let right = simulate(segment, Foot::Right).footing.awkwardness();
            left.abs_diff(right)
        })
        .sum()
}

/// One segment simulated from one starting foot.
struct Run {
    footing: Footing,
    /// Whether the first single note on Left or Right was stepped from its own side (true if
    /// there is none)
    natural: bool,
}

/// What stepped the previous note.
#[derive(Clone, Copy)]
enum Last {
    Nothing,
    Foot(Foot),
    BothFeet,
}

/// `first` steps the first note, if that's a single arrow.
fn simulate(segment: &[TimedNote], first: Foot) -> Run {
    // Arrow under each foot, or None once the other foot takes it
    let mut on: [Option<usize>; 2] = [Some(LEFT_ARROW), Some(RIGHT_ARROW)];
    // A foot can't step until its hold or roll ends
    let mut held_until = [f64::NEG_INFINITY; 2];
    let mut last = Last::Nothing;
    let mut natural = None;
    let mut footing = Footing::default();
    let mut stance = Stance::of(facing_of(on));
    let mut turning = Turning::new(facing_of(on));

    for (i, note) in segment.iter().enumerate() {
        let free = held_until.map(|until| until <= note.time + TIME_EPSILON);
        let arrows: Vec<usize> = (0..4).filter(|&arrow| steps_on(note.row[arrow])).collect();

        if let [arrow] = arrows[..] {
            let foot = match free {
                [true, false] => Foot::Left,
                [false, true] => Foot::Right,
                // Both feet are holding: a hand, which the feet can't take
                [false, false] => continue,
                [true, true] => match foot_on(on, arrow) {
                    Some(foot) if i > 0 && shares_an_arrow(&segment[i - 1], note) => foot,
                    _ => match last {
                        Last::Foot(foot) => foot.other(),
                        Last::Nothing => first,
                        Last::BothFeet => squarer_foot(on, arrow, first),
                    },
                },
            };
            if on[foot.other() as usize] == Some(arrow) {
                on[foot.other() as usize] = None;
            }
            on[foot as usize] = Some(arrow);
            if let Some(end) = note.hold_ends[arrow] {
                held_until[foot as usize] = end;
            }
            if natural.is_none() && (arrow == LEFT_ARROW || arrow == RIGHT_ARROW) {
                natural = Some((arrow == LEFT_ARROW) == (foot == Foot::Left));
            }
            last = Last::Foot(foot);
        } else {
            let (left, right) = jump_feet(&arrows, turning.facing());
            on = [Some(left), Some(right)];
            for (foot, arrow) in [(Foot::Left, left), (Foot::Right, right)] {
                if let Some(end) = note.hold_ends[arrow] {
                    held_until[foot as usize] = end;
                }
            }
            last = Last::BothFeet;
        }

        let facing = facing_of(on);
        let new_stance = Stance::of(facing);
        if new_stance == Stance::Crossed && !matches!(stance, Stance::Crossed | Stance::Lateral) {
            footing.crossovers += 1;
        }
        if new_stance == Stance::Lateral && stance != Stance::Lateral {
            footing.laterals += 1;
        }
        stance = new_stance;
        turning.face(facing, &mut footing);
    }
    turning.finish(&mut footing);
    Run { footing, natural: natural.unwrap_or(true) }
}

fn foot_on(on: [Option<usize>; 2], arrow: usize) -> Option<Foot> {
    [Foot::Left, Foot::Right].into_iter().find(|&foot| on[foot as usize] == Some(arrow))
}

/// The foot that faces the player more squarely to the screen by stepping on `arrow`, or `tie`.
/// For after a jump, when alternating doesn't decide.
fn squarer_foot(on: [Option<usize>; 2], arrow: usize, tie: Foot) -> Foot {
    let turn = |foot: Foot| {
        let mut after = on;
        after[foot as usize] = Some(arrow);
        if after[foot.other() as usize] == Some(arrow) {
            after[foot.other() as usize] = None;
        }
        facing_of(after).map_or(i32::MAX, i32::abs)
    };
    let (left, right) = (turn(Foot::Left), turn(Foot::Right));
    if left < right {
        Foot::Left
    } else if right < left {
        Foot::Right
    } else {
        tie
    }
}

/// Arrows the (left, right) feet take in a jump or hand: the leftmost and rightmost. Up and Down
/// go whichever way turns the player less from `facing`.
fn jump_feet(arrows: &[usize], facing: Option<i32>) -> (usize, usize) {
    let x = |arrow: usize| ARROW_POSITIONS[arrow].0;
    let left = *arrows.iter().min_by_key(|&&arrow| x(arrow)).expect("a note has arrows");
    let right = *arrows.iter().rev().max_by_key(|&&arrow| x(arrow)).expect("a note has arrows");
    if x(left) != x(right) {
        return (left, right);
    }
    let turn = |left, right| facing.map_or(0, |facing| wrap(self::facing(left, right) - facing).abs());
    if turn(UP_ARROW, DOWN_ARROW) < turn(DOWN_ARROW, UP_ARROW) {
        (UP_ARROW, DOWN_ARROW)
    } else {
        (DOWN_ARROW, UP_ARROW)
    }
}

/// Degrees clockwise from facing the screen, in (-180, 180]: 0 at home, 45 with the right foot on
/// Down, 90 with the left on Up and the right on Down, 135 in a crossover, 180 in a lateral.
fn facing(left: usize, right: usize) -> i32 {
    let (left_x, left_y) = ARROW_POSITIONS[left];
    let (right_x, right_y) = ARROW_POSITIONS[right];
    // Square to the line from the left foot to the right foot
    let (dx, dy) = ((right_x - left_x) as f64, (right_y - left_y) as f64);
    wrap((-dy).atan2(dx).to_degrees().round() as i32)
}

/// None unless both feet are on an arrow.
fn facing_of(on: [Option<usize>; 2]) -> Option<i32> {
    match on {
        [Some(left), Some(right)] if left != right => Some(facing(left, right)),
        _ => None,
    }
}

/// Wraps degrees to (-180, 180].
fn wrap(angle: i32) -> i32 {
    let angle = angle.rem_euclid(360);
    if angle > 180 { angle - 360 } else { angle }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stance {
    /// A foot isn't on an arrow
    Unknown,
    /// Facing within 90 degrees of the screen
    Open,
    /// Facing 135 degrees from the screen
    Crossed,
    /// Facing away
    Lateral,
}

impl Stance {
    fn of(facing: Option<i32>) -> Stance {
        match facing {
            None => Stance::Unknown,
            Some(facing) if facing.abs() <= 90 => Stance::Open,
            Some(180) => Stance::Lateral,
            Some(_) => Stance::Crossed,
        }
    }
}

/// Tracks the player's facing through a segment, to count spins.
struct Turning {
    /// Facing, unwrapped (405 is a full clockwise turn plus 45), or None while unknown
    angle: Option<i32>,
    /// Angle the current turn started at
    turn_start: i32,
    /// 1 clockwise, -1 counterclockwise, 0 before the turn moves
    direction: i32,
}

impl Turning {
    fn new(facing: Option<i32>) -> Self {
        Turning { angle: facing, turn_start: facing.unwrap_or(0), direction: 0 }
    }

    /// Current facing, wrapped.
    fn facing(&self) -> Option<i32> {
        self.angle.map(wrap)
    }

    /// Turns to `facing` the shorter way round. None loses track.
    fn face(&mut self, facing: Option<i32>, footing: &mut Footing) {
        let (Some(angle), Some(facing)) = (self.angle, facing) else {
            // Losing track ends the turn. Finding the feet again starts a new one.
            self.finish(footing);
            *self = Turning::new(facing);
            return;
        };
        let mut delta = wrap(facing - angle);
        if delta == 0 {
            return;
        }
        // A 180 flip (only a jump can do it) has no direction: take it as back toward the screen.
        if delta == 180 && angle > 0 {
            delta = -180;
        }
        if self.direction != 0 && delta.signum() != self.direction {
            self.finish(footing);
        }
        self.direction = delta.signum();
        self.angle = Some(angle + delta);
    }

    /// Ends the current turn, counting any spins.
    fn finish(&mut self, footing: &mut Footing) {
        if let Some(angle) = self.angle {
            let (low, high) = (angle.min(self.turn_start), angle.max(self.turn_start));
            let turned = high - low;
            footing.full_spins += (turned / 360) as usize;
            // A half spin has to pass through facing away. `away` is the first such angle (180,
            // 540, ...) at or above `low`.
            let away = low + (180 - low).rem_euclid(360);
            if turned % 360 >= 180 && away <= high {
                footing.half_spins += 1;
            }
            self.turn_start = angle;
        }
        self.direction = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chart_analyzer::{Timing, to_timed_notes};
    use crate::sim_loader::{BPMChange, Measure, Note};

    /// A segment from tokens 0.125 s apart, one row each: `L`, `D`, `U`, `R` for taps (`LR` is a
    /// jump), a trailing `+` for hold heads, a trailing `-` for hold ends, `.` for an empty row.
    fn segment(tokens: &str) -> Vec<TimedNote> {
        let measures: Vec<Measure> = tokens
            .split_whitespace()
            .map(|token| {
                let kind = match token.chars().last() {
                    Some('+') => Note::HoldStart,
                    Some('-') => Note::End,
                    _ => Note::Tap,
                };
                let mut row = [Note::Empty; 4];
                for c in token.chars() {
                    if let Some(arrow) = "LDUR".find(c) {
                        row[arrow] = kind;
                    }
                }
                // One beat per row at 480 BPM: 0.125 s apart
                Measure { num_beats: 1, rows: vec![row] }
            })
            .collect();
        to_timed_notes(&measures, &Timing::new(&[BPMChange { beat: 0.0, bpm: 480.0 }], &[]))
    }

    fn footing(crossovers: usize, laterals: usize, half_spins: usize, full_spins: usize) -> Footing {
        Footing { crossovers, laterals, half_spins, full_spins }
    }

    fn run(tokens: &str, first: Foot) -> Footing {
        simulate(&segment(tokens), first).footing
    }

    #[test]
    fn faces_square_to_the_feet() {
        use super::{DOWN_ARROW as D, LEFT_ARROW as L, RIGHT_ARROW as R, UP_ARROW as U};
        // (left foot, right foot, facing)
        let cases = [
            (L, R, 0),
            (L, D, 45),
            (U, R, 45),
            (L, U, -45),
            (D, R, -45),
            (U, D, 90),
            (D, U, -90),
            (R, D, 135),
            (U, L, 135),
            (R, U, -135),
            (D, L, -135),
            (R, L, 180),
        ];
        for (left, right, expected) in cases {
            assert_eq!(facing(left, right), expected, "left foot on {left}, right on {right}");
        }
    }

    #[test]
    fn finds_crossovers_laterals_and_spins() {
        use Foot::Left;
        // The left foot crosses over to Right
        assert_eq!(run("L D R", Left), footing(1, 0, 0, 0));
        // Then back: the turn reverses, so no spin
        assert_eq!(run("L D R D L", Left), footing(1, 0, 0, 0));
        // The right foot goes on to Left: a lateral, 180 degrees round
        assert_eq!(run("L D R L", Left), footing(1, 1, 1, 0));
        // Round the pad: 225 degrees by Up (a half spin), 405 by Down again (a full spin), with
        // one crossover each time round
        assert_eq!(run("L D R U", Left), footing(1, 0, 1, 0));
        assert_eq!(run("L D R U L D", Left), footing(1, 0, 0, 1));
        assert_eq!(run("L D R U L D R U L D", Left), footing(2, 0, 0, 2));
        // Mirrored
        assert_eq!(run("R D L U R D", Foot::Right), footing(1, 0, 0, 1));
        // Facing left round to facing right is 180 degrees, but through facing the screen: no spin
        assert_eq!(run("D U L D U", Left), footing(0, 0, 0, 0));
    }

    #[test]
    fn steps_jacks_with_the_same_foot() {
        // The left foot steps Left twice, then crosses over to Right
        assert_eq!(run("L L D R", Foot::Left), footing(1, 0, 0, 0));
        // A jump jack. The right foot on Up after it doesn't cross.
        assert_eq!(run("LR LR U", Foot::Left), footing(0, 0, 0, 0));
    }

    #[test]
    fn lets_the_other_foot_step_during_a_hold() {
        // The right foot steps Down and Right under the left's hold. Alternating would cross the
        // left foot over to Right while it's still holding.
        assert_eq!(run("L+ D R L- D", Foot::Left), footing(0, 0, 0, 0));
    }

    #[test]
    fn steps_after_a_jump_with_the_foot_that_faces_the_screen() {
        // The right foot (on Down) takes Right, rather than the left foot crossing over
        assert_eq!(run("LD R U", Foot::Left), footing(0, 0, 0, 0));
        assert_eq!(run("LD R U", Foot::Right), footing(0, 0, 0, 0));
    }

    #[test]
    fn starting_foot_matters() {
        // Started on the right foot, the left steps Right while the right is on Up: a crossover
        let up_right = segment("U R U R");
        assert_eq!(simulate(&up_right, Foot::Left).footing, footing(0, 0, 0, 0));
        assert_eq!(simulate(&up_right, Foot::Right).footing, footing(1, 0, 0, 0));
        assert_eq!(simulate_segment(&up_right, Start::Left), footing(0, 0, 0, 0));
        assert_eq!(simulate_segment(&up_right, Start::Right), footing(1, 0, 0, 0));
        assert_eq!(simulate_segment(&up_right, Start::Easier), footing(0, 0, 0, 0));
        assert_eq!(simulate_segment(&up_right, Start::Harder), footing(1, 0, 0, 0));
        // The right foot steps the first side arrow (Right) if the left foot starts
        assert_eq!(simulate_segment(&up_right, Start::Natural), footing(0, 0, 0, 0));
        // Mirrored: the right foot is the natural start
        let up_left = segment("U L U L");
        assert_eq!(simulate_segment(&up_left, Start::Natural), footing(0, 0, 0, 0));
        assert_eq!(simulate_segment(&up_left, Start::Left), footing(1, 0, 0, 0));
    }

    #[test]
    fn finds_the_wrong_foot_penalty() {
        // Each is 1 more awkward on the worse foot. In the last, that's 2 crossovers and a full
        // spin against 1 crossover and a full spin.
        let one_worse = [segment("U R U R"), segment("U L U L"), segment("L D R U L D")];
        // A crossover whichever foot starts
        let no_worse = segment("L D R");
        let mut segments: Vec<&[TimedNote]> = one_worse.iter().map(Vec::as_slice).collect();
        segments.push(&no_worse);
        assert_eq!(wrong_foot_penalty(&segments), 3);
        assert_eq!(wrong_foot_penalty(&[]), 0);
    }
}
