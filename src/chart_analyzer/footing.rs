//! Foot simulation by tree search. Each note is tried every way the feet can take it (either
//! foot for a single arrow; a foot each or one foot bracketing both for a jump; brackets for three
//! or four arrows; under a hold, the free foot or the holding one, its hold passing to the other
//! in a freeze switch), and the cheapest footing by `Costs` wins.
//!
//! Each kind of event has its own penalty and its own growth with speed (see `Cost`).
//!
//! The tree grows a note at a time. Branches that reach the same position at the same note merge
//! (see `Key`): only the cheapest goes on. If more than `max_width` are left, the costliest are
//! dropped and the footing may not be the cheapest.
//!
//! A segment starts with the feet at home, left foot on Left and right foot on Right. Positions
//! are judged by which way the player faces (see `facing`).

use std::collections::{HashMap, hash_map::Entry};
use std::ops::AddAssign;

use super::technique::{Occurrence, Technique, gaps};
use super::{TIME_EPSILON, TimedNote, steps_on};

pub mod moves;
pub mod patterns;

const LEFT_ARROW: usize = 0;
const DOWN_ARROW: usize = 1;
const UP_ARROW: usize = 2;
const RIGHT_ARROW: usize = 3;

/// (x, y) of each arrow on the pad: x to the right, y toward the screen.
const ARROW_POSITIONS: [(i32, i32); 4] = [(-1, 0), (0, -1), (0, 1), (1, 0)];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Foot {
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

/// The events that are positions the feet are in, judged by which way the player faces (see
/// `facing`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Footing {
    /// Steps into a crossover from an uncrossed position: one foot on the far side's arrow, the
    /// other on Up or Down (facing 135 degrees from the screen)
    pub horizontal_crossovers: usize,
    /// Steps into a lateral: left foot on Right, right foot on Left (facing away)
    pub laterals: usize,
    /// Turns of 180 to 359 degrees in one direction that pass through facing away (facing left
    /// round to facing right doesn't)
    pub half_spins: usize,
    /// Full 360 degree turns in one direction
    pub full_spins: usize,
}

impl AddAssign for Footing {
    fn add_assign(&mut self, other: Footing) {
        self.horizontal_crossovers += other.horizontal_crossovers;
        self.laterals += other.laterals;
        self.half_spins += other.half_spins;
        self.full_spins += other.full_spins;
    }
}

/// What stepped the previous note.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Last {
    Nothing,
    Foot(Foot),
    BothFeet,
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
#[derive(Clone, Copy)]
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
        if delta == 180 && wrap(angle) > 0 {
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

/// One kind of event's penalty: `base` at any speed, plus `speed` for each multiple of the
/// reference speed (see `Costs::reference_gap`).
#[derive(Debug, Clone, Copy)]
pub struct Cost {
    pub base: f64,
    pub speed: f64,
}

impl Cost {
    /// In thousandths.
    fn at(&self, speed: f64) -> u64 {
        ((self.base + self.speed * speed) * COST_UNIT).round().max(0.0) as u64
    }
}

/// Penalty per event. Alternating steps are free.
///
/// The speed of a horizontal crossover, lateral or spin is from the shorter of the gaps before and
/// after: leaving the position fast is as hard as entering it fast. That of any other event is
/// from the gap since the note before.
#[derive(Debug, Clone, Copy)]
pub struct Costs {
    pub jack: Cost,
    pub vertical_footswitch: Cost,
    pub sideswitch: Cost,
    pub horizontal_crossover: Cost,
    pub vertical_crossover: Cost,
    pub doublestep: Cost,
    /// Needed or not
    pub bracket: Cost,
    pub freeze_switch: Cost,
    pub lateral: Cost,
    pub half_spin: Cost,
    pub full_spin: Cost,

    /// Seconds between notes at speed 1. Half the gap is speed 2.
    pub reference_gap: f64,
    /// Most a speed counts as
    pub max_speed: f64,
}

pub const COSTS: Costs = Costs {
    jack: Cost { base: 0.0, speed: 2.0 },
    vertical_footswitch: Cost { base: 1.5, speed: 0.5 },
    sideswitch: Cost { base: 3.0, speed: 1.0 },
    horizontal_crossover: Cost { base: 1.5, speed: 0.2 },
    vertical_crossover: Cost { base: 0.0, speed: 0.0 },
    doublestep: Cost { base: 2.0, speed: 7.0 },
    bracket: Cost { base: 6.0, speed: 1.0 },
    freeze_switch: Cost { base: 12.0, speed: 1.0 },
    lateral: Cost { base: 8.0, speed: 2.0 },
    half_spin: Cost { base: 10.0, speed: 2.0 },
    full_spin: Cost { base: 16.0, speed: 4.0 },
    reference_gap: 0.125,
    max_speed: 4.0,
};

/// The costs the tests were written with, so that changing `COSTS` doesn't change what they expect.
#[cfg(test)]
pub const TEST_COSTS: Costs = Costs {
    jack: Cost { base: 0.0, speed: 3.0 },
    vertical_footswitch: Cost { base: 3.0, speed: 1.0 },
    sideswitch: Cost { base: 3.0, speed: 1.0 },
    horizontal_crossover: Cost { base: 4.0, speed: 1.0 },
    vertical_crossover: Cost { base: 0.0, speed: 0.0 },
    doublestep: Cost { base: 1.0, speed: 7.0 },
    bracket: Cost { base: 6.0, speed: 1.0 },
    freeze_switch: Cost { base: 12.0, speed: 1.0 },
    lateral: Cost { base: 8.0, speed: 2.0 },
    half_spin: Cost { base: 10.0, speed: 2.0 },
    full_spin: Cost { base: 16.0, speed: 4.0 },
    reference_gap: 0.125,
    max_speed: 4.0,
};

/// Costs are whole thousandths of a penalty point, so that equal footings cost exactly the same.
const COST_UNIT: f64 = 1000.0;

impl Costs {
    /// Every number in the costs, by name.
    pub fn parameters(&mut self) -> [(&'static str, &mut f64); 24] {
        let Costs {
            jack,
            vertical_footswitch,
            sideswitch,
            horizontal_crossover,
            vertical_crossover,
            doublestep,
            bracket,
            freeze_switch,
            lateral,
            half_spin,
            full_spin,
            reference_gap,
            max_speed,
        } = self;
        [
            ("jack base", &mut jack.base),
            ("jack speed", &mut jack.speed),
            ("vertical footswitch base", &mut vertical_footswitch.base),
            ("vertical footswitch speed", &mut vertical_footswitch.speed),
            ("sideswitch base", &mut sideswitch.base),
            ("sideswitch speed", &mut sideswitch.speed),
            ("horizontal crossover base", &mut horizontal_crossover.base),
            ("horizontal crossover speed", &mut horizontal_crossover.speed),
            ("vertical crossover base", &mut vertical_crossover.base),
            ("vertical crossover speed", &mut vertical_crossover.speed),
            ("doublestep base", &mut doublestep.base),
            ("doublestep speed", &mut doublestep.speed),
            ("bracket base", &mut bracket.base),
            ("bracket speed", &mut bracket.speed),
            ("freeze switch base", &mut freeze_switch.base),
            ("freeze switch speed", &mut freeze_switch.speed),
            ("lateral base", &mut lateral.base),
            ("lateral speed", &mut lateral.speed),
            ("half spin base", &mut half_spin.base),
            ("half spin speed", &mut half_spin.speed),
            ("full spin base", &mut full_spin.base),
            ("full spin speed", &mut full_spin.speed),
            ("reference gap", reference_gap),
            ("max speed", max_speed),
        ]
    }

    /// 0 for an infinite gap: where a segment has no note before or after.
    fn speed(&self, gap: f64) -> f64 {
        (self.reference_gap / gap).min(self.max_speed)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Events {
    pub footing: Footing,
    /// Repeated arrow, same foot. Per arrow: a repeated jump is two.
    pub jacks: usize,
    /// Repeated Up or Down, other foot. Per arrow.
    pub vertical_footswitches: usize,
    /// Repeated Left or Right, other foot. Per arrow.
    pub sideswitches: usize,
    /// A candle: a foot leaves Up for Down or Down for Up, keeping no arrow. Not in a freeze
    /// switch, nor once the other foot has taken its arrow.
    pub vertical_crossovers: usize,
    /// Same foot alone on two notes in a row, onto a new arrow. Under the other foot's hold too.
    pub doublesteps: usize,
    /// A foot stepping onto two arrows at once, or onto one while it keeps holding another
    pub brackets: usize,
    /// The holding foot steps the note and the other foot takes its hold over
    pub freeze_switches: usize,
}

impl AddAssign for Events {
    fn add_assign(&mut self, other: Events) {
        let Events {
            footing,
            jacks,
            vertical_footswitches,
            sideswitches,
            vertical_crossovers,
            doublesteps,
            brackets,
            freeze_switches,
        } = other;
        self.footing += footing;
        self.jacks += jacks;
        self.vertical_footswitches += vertical_footswitches;
        self.sideswitches += sideswitches;
        self.vertical_crossovers += vertical_crossovers;
        self.doublesteps += doublesteps;
        self.brackets += brackets;
        self.freeze_switches += freeze_switches;
    }
}

impl Events {
    /// How many of each technique.
    pub fn counts(&self) -> [(Technique, usize); 11] {
        [
            (Technique::Jack, self.jacks),
            (Technique::VerticalFootswitch, self.vertical_footswitches),
            (Technique::Sideswitch, self.sideswitches),
            (Technique::HorizontalCrossover, self.footing.horizontal_crossovers),
            (Technique::VerticalCrossover, self.vertical_crossovers),
            (Technique::Doublestep, self.doublesteps),
            (Technique::Bracket, self.brackets),
            (Technique::FreezeSwitch, self.freeze_switches),
            (Technique::Lateral, self.footing.laterals),
            (Technique::HalfSpin, self.footing.half_spins),
            (Technique::FullSpin, self.footing.full_spins),
        ]
    }

    /// The events added since `earlier`.
    fn since(&self, earlier: &Events) -> Events {
        Events {
            footing: Footing {
                horizontal_crossovers: self.footing.horizontal_crossovers - earlier.footing.horizontal_crossovers,
                laterals: self.footing.laterals - earlier.footing.laterals,
                half_spins: self.footing.half_spins - earlier.footing.half_spins,
                full_spins: self.footing.full_spins - earlier.footing.full_spins,
            },
            jacks: self.jacks - earlier.jacks,
            vertical_footswitches: self.vertical_footswitches - earlier.vertical_footswitches,
            sideswitches: self.sideswitches - earlier.sideswitches,
            vertical_crossovers: self.vertical_crossovers - earlier.vertical_crossovers,
            doublesteps: self.doublesteps - earlier.doublesteps,
            brackets: self.brackets - earlier.brackets,
            freeze_switches: self.freeze_switches - earlier.freeze_switches,
        }
    }

    /// In thousandths, with the horizontal crossovers, laterals and spins at `position_speed` and
    /// the rest at `step_speed`.
    fn cost(&self, costs: &Costs, step_speed: f64, position_speed: f64) -> u64 {
        self.jacks as u64 * costs.jack.at(step_speed)
            + self.vertical_footswitches as u64 * costs.vertical_footswitch.at(step_speed)
            + self.sideswitches as u64 * costs.sideswitch.at(step_speed)
            + self.vertical_crossovers as u64 * costs.vertical_crossover.at(step_speed)
            + self.doublesteps as u64 * costs.doublestep.at(step_speed)
            + self.brackets as u64 * costs.bracket.at(step_speed)
            + self.freeze_switches as u64 * costs.freeze_switch.at(step_speed)
            + self.footing.horizontal_crossovers as u64 * costs.horizontal_crossover.at(position_speed)
            + self.footing.laterals as u64 * costs.lateral.at(position_speed)
            + self.footing.half_spins as u64 * costs.half_spin.at(position_speed)
            + self.footing.full_spins as u64 * costs.full_spin.at(position_speed)
    }

    /// `cost` of events added at note `i` of `segment`.
    fn cost_at(&self, costs: &Costs, segment: &[TimedNote], i: usize) -> u64 {
        let (before, after) = gaps(segment, i);
        self.cost(costs, costs.speed(before), costs.speed(before.min(after)))
    }
}

/// A set of arrows: bit `arrow` (0 to 3: Left, Down, Up, Right).
type Arrows = u8;

fn bit(arrow: usize) -> Arrows {
    1 << arrow
}

fn has(arrows: Arrows, arrow: usize) -> bool {
    arrows & bit(arrow) != 0
}

/// One arrow, or two a foot can bracket: any pair but Left with Right and Down with Up.
fn fits_a_foot(arrows: Arrows) -> bool {
    match arrows.count_ones() {
        1 => true,
        2 => arrows != bit(LEFT_ARROW) | bit(RIGHT_ARROW) && arrows != bit(DOWN_ARROW) | bit(UP_ARROW),
        _ => false,
    }
}

/// As `facing`, with a bracketing foot midway between its arrows. None unless both feet
/// are on an arrow.
fn facing_of(on: [Arrows; 2]) -> Option<i32> {
    if on[0] == 0 || on[1] == 0 {
        return None;
    }
    // Twice the foot's (x, y)
    let place = |arrows: Arrows| {
        let (mut x, mut y) = (0, 0);
        for arrow in (0..4).filter(|&arrow| has(arrows, arrow)) {
            x += ARROW_POSITIONS[arrow].0;
            y += ARROW_POSITIONS[arrow].1;
        }
        let times = if arrows.count_ones() == 1 { 2 } else { 1 };
        (x * times, y * times)
    };
    let (left, right) = (place(on[0]), place(on[1]));
    let (dx, dy) = ((right.0 - left.0) as f64, (right.1 - left.1) as f64);
    Some(wrap((-dy).atan2(dx).to_degrees().round() as i32))
}

/// How one note is stepped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    /// The arrows the (left, right) foot steps. Neither if both feet are held down out of reach.
    feet: [Arrows; 2],
    /// Freeze switch: the held arrows the (left, right) foot takes over from the other
    regrabs: [Arrows; 2],
}

impl Step {
    /// The foot that steps `arrow` (0 to 3: Left, Down, Up, Right).
    pub fn foot(&self, arrow: usize) -> Option<Foot> {
        [Foot::Left, Foot::Right].into_iter().find(|&foot| has(self.feet[foot as usize], arrow))
    }

    /// The foot that takes over the hold on `arrow` from the other, in a freeze switch.
    pub fn regrabber(&self, arrow: usize) -> Option<Foot> {
        [Foot::Left, Foot::Right].into_iter().find(|&foot| has(self.regrabs[foot as usize], arrow))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// One per note
    pub steps: Vec<Step>,
    /// Each event, on the note whose step adds it, and each pattern, on its first note, in note
    /// order. A spin is where its turn ends: the note that turns back, or the last.
    pub occurrences: Vec<Occurrence>,
    pub events: Events,
    /// The events' penalties, each at its speed
    pub cost: f64,
    /// One per note: the penalties of the events its step adds. They add up to `cost`.
    pub note_costs: Vec<f64>,
    /// Each foot's steps after its first
    pub moves: Vec<moves::Move>,
    /// Branches tried
    pub nodes: u64,
    /// Branches that reached a position another had reached
    pub merged: u64,
    /// Most branches left after any note
    pub widest: usize,
    /// False if branches were dropped: `steps` may not be the cheapest
    pub exact: bool,
}

impl Solution {
    /// What is found on note `note`.
    pub fn at(&self, note: usize) -> impl Iterator<Item = &Occurrence> {
        self.occurrences.iter().filter(move |occurrence| occurrence.note == note)
    }

    #[cfg(test)]
    pub fn of(&self, technique: Technique) -> impl Iterator<Item = &Occurrence> {
        self.occurrences.iter().filter(move |occurrence| occurrence.technique == technique)
    }

    #[cfg(test)]
    pub fn count(&self, technique: Technique) -> usize {
        self.of(technique).count()
    }
}

/// All that the rest of a segment depends on.
#[derive(PartialEq, Eq, Hash)]
struct Key {
    on: [Arrows; 2],
    /// Per arrow: bits of the hold end, or of -inf if it's over by the next note
    held_until: [u64; 4],
    last: Last,
    /// (facing, start of the turn in progress, direction)
    turn: Option<(i32, i32, i32)>,
}

#[derive(Clone, Copy)]
struct State {
    /// Arrows under the (left, right) foot: two in a bracket, none once the other foot takes them
    on: [Arrows; 2],
    /// Per arrow: when the hold or roll on it ends
    held_until: [f64; 4],
    last: Last,
    stance: Stance,
    turning: Turning,
    events: Events,
    /// In thousandths
    cost: u64,
}

impl State {
    fn home() -> State {
        let on = [bit(LEFT_ARROW), bit(RIGHT_ARROW)];
        State {
            on,
            held_until: [f64::NEG_INFINITY; 4],
            last: Last::Nothing,
            stance: Stance::of(facing_of(on)),
            turning: Turning::new(facing_of(on)),
            events: Events::default(),
            cost: 0,
        }
    }

    /// `next_time`: when the next note is, or infinity after the last.
    fn key(&self, next_time: f64) -> Key {
        let held_until = self
            .held_until
            .map(|until| if until <= next_time + TIME_EPSILON { f64::NEG_INFINITY } else { until }.to_bits());
        let turn = self.turning.angle.map(|angle| {
            // Whole turns made before the turn in progress are already counted
            let facing = wrap(angle);
            (facing, self.turning.turn_start - (angle - facing), self.turning.direction)
        });
        Key { on: self.on, held_until, last: self.last, turn }
    }

    /// The arrows `foot` is holding down at `time`, which it can't leave.
    fn held(&self, foot: Foot, time: f64) -> Arrows {
        (0..4)
            .filter(|&arrow| has(self.on[foot as usize], arrow) && self.held_until[arrow] > time + TIME_EPSILON)
            .fold(0, |held, arrow| held | bit(arrow))
    }

    fn foot_on(&self, arrow: usize) -> Option<Foot> {
        [Foot::Left, Foot::Right].into_iter().find(|&foot| has(self.on[foot as usize], arrow))
    }

    /// The foot that faces the player more squarely to the screen by stepping `arrow`, or `tie`.
    fn squarer_foot(&self, arrow: usize, tie: Foot) -> Foot {
        let turn = |foot: Foot| {
            let mut after = self.on;
            after[foot as usize] = bit(arrow);
            after[foot.other() as usize] &= !bit(arrow);
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

    /// The feet after `step`, or None if a foot can't reach its arrows. `repeats`: the arrows the
    /// note before had too.
    fn step(&self, note: &TimedNote, step: Step, repeats: Arrows) -> Option<State> {
        let held = [Foot::Left, Foot::Right].map(|foot| self.held(foot, note.time));
        let (stepped, regrabs) = (step.feet, step.regrabs);
        let mut next = *self;
        for foot in [Foot::Left, Foot::Right] {
            let (this, other) = (foot as usize, foot.other() as usize);
            if stepped[this] != 0 {
                next.on[this] = (held[this] & !regrabs[other]) | stepped[this];
            } else if regrabs[this] != 0 {
                next.on[this] = held[this] | regrabs[this];
                next.events.freeze_switches += 1;
            } else {
                // Makes way
                next.on[this] &= !stepped[other];
                continue;
            }
            if !fits_a_foot(next.on[this]) {
                return None;
            }
            if next.on[this].count_ones() == 2 {
                next.events.brackets += 1;
            }
            let vertical = bit(DOWN_ARROW) | bit(UP_ARROW);
            let (before, after) = (self.on[this], next.on[this]);
            if regrabs == [0, 0] && before & after == 0 && (before | after) & vertical == vertical {
                next.events.vertical_crossovers += 1;
            }
            for arrow in (0..4).filter(|&arrow| has(stepped[this] & repeats, arrow)) {
                if has(self.on[this], arrow) {
                    next.events.jacks += 1;
                } else if has(self.on[other], arrow) && has(vertical, arrow) {
                    next.events.vertical_footswitches += 1;
                } else if has(self.on[other], arrow) {
                    next.events.sideswitches += 1;
                }
            }
            let alone = stepped[other] == 0;
            let moves = stepped[this] & !self.on[this] != 0;
            if alone && moves && matches!(self.last, Last::Foot(last) if last == foot) {
                next.events.doublesteps += 1;
            }
        }
        for arrow in (0..4).filter(|&arrow| has(stepped[0] | stepped[1], arrow)) {
            if let Some(end) = note.hold_ends[arrow] {
                next.held_until[arrow] = end;
            }
        }
        next.last = match stepped {
            [_, 0] => Last::Foot(Foot::Left),
            [0, _] => Last::Foot(Foot::Right),
            _ => Last::BothFeet,
        };
        next.settle();
        Some(next)
    }

    /// Counts the stance and turn the feet are now in.
    fn settle(&mut self) {
        let facing = facing_of(self.on);
        let stance = Stance::of(facing);
        if stance == Stance::Crossed && !matches!(self.stance, Stance::Crossed | Stance::Lateral) {
            self.events.footing.horizontal_crossovers += 1;
        }
        if stance == Stance::Lateral && self.stance != Stance::Lateral {
            self.events.footing.laterals += 1;
        }
        self.stance = stance;
        self.turning.face(facing, &mut self.events.footing);
        // Past two whole turns, count one now rather than at the turn's end: the same total, and
        // it keeps the turn in progress (part of `Key`) bounded
        if let Some(angle) = self.turning.angle {
            while (angle - self.turning.turn_start).abs() >= 720 {
                self.turning.turn_start += 360 * (angle - self.turning.turn_start).signum();
                self.events.footing.full_spins += 1;
            }
        }
    }
}

/// Ways to step note `i`, the usual one first.
fn options(segment: &[TimedNote], i: usize, state: &State, costs: &Costs) -> Vec<(Step, State)> {
    let mut options = uncosted_options(segment, i, state);
    for (_, next) in &mut options {
        next.cost += next.events.since(&state.events).cost_at(costs, segment, i);
    }
    options
}

/// `options` before costing.
fn uncosted_options(segment: &[TimedNote], i: usize, state: &State) -> Vec<(Step, State)> {
    let note = &segment[i];
    let row = |note: &TimedNote| (0..4).filter(|&arrow| steps_on(note.row[arrow])).fold(0, |row, arrow| row | bit(arrow));
    let arrows = row(note);
    let repeats = if i > 0 { row(&segment[i - 1]) } else { 0 };
    let listed: Vec<usize> = (0..4).filter(|&arrow| has(arrows, arrow)).collect();

    // The arrows each foot could step, as (left, right)
    let splits: Vec<[Arrows; 2]> = match listed[..] {
        [arrow] => {
            let usual = match (state.foot_on(arrow), state.last) {
                (Some(foot), _) if has(repeats, arrow) => foot,
                (_, Last::Foot(foot)) => foot.other(),
                (_, Last::BothFeet | Last::Nothing) => state.squarer_foot(arrow, Foot::Left),
            };
            if usual == Foot::Left { vec![[arrows, 0], [0, arrows]] } else { vec![[0, arrows], [arrows, 0]] }
        }
        // A foot each, then one foot for both
        [_, _] => {
            let (left, right) = jump_feet(&listed, state.turning.facing());
            vec![[bit(left), bit(right)], [bit(right), bit(left)], [arrows, 0], [0, arrows]]
        }
        // Both feet, one at least bracketing
        _ => (1..arrows).filter(|left| left & !arrows == 0).map(|left| [left, arrows ^ left]).collect(),
    };
    let mut options: Vec<(Step, State)> = Vec::new();
    for feet in splits {
        let mut ways = vec![Step { feet, regrabs: [0, 0] }];
        // A foot alone stepping off its hold, which the other takes over
        for foot in [Foot::Left, Foot::Right] {
            let (this, other) = (foot as usize, foot.other() as usize);
            let held = state.held(foot, note.time);
            if feet[this] != 0 && feet[other] == 0 && held != 0 {
                let mut regrabs = [0, 0];
                regrabs[other] = held;
                ways.push(Step { feet, regrabs });
            }
        }
        options.extend(ways.into_iter().filter_map(|step| state.step(note, step, repeats).map(|next| (step, next))));
    }

    // Freeze switches last. Fewest brackets first. Three or four arrows: then the least turn.
    let turn = |next: &State| match (state.turning.facing(), next.turning.facing()) {
        (Some(before), Some(after)) if listed.len() > 2 => wrap(after - before).abs(),
        _ => 0,
    };
    options.sort_by_key(|(step, next)| (step.regrabs != [0, 0], next.events.brackets, turn(next)));
    if options.is_empty() {
        options.push((Step { feet: [0, 0], regrabs: [0, 0] }, *state));
    }
    options
}

/// A branch after some note.
struct Branch {
    state: State,
    /// Index in the layer before
    parent: usize,
    step: Step,
}

/// The cheapest footing of a segment starting from home. Among equally cheap ones, the one that
/// keeps to the usual foot longest.
pub fn solve(segment: &[TimedNote], costs: &Costs, max_width: usize) -> Solution {
    search(segment, costs, max_width, true)
}

fn search(segment: &[TimedNote], costs: &Costs, max_width: usize, merge: bool) -> Solution {
    // Branches after each note. Kept in the order of their steps, usual before unusual, so the
    // first of equals is the more usual.
    let mut layer = vec![State::home()];
    // Per note: each branch's (parent, step)
    let mut links: Vec<Vec<(usize, Step)>> = Vec::with_capacity(segment.len());
    let (mut nodes, mut merged, mut widest, mut exact) = (0, 0, 1, true);

    for i in 0..segment.len() {
        let next_time = segment.get(i + 1).map_or(f64::INFINITY, |note| note.time);
        // None: merged into a cheaper branch
        let mut next: Vec<Option<Branch>> = Vec::new();
        let mut reached: HashMap<Key, usize> = HashMap::new();
        for (parent, state) in layer.iter().enumerate() {
            for (step, state) in options(segment, i, state, costs) {
                nodes += 1;
                if merge {
                    match reached.entry(state.key(next_time)) {
                        Entry::Occupied(mut entry) => {
                            merged += 1;
                            let other = *entry.get();
                            if next[other].as_ref().is_some_and(|other| other.state.cost <= state.cost) {
                                continue;
                            }
                            next[other] = None;
                            entry.insert(next.len());
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(next.len());
                        }
                    }
                }
                next.push(Some(Branch { state, parent, step }));
            }
        }
        let mut next: Vec<Branch> = next.into_iter().flatten().collect();
        if next.len() > max_width {
            next.sort_by_key(|branch| branch.state.cost);
            next.truncate(max_width);
            exact = false;
        }
        widest = widest.max(next.len());
        links.push(next.iter().map(|branch| (branch.parent, branch.step)).collect());
        layer = next.into_iter().map(|branch| branch.state).collect();
    }

    let mut best: Option<(u64, Events, usize)> = None;
    for (index, state) in layer.iter().enumerate() {
        let mut end = *state;
        // The last turn's spins, on the last note
        end.turning.finish(&mut end.events.footing);
        let last = segment.len().saturating_sub(1);
        let cost = state.cost + end.events.since(&state.events).cost_at(costs, segment, last);
        if best.is_none_or(|(best_cost, ..)| cost < best_cost) {
            best = Some((cost, end.events, index));
        }
    }
    let (cost, events, mut index) = best.expect("a branch is always left");
    let mut steps = Vec::with_capacity(segment.len());
    for links in links.iter().rev() {
        let (parent, step) = links[index];
        steps.push(step);
        index = parent;
    }
    steps.reverse();
    let (step_events, kinds) = replay(segment, &steps, &events);
    let mut occurrences: Vec<Occurrence> = Vec::new();
    for (i, added) in step_events.iter().enumerate() {
        for (technique, count) in added.counts() {
            occurrences.extend((0..count).map(|_| Occurrence::at(technique, segment, i)));
        }
    }
    occurrences.extend(kinds.into_iter().map(|(i, kind)| Occurrence::at(kind, segment, i)));
    occurrences.extend(patterns::find(segment, &steps, &step_events));
    // Stable: a note's events stay before their kinds, and those before its patterns
    occurrences.sort_by_key(|occurrence| occurrence.note);
    let note_costs: Vec<u64> = step_events.iter().enumerate().map(|(i, added)| added.cost_at(costs, segment, i)).collect();
    debug_assert_eq!(note_costs.iter().sum::<u64>(), cost);
    let in_points = |cost: u64| cost as f64 / COST_UNIT;
    let note_costs = note_costs.into_iter().map(in_points).collect();
    let moves = moves::find(segment, &steps);
    Solution { steps, occurrences, events, cost: in_points(cost), note_costs, moves, nodes, merged, widest, exact }
}

/// Replays `steps` for the events each adds, and for which kind each vertical footswitch and
/// horizontal crossover is (see `Technique::kinds`), as (note, kind). `total`: the events once
/// the last turn has ended.
fn replay(segment: &[TimedNote], steps: &[Step], total: &Events) -> (Vec<Events>, Vec<(usize, Technique)>) {
    let mut state = State::home();
    // Per foot: the arrows it is on or was last on, and those it came there from
    let (mut stands, mut came_from) = (state.on, state.on);
    let (mut added, mut kinds) = (Vec::with_capacity(steps.len()), Vec::new());
    for (i, step) in steps.iter().enumerate() {
        let (_, next) = uncosted_options(segment, i, &state)
            .into_iter()
            .find(|(option, _)| option == step)
            .expect("a step is one of its note's options");
        for foot in 0..2 {
            if next.on[foot] != 0 && next.on[foot] != stands[foot] {
                (came_from[foot], stands[foot]) = (stands[foot], next.on[foot]);
            }
        }
        let events = next.events.since(&state.events);
        // A footswitch: the arrow the other foot was on, repeated
        let switched = |arrow: usize| {
            let repeated = segment[i].steps(arrow) && i > 0 && segment[i - 1].steps(arrow);
            repeated && step.foot(arrow).is_some_and(|foot| has(state.on[foot.other() as usize], arrow))
        };
        let footswitches = [(UP_ARROW, Technique::UpFootswitch), (DOWN_ARROW, Technique::DownFootswitch)];
        let before = kinds.len();
        kinds.extend(footswitches.into_iter().filter(|&(arrow, _)| switched(arrow)).map(|(_, kind)| (i, kind)));
        debug_assert_eq!(kinds.len() - before, events.vertical_footswitches);
        if events.footing.horizontal_crossovers > 0 {
            // The foot across the pad: the left on Right, or else the right on Left
            let (crossing, home) = if has(next.on[Foot::Left as usize], RIGHT_ARROW) {
                (Foot::Left, LEFT_ARROW)
            } else {
                (Foot::Right, RIGHT_ARROW)
            };
            let from_home = has(came_from[crossing as usize], home);
            kinds.push((i, if from_home { Technique::FullCrossover } else { Technique::HalfCrossover }));
        }
        added.push(events);
        state = next;
    }
    if let Some(last) = added.last_mut() {
        *last += total.since(&state.events);
    }
    (added, kinds)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::chart_analyzer::technique::Source;
    use crate::chart_analyzer::{Timing, to_timed_notes};
    use crate::sim_loader::{BPMChange, Measure, Note};

    /// A segment from tokens 0.125 s apart, one row each: `L`, `D`, `U`, `R` for taps (`LR` is a
    /// jump), a trailing `+` for hold heads, a trailing `-` for hold ends, `.` for an empty row.
    pub(in crate::chart_analyzer) fn segment(tokens: &str) -> Vec<TimedNote> {
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


    const COSTS: Costs = TEST_COSTS;
    const WIDTH: usize = 1_000_000;

    /// `costs` with every penalty what it is at the reference speed, at any speed.
    fn flat(costs: &Costs) -> Costs {
        let flat = |cost: Cost| Cost { base: cost.base + cost.speed, speed: 0.0 };
        let Costs {
            jack,
            vertical_footswitch,
            sideswitch,
            horizontal_crossover,
            vertical_crossover,
            doublestep,
            bracket,
            freeze_switch,
            lateral,
            half_spin,
            full_spin,
            reference_gap,
            max_speed,
        } = *costs;
        Costs {
            jack: flat(jack),
            vertical_footswitch: flat(vertical_footswitch),
            sideswitch: flat(sideswitch),
            horizontal_crossover: flat(horizontal_crossover),
            vertical_crossover: flat(vertical_crossover),
            doublestep: flat(doublestep),
            bracket: flat(bracket),
            freeze_switch: flat(freeze_switch),
            lateral: flat(lateral),
            half_spin: flat(half_spin),
            full_spin: flat(full_spin),
            reference_gap,
            max_speed,
        }
    }

    /// `l` or `r` per note one foot steps one arrow of. Otherwise the left foot's arrows, `|`,
    /// the right foot's, in brackets: `[L|R]` for a jump, `[LU|]` for a bracket. Then `*` for a
    /// freeze switch.
    fn feet(solution: &Solution) -> String {
        let letters = |arrows: Arrows| -> String { (0..4).filter(|&arrow| has(arrows, arrow)).map(|arrow| &"LDUR"[arrow..=arrow]).collect() };
        solution
            .steps
            .iter()
            .map(|step| {
                let feet = match step.feet {
                    [left, 0] if left.count_ones() == 1 => "l".to_string(),
                    [0, right] if right.count_ones() == 1 => "r".to_string(),
                    [left, right] => format!("[{}|{}]", letters(left), letters(right)),
                };
                feet + if step.regrabs == [0, 0] { "" } else { "*" }
            })
            .collect()
    }

    /// `footswitches`: vertical ones.
    fn events(
        jacks: usize,
        footswitches: usize,
        crossovers: usize,
        doublesteps: usize,
        laterals: usize,
        half_spins: usize,
        full_spins: usize,
    ) -> Events {
        Events {
            footing: Footing { horizontal_crossovers: crossovers, laterals, half_spins, full_spins },
            jacks,
            vertical_footswitches: footswitches,
            sideswitches: 0,
            vertical_crossovers: 0,
            doublesteps,
            brackets: 0,
            freeze_switches: 0,
        }
    }

    fn bracketed(brackets: usize, events: Events) -> Events {
        Events { brackets, ..events }
    }

    fn switched(freeze_switches: usize, events: Events) -> Events {
        Events { freeze_switches, ..events }
    }

    fn candled(vertical_crossovers: usize, events: Events) -> Events {
        Events { vertical_crossovers, ..events }
    }

    fn solved(tokens: &str, costs: &Costs) -> (String, Events) {
        let notes = segment(tokens);
        let solution = solve(&notes, costs, WIDTH);
        assert!(solution.exact, "{tokens}: branches dropped");
        // `segment` spaces notes at the reference gap
        assert_eq!(solution.cost, solution.events.cost(costs, 1.0, 1.0) as f64 / COST_UNIT, "{tokens}: cost");
        let mut added = Events::default();
        for events in replay(&notes, &solution.steps, &solution.events).0 {
            added += events;
        }
        assert_eq!(added, solution.events, "{tokens}: events step by step");
        for (technique, count) in solution.events.counts() {
            assert_eq!(solution.count(technique), count, "{tokens}: {} found", technique.plural());
        }
        (feet(&solution), solution.events)
    }

    /// The events on note `note`, with their speeds.
    fn found_on(solution: &Solution, note: usize) -> Vec<(Technique, f64)> {
        solution
            .at(note)
            .filter(|occurrence| occurrence.technique.source() == Source::FootingEvent)
            .map(|occurrence| (occurrence.technique, occurrence.speed))
            .collect()
    }

    #[test]
    fn names_every_cost() {
        let mut costs = COSTS;
        let names: Vec<&str> = costs.parameters().iter().map(|(name, _)| *name).collect();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len());
        // Each is its own number
        for (index, (_, value)) in costs.parameters().into_iter().enumerate() {
            *value = 100.0 + index as f64;
        }
        let values: Vec<f64> = costs.parameters().iter().map(|(_, value)| **value).collect();
        assert_eq!(values, (0..24).map(|index| 100.0 + index as f64).collect::<Vec<_>>());
        assert_eq!((costs.doublestep.speed, costs.max_speed), (111.0, 123.0));
    }

    #[test]
    fn alternates_when_nothing_is_awkward() {
        assert_eq!(solved("L D U R L D U R", &COSTS), ("lrlrlrlr".into(), Events::default()));
        // Starts on the foot whose side the first arrow is on
        assert_eq!(solved("R L R L", &COSTS), ("rlrl".into(), Events::default()));
        assert_eq!(solved("U R U R", &COSTS), ("lrlr".into(), Events::default()));
        assert_eq!(solved("U L U L", &COSTS), ("rlrl".into(), Events::default()));
    }

    #[test]
    fn crosses_over_rather_than_doublestep() {
        assert_eq!(solved("L D R D L", &COSTS), ("lrlrl".into(), events(0, 0, 1, 0, 0, 0, 0)));
        assert_eq!(solved("R U L U R", &COSTS), ("rlrlr".into(), events(0, 0, 1, 0, 0, 0, 0)));
    }

    #[test]
    fn jacks_or_footswitches_a_repeated_arrow() {
        // A jack then Left again is free of anything worse
        assert_eq!(solved("L D D L", &COSTS), ("lrrl".into(), events(1, 0, 0, 0, 0, 0, 0)));
        // A jack here would cross the left foot over to Right. Switching feet doesn't.
        assert_eq!(solved("L D D R", &COSTS), ("lrlr".into(), events(0, 1, 0, 0, 0, 0, 0)));
        assert_eq!(solved("L U U R L R", &COSTS), ("lrlrlr".into(), events(0, 1, 0, 0, 0, 0, 0)));
        // A run on one arrow is all jacks
        assert_eq!(solved("U U U U", &COSTS), ("llll".into(), events(3, 0, 0, 0, 0, 0, 0)));
    }

    #[test]
    fn avoids_a_lateral() {
        // Left foot first: crossover, lateral and half spin. Right foot first: one crossover.
        assert_eq!(solved("L D R L", &COSTS), ("rlrl".into(), events(0, 0, 1, 0, 0, 0, 0)));
        // The first two notes fix the feet, so a doublestep is the cheapest way out
        assert_eq!(solved("R L D R L", &COSTS), ("rlrrl".into(), events(0, 0, 0, 1, 0, 0, 0)));
    }

    #[test]
    fn costs_decide_between_spinning_and_doublestepping() {
        assert_eq!(solved("L D R U L D", &COSTS), ("rlrllr".into(), candled(1, events(0, 0, 1, 1, 0, 0, 0))));
        let spinner = Costs { doublestep: Cost { base: 30.0, speed: 0.0 }, ..COSTS };
        assert_eq!(solved("L D R U L D", &spinner), ("lrlrlr".into(), candled(2, events(0, 0, 1, 0, 0, 0, 1))));
    }

    /// `tokens` with the notes `times` as fast.
    fn at_speed(tokens: &str, times: f64) -> Vec<TimedNote> {
        let faster = |note: TimedNote| TimedNote {
            time: note.time / times,
            hold_ends: note.hold_ends.map(|end| end.map(|end| end / times)),
            ..note
        };
        segment(tokens).into_iter().map(faster).collect()
    }

    #[test]
    fn costs_each_event_by_its_own_speed_scaling() {
        assert_eq!(COSTS.speed(0.125), 1.0);
        assert_eq!(COSTS.speed(0.0625), 2.0);
        assert_eq!(COSTS.speed(0.25), 0.5);
        assert_eq!(COSTS.speed(f64::INFINITY), 0.0);
        // Capped
        assert_eq!(COSTS.speed(0.0), 4.0);
        assert_eq!(Cost { base: 1.0, speed: 7.0 }.at(0.5), 4500);

        // The same crossover at other speeds: its base, plus its speed scaling
        let crossover = |times: f64| solve(&at_speed("L D R D L", times), &COSTS, WIDTH).cost;
        assert_eq!(crossover(1.0), COSTS.horizontal_crossover.base + COSTS.horizontal_crossover.speed);
        assert_eq!(crossover(2.0), COSTS.horizontal_crossover.base + COSTS.horizontal_crossover.speed * 2.0);
        assert_eq!(crossover(0.25), COSTS.horizontal_crossover.base + COSTS.horizontal_crossover.speed * 0.25);
        assert_eq!(solve(&at_speed("L D R D L", 2.0), &flat(&COSTS), WIDTH).cost, crossover(1.0));

        // A crossover is no cheaper for a pause before it, if the next note comes fast
        let paused = solve(&segment("L D . . . R D L"), &COSTS, WIDTH);
        assert_eq!((paused.events, paused.cost), (events(0, 0, 1, 0, 0, 0, 0), crossover(1.0)));
        // Nor a crossed jump for starting a segment
        assert_eq!(solved("LU UR", &COSTS).1, events(0, 1, 0, 0, 0, 0, 0));
    }

    #[test]
    fn speed_decides_between_jacks_and_footswitches() {
        let footing = |times: f64| {
            let solution = solve(&at_speed("U U U U", times), &COSTS, WIDTH);
            (feet(&solution), solution.events)
        };
        // One foot can repeat an arrow up to a point
        assert_eq!(footing(0.5), ("llll".into(), events(3, 0, 0, 0, 0, 0, 0)));
        assert_eq!(footing(1.0), ("llll".into(), events(3, 0, 0, 0, 0, 0, 0)));
        // Past it, the feet take turns
        assert_eq!(footing(2.0), ("lrlr".into(), events(0, 3, 0, 0, 0, 0, 0)));
        assert_eq!(footing(4.0), ("lrlr".into(), events(0, 3, 0, 0, 0, 0, 0)));
    }

    #[test]
    fn speed_decides_between_a_crossover_and_a_doublestep() {
        // A doublestep costs little slow and a lot fast, a crossover much the same
        let footing = |times: f64| {
            let solution = solve(&at_speed("R L U R U R", times), &COSTS, WIDTH);
            (feet(&solution), solution.events)
        };
        assert_eq!(footing(2.0), ("rlrlrl".into(), events(0, 0, 1, 0, 0, 0, 0)));
        assert_eq!(footing(1.0), ("rlrlrl".into(), events(0, 0, 1, 0, 0, 0, 0)));
        assert_eq!(footing(0.25), ("rlrrlr".into(), events(0, 0, 0, 1, 0, 0, 0)));
        // Without speed scaling, a crossover at any speed
        let flat = flat(&COSTS);
        assert_eq!(feet(&solve(&at_speed("R L U R U R", 0.25), &flat, WIDTH)), "rlrlrl");

        // After a pause, a doublestep onto Up rather than the crossover that would follow
        let paused = solve(&segment("R L . . . U R U R"), &COSTS, WIDTH);
        assert_eq!((feet(&paused), paused.events), ("rllrlr".into(), events(0, 0, 0, 1, 0, 0, 0)));
        assert_eq!(paused.cost, COSTS.doublestep.base + COSTS.doublestep.speed * 0.25);
        assert_eq!(feet(&solve(&segment("R L . . . U R U R"), &flat, WIDTH)), "rlrlrl");
    }

    #[test]
    fn flips_back_toward_the_screen_whichever_way_it_has_turned() {
        // Facing each way in turn, then the turn ended: (full spins, half spins) and the angle
        let turned = |facings: &[i32]| {
            let (mut turning, mut footing) = (Turning::new(Some(0)), Footing::default());
            for &facing in facings {
                turning.face(Some(facing), &mut footing);
            }
            let angle = turning.angle;
            turning.finish(&mut footing);
            ((footing.full_spins, footing.half_spins), angle)
        };
        // Three quarters round clockwise, facing left. The flip to facing right goes on round
        // past the screen: one full spin, not a half spin there and another back.
        assert_eq!(turned(&[90, 180, -90, 90]), ((1, 0), Some(450)));
        assert_eq!(turned(&[-90, 180, 90, -90]), ((1, 0), Some(-450)));
        // A quarter round, the flip comes back past the screen: no spin at all
        assert_eq!(turned(&[90, -90]), ((0, 0), Some(-90)));
        assert_eq!(turned(&[-90, 90]), ((0, 0), Some(90)));
        // Without a flip, three quarters round is a half spin
        assert_eq!(turned(&[90, 180, -90]), ((0, 1), Some(270)));
    }

    #[test]
    fn says_which_step_each_event_is_on() {
        // Notes 0.125 s apart: 8 a second
        let crossover = solve(&segment("L D R D L"), &COSTS, WIDTH);
        let on_right = |technique| Occurrence { technique, note: 2, notes: 1, speed: 8.0 };
        assert_eq!(crossover.occurrences, [on_right(Technique::HorizontalCrossover), on_right(Technique::FullCrossover)]);
        // The spin is counted on the last note
        let spinner = Costs { doublestep: Cost { base: 30.0, speed: 0.0 }, ..COSTS };
        let spin = solve(&segment("L D R U L D"), &spinner, WIDTH);
        assert_eq!(found_on(&spin, 2), [(Technique::HorizontalCrossover, 8.0), (Technique::FullCrossover, 8.0)]);
        assert_eq!(found_on(&spin, 5), [(Technique::VerticalCrossover, 8.0), (Technique::FullSpin, 8.0)]);
    }

    #[test]
    fn tells_up_footswitches_from_down() {
        let no_jacks = Costs { jack: Cost { base: 30.0, speed: 0.0 }, ..COSTS };
        let kinds = |tokens: &str, costs: &Costs| {
            let solution = solve(&segment(tokens), costs, WIDTH);
            let kinds: Vec<(usize, Technique)> = solution
                .occurrences
                .iter()
                .filter(|found| Technique::VerticalFootswitch.kinds().contains(&found.technique))
                .map(|found| (found.note, found.technique))
                .collect();
            assert_eq!(kinds.len(), solution.count(Technique::VerticalFootswitch), "{tokens}: each footswitch has a kind");
            kinds
        };
        assert_eq!(kinds("L D D R", &COSTS), [(2, Technique::DownFootswitch)]);
        assert_eq!(kinds("U U", &no_jacks), [(1, Technique::UpFootswitch)]);
        assert_eq!(kinds("D D D", &no_jacks), [(1, Technique::DownFootswitch), (2, Technique::DownFootswitch)]);
        // Only the foot on Up switches
        assert_eq!(kinds("LU UR", &COSTS), [(1, Technique::UpFootswitch)]);
        // A sideswitch is neither, nor is a jack
        assert_eq!(kinds("L L", &no_jacks), []);
        assert_eq!(kinds("U U", &COSTS), []);
        // Timed as the footswitch it is
        let switched = solve(&segment("L D . D R"), &COSTS, WIDTH);
        assert_eq!(found_on(&switched, 2), [(Technique::VerticalFootswitch, 4.0), (Technique::DownFootswitch, 4.0)]);
    }

    #[test]
    fn tells_full_crossovers_from_half() {
        let costs = Costs { doublestep: Cost { base: 100.0, speed: 0.0 }, ..COSTS };
        let kinds = |tokens: &str| {
            let solution = solve(&segment(tokens), &costs, WIDTH);
            let kinds: Vec<(usize, Technique)> = Technique::HorizontalCrossover
                .kinds()
                .iter()
                .flat_map(|&kind| solution.of(kind).map(move |found| (found.note, kind)))
                .collect();
            assert_eq!(kinds.len(), solution.count(Technique::HorizontalCrossover), "{tokens}: each crossover has a kind");
            (feet(&solution), kinds)
        };
        // The left foot goes from Left to Right, past the right foot on Down
        assert_eq!(kinds("L D R D L"), ("lrlrl".into(), vec![(2, Technique::FullCrossover)]));
        // From Up to Right: it was already beside the right foot
        assert_eq!(kinds("L R U D R"), ("lrlrl".into(), vec![(4, Technique::HalfCrossover)]));
        // Mirrored
        assert_eq!(kinds("R D L D R"), ("rlrlr".into(), vec![(2, Technique::FullCrossover)]));
        assert_eq!(kinds("R L U D L"), ("rlrlr".into(), vec![(4, Technique::HalfCrossover)]));
        // Uncrossed, then across again from the arrow it uncrossed to
        assert_eq!(kinds("L D R D L D R"), ("lrlrlrl".into(), vec![(2, Technique::FullCrossover), (6, Technique::FullCrossover)]));
        // The right foot starts on Left, straight from home. It's a crossover once the left foot
        // is on Down.
        assert_eq!(kinds("L D R D U D R"), ("rlrlrlr".into(), vec![(1, Technique::FullCrossover)]));
    }

    #[test]
    fn times_each_event() {
        // The crossover onto Right: 0.25 s after Down, 0.125 s before the next. A position goes
        // by the shorter gap.
        let costs = Costs { doublestep: Cost { base: 100.0, speed: 0.0 }, ..COSTS };
        let crossover = solve(&segment("L D . R D L"), &costs, WIDTH);
        assert_eq!(found_on(&crossover, 2), [(Technique::HorizontalCrossover, 8.0), (Technique::FullCrossover, 8.0)]);
        let slow = solve(&segment("L D . R . D L"), &costs, WIDTH);
        assert_eq!(found_on(&slow, 2), [(Technique::HorizontalCrossover, 4.0), (Technique::FullCrossover, 4.0)]);
        // A step goes by the gap before it
        let doublestep = solve(&segment("L . . . D . . . R D"), &COSTS, WIDTH);
        assert_eq!(found_on(&doublestep, 2), [(Technique::Doublestep, 2.0)]);
        // A repeated jump: two jacks on one note
        let jumps = solve(&segment("LR . LR"), &COSTS, WIDTH);
        assert_eq!(found_on(&jumps, 1), [(Technique::Jack, 4.0), (Technique::Jack, 4.0)]);
        assert_eq!(jumps.count(Technique::Jack), 2);
    }

    #[test]
    fn costs_each_note() {
        // Slow enough to doublestep onto Right, 0.5 s after Down
        let doublestep = solve(&segment("L . . . D . . . R"), &COSTS, WIDTH);
        assert_eq!(feet(&doublestep), "lrr");
        let cost = COSTS.doublestep.base + COSTS.doublestep.speed * 0.25;
        assert_eq!(doublestep.note_costs, [0.0, 0.0, cost]);
        assert_eq!(doublestep.cost, cost);
        // A repeated jump: both jacks on its note
        let jumps = solve(&segment("LR . LR"), &COSTS, WIDTH);
        assert_eq!(jumps.note_costs, [0.0, 2.0 * COSTS.jack.speed * 0.5]);
        // A spin is paid where its turn ends
        let spin = solve(&segment("L D R U L D"), &Costs { doublestep: Cost { base: 100.0, speed: 0.0 }, ..COSTS }, WIDTH);
        assert_eq!(spin.count(Technique::FullSpin), 1);
        assert_eq!(spin.note_costs.iter().sum::<f64>(), spin.cost);
        assert_eq!(spin.note_costs.len(), 6);
        assert!(solve(&[], &COSTS, WIDTH).note_costs.is_empty());
    }

    #[test]
    fn keeps_a_holding_foot_down() {
        // The right foot takes both notes under the hold: a doublestep
        assert_eq!(solved("L+ D R L- D", &COSTS), ("lrrl".into(), events(0, 0, 0, 1, 0, 0, 0)));
    }

    #[test]
    fn switches_feet_on_a_hold_when_one_foot_cant_keep_up() {
        // Down can't be bracketed with Up, and not with Right at this cost
        let costs = Costs { bracket: Cost { base: 100.0, speed: 0.0 }, ..COSTS };
        let footing = |times: f64| {
            let solution = solve(&at_speed("D+ R U R U D-", times), &costs, WIDTH);
            (feet(&solution), solution.events)
        };
        // Slow: the right foot takes every note under the left's hold
        assert_eq!(footing(1.0), ("lrrrr".into(), events(0, 0, 0, 3, 0, 0, 0)));
        // Fast: the feet take turns, the one that stepped last taking the hold over
        assert_eq!(footing(4.0), ("lrl*r*l*".into(), switched(3, Events::default())));
        let fast = solve(&at_speed("D+ R U R U D-", 4.0), &costs, WIDTH);
        assert_eq!(fast.cost, 3.0 * (costs.freeze_switch.base + costs.freeze_switch.speed * 4.0));
        assert_eq!(found_on(&fast, 2), [(Technique::FreezeSwitch, 32.0)]);
        // Each switch moves both feet: one onto the note, the other onto the hold
        assert_eq!(fast.moves.iter().map(|m| m.note).collect::<Vec<_>>(), [2, 2, 3, 3, 4, 4]);
        assert!(COSTS.freeze_switch.base > COSTS.freeze_switch.speed);

        // Never, at a price
        let never = Costs { freeze_switch: Cost { base: 1000.0, speed: 0.0 }, ..costs };
        assert_eq!(feet(&solve(&at_speed("D+ R U R U D-", 4.0), &never, WIDTH)), "lrrrr");
    }

    #[test]
    fn a_freeze_switch_leaves_the_other_foot_holding() {
        let costs = Costs { bracket: Cost { base: 100.0, speed: 0.0 }, ..COSTS };
        // The left foot leaves Down to the right for Up, then is the one free to jack it
        let solution = solve(&at_speed("D+ R U U U D-", 4.0), &costs, WIDTH);
        assert_eq!((feet(&solution), solution.events), ("lrl*ll".into(), switched(1, events(2, 0, 0, 0, 0, 0, 0))));
        let step = solution.steps[2];
        assert_eq!((step.foot(UP_ARROW), step.regrabber(DOWN_ARROW)), (Some(Foot::Left), Some(Foot::Right)));
        assert_eq!(solution.steps[3].regrabber(DOWN_ARROW), None);
        // Not once the hold is over
        let over = solve(&at_speed("D+ . D- R U R U", 4.0), &costs, WIDTH);
        assert_eq!((feet(&over), over.events), ("lrlrl".into(), candled(1, Events::default())));
    }

    #[test]
    fn steps_jumps_uncrossed() {
        assert_eq!(solved("LD R U", &COSTS), ("[L|D]rl".into(), Events::default()));
        // Up and Down go the way the player is already turned. The foot on Down jacks it.
        assert_eq!(solved("L D UD", &COSTS), ("lr[U|D]".into(), events(1, 0, 0, 0, 0, 0, 0)));
        assert_eq!(solved("R D UD", &COSTS), ("rl[D|U]".into(), events(1, 0, 0, 0, 0, 0, 0)));
    }

    #[test]
    fn counts_jacks_and_footswitches_in_jumps() {
        // Each foot jacks
        assert_eq!(solved("LR LR U", &COSTS).1, events(2, 0, 0, 0, 0, 0, 0));
        // Only the foot on Up
        assert_eq!(solved("LU UR", &COSTS), ("[L|U][U|R]".into(), events(0, 1, 0, 0, 0, 0, 0)));
        assert_eq!(solved("UR U", &COSTS), ("[U|R]l".into(), events(1, 0, 0, 0, 0, 0, 0)));
        // Starting on the right foot keeps it on Up throughout
        assert_eq!(solved("U UD LU", &COSTS), ("r[D|U][L|U]".into(), events(2, 0, 0, 0, 0, 0, 0)));
        // The feet are set by the jump: Up has to change feet for the last one
        assert_eq!(solved("LR R U UD LU", &COSTS).1, candled(1, events(2, 1, 0, 0, 0, 0, 0)));
    }

    #[test]
    fn tells_sideswitches_from_vertical_footswitches() {
        assert_eq!(solved("L D D R", &COSTS), ("lrlr".into(), events(0, 1, 0, 0, 0, 0, 0)));
        let no_jacks = Costs { jack: Cost { base: 30.0, speed: 0.0 }, ..COSTS };
        assert_eq!(solved("U U", &no_jacks), ("lr".into(), events(0, 1, 0, 0, 0, 0, 0)));
        assert_eq!(solved("D D D", &no_jacks).1, events(0, 2, 0, 0, 0, 0, 0));
        let side = Events { sideswitches: 1, ..Events::default() };
        assert_eq!(solved("L L", &no_jacks), ("lr".into(), side));
        assert_eq!(solved("R R", &no_jacks), ("rl".into(), side));
        // Each has its own cost
        let cheap_sides = Costs { sideswitch: Cost { base: 0.5, speed: 0.0 }, ..COSTS };
        assert_eq!(solved("L L", &cheap_sides), ("lr".into(), side));
        assert_eq!(solved("U U", &cheap_sides), ("ll".into(), events(1, 0, 0, 0, 0, 0, 0)));
        assert_eq!(solve(&segment("L L"), &cheap_sides, WIDTH).cost, 0.5);
    }

    #[test]
    fn counts_vertical_crossovers() {
        // A candle: the right foot from Up to Down, the left on Left
        assert_eq!(solved("U L D", &COSTS), ("rlr".into(), candled(1, Events::default())));
        assert_eq!(solved("D R U", &COSTS), ("lrl".into(), candled(1, Events::default())));
        assert_eq!(solved("U L D L U", &COSTS).1, candled(2, Events::default()));
        // Each foot keeps to its arrows
        assert_eq!(solved("L D L D", &COSTS).1, Events::default());
        assert_eq!(solved("U D U D", &COSTS).1, Events::default());
        // On a doublestep too: the right foot under the left's hold
        let costs = Costs { bracket: Cost { base: 100.0, speed: 0.0 }, ..COSTS };
        let held = solve(&segment("L+ U D L-"), &costs, WIDTH);
        assert_eq!(feet(&held), "lrr");
        assert_eq!(found_on(&held, 2), [(Technique::VerticalCrossover, 8.0), (Technique::Doublestep, 8.0)]);
        // Not once the other foot has taken its arrow
        let no_jacks = Costs { jack: Cost { base: 30.0, speed: 0.0 }, ..COSTS };
        assert_eq!(solved("U U D", &no_jacks), ("lrl".into(), events(0, 1, 0, 0, 0, 0, 0)));
        // Its own cost: paid, or avoided if anything else is cheaper
        let dear = Costs { vertical_crossover: Cost { base: 2.0, speed: 0.5 }, ..COSTS };
        assert_eq!(solve(&segment("U L D"), &dear, WIDTH).cost, 2.5);
        let dearer = Costs { vertical_crossover: Cost { base: 20.0, speed: 0.0 }, ..COSTS };
        assert_eq!(solved("U L D", &dearer).1.vertical_crossovers, 0);
    }

    #[test]
    fn a_foot_turning_on_an_arrow_it_keeps_doesnt_cross() {
        let note = |tokens: &str| segment(tokens)[0];
        let on = |left: Arrows, right: Arrows| State { on: [left, right], ..State::home() };
        let step = |feet: [Arrows; 2]| Step { feet, regrabs: [0, 0] };
        let (left, down, up, right) = (bit(LEFT_ARROW), bit(DOWN_ARROW), bit(UP_ARROW), bit(RIGHT_ARROW));
        let crossed = |state: State, feet: [Arrows; 2]| state.step(&note("D"), step(feet), 0).unwrap().events.vertical_crossovers;

        assert_eq!(crossed(on(up, right), [down, 0]), 1);
        // From or into a bracket
        assert_eq!(crossed(on(left | up, right), [down, 0]), 1);
        assert_eq!(crossed(on(up, right), [left | down, 0]), 1);
        // Left stays under the foot
        assert_eq!(crossed(on(left | up, right), [left | down, 0]), 0);
        let holding = State { held_until: [f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY], ..on(left | up, right) };
        assert_eq!(crossed(holding, [down, 0]), 0);
        // The other foot
        assert_eq!(crossed(on(left, up), [0, down]), 1);
        assert_eq!(crossed(on(left, right), [down, 0]), 0);
    }

    #[test]
    fn brackets_three_or_four_arrows() {
        // One foot takes two arrows next to each other, without crossing the feet
        let three = solve(&segment("R LDU"), &COSTS, WIDTH);
        assert_eq!((feet(&three), three.events), ("r[LD|U]".into(), bracketed(1, Events::default())));
        let four = solve(&segment("R LDUR"), &COSTS, WIDTH);
        assert_eq!(four.events, bracketed(2, events(1, 0, 0, 0, 0, 0, 0)));
        assert!(["r[LD|UR]", "r[LU|DR]"].contains(&feet(&four).as_str()), "{}", feet(&four));
        // Left and Right aren't next to each other, nor Down and Up
        assert!(!fits_a_foot(bit(LEFT_ARROW) | bit(RIGHT_ARROW)));
        assert!(!fits_a_foot(bit(DOWN_ARROW) | bit(UP_ARROW)));
        assert!(fits_a_foot(bit(LEFT_ARROW) | bit(UP_ARROW)));
    }

    #[test]
    fn brackets_when_the_feet_are_holding() {
        // Both feet hold: one keeps its hold and takes Up as well
        let tap = solve(&segment("L+ R+ U L- R-"), &COSTS, WIDTH);
        assert_eq!((feet(&tap), tap.events), ("lrl".into(), bracketed(1, Events::default())));
        // The left foot holds through a jump: it can't leave for one of the jump's arrows
        let jump = solve(&segment("L+ . UR L-"), &COSTS, WIDTH);
        assert_eq!(jump.events, bracketed(1, Events::default()));
        assert!(["l[U|R]", "l[|UR]"].contains(&feet(&jump).as_str()), "{}", feet(&jump));
        let apart = solve(&segment("L+ . DU L-"), &COSTS, WIDTH);
        assert_eq!(apart.events.brackets, 1);
        assert!(["l[U|D]", "l[D|U]"].contains(&feet(&apart).as_str()), "{}", feet(&apart));
    }

    #[test]
    fn brackets_by_choice_at_a_cost() {
        // A foot each, switching feet on Up
        assert_eq!(solved("UR LU", &COSTS), ("[U|R][L|U]".into(), events(0, 1, 0, 0, 0, 0, 0)));
        // With brackets cheap, the left foot stays on Up and takes Left too
        let cheap = Costs { bracket: Cost { base: 0.5, speed: 0.0 }, ..COSTS };
        assert_eq!(solved("UR LU", &cheap), ("[U|R][LU|]".into(), bracketed(1, events(1, 0, 0, 0, 0, 0, 0))));
        // The cost: mostly base
        let bracket = solve(&segment("UR LU"), &cheap, WIDTH).cost - COSTS.jack.base - COSTS.jack.speed;
        assert_eq!(bracket, 0.5);
        assert!(COSTS.bracket.base > COSTS.bracket.speed);
    }

    #[test]
    fn merging_finds_the_same_footing() {
        // Pseudo-random segments, repeats and jumps included
        let tokens = ["L", "D", "U", "R", "L", "D", "U", "R", "LR", "UD", "LD", "UR", "LDU", "LDUR"];
        let mut seed: u64 = 1;
        let mut next = |below: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % below
        };
        let mut patterns: Vec<String> = (0..200)
            .map(|_| (0..9).map(|_| tokens[next(tokens.len())]).collect::<Vec<_>>().join(" "))
            .collect();
        let held = ["L+ D R L- D U D R", "U+ L R L U- D D R L", "L+ R+ D L- R- U D", "D+ R U R U L U D- L", "R D+ U+ L R L D- U-"];
        patterns.extend(held.map(String::from));
        let mut merged = 0;
        for pattern in patterns {
            let notes = segment(&pattern);
            let whole_tree = search(&notes, &COSTS, WIDTH, false);
            let merging = solve(&notes, &COSTS, WIDTH);
            assert!(whole_tree.exact && merging.exact, "{pattern}: branches dropped");
            assert_eq!(whole_tree.merged, 0);
            assert_eq!((merging.cost, &merging.steps), (whole_tree.cost, &whole_tree.steps), "{pattern}");
            assert!(merging.nodes <= whole_tree.nodes, "{pattern}: nodes");
            merged += merging.merged;
        }
        assert!(merged > 0);
    }

    #[test]
    fn merging_keeps_long_segments_small() {
        // 400 notes of crossovers, jacks, footswitches and spins: 2^400 branches unmerged
        let pattern = "L D R D L U U R L D R U L D D R U L R D ".repeat(20);
        let solution = solve(&segment(&pattern), &COSTS, 1000);
        assert!(solution.exact);
        assert!(solution.widest < 100);
    }

    #[test]
    fn drops_the_costliest_branches_past_the_width() {
        let one_branch = solve(&segment("L D R L"), &COSTS, 1);
        assert!(!one_branch.exact);
        assert_eq!(one_branch.steps.len(), 4);
        assert_eq!(one_branch.widest, 1);
        assert!(one_branch.cost > solve(&segment("L D R L"), &COSTS, WIDTH).cost);
    }

    struct SongChart {
        title: String,
        difficulty: String,
        rating: String,
        notes: Vec<TimedNote>,
    }

    /// The dance-single charts of the .sm files under `DDR_SONGS`, a folder of song folders.
    fn song_charts() -> Vec<SongChart> {
        use super::super::{Timing, to_timed_notes};
        use crate::sim_loader::{SimFile, StepChart};

        let songs = std::env::var("DDR_SONGS").expect("DDR_SONGS: folder of song folders");
        let mut paths: Vec<_> = std::fs::read_dir(songs)
            .expect("a readable folder")
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .flat_map(|entry| std::fs::read_dir(entry.path()).expect("a readable folder").flatten())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "sm"))
            .collect();
        paths.sort();

        let mut charts = Vec::new();
        for path in paths {
            let Ok(sim) = SimFile::from_file(&path) else { continue };
            let timing = Timing::new(&sim.bpms, &sim.stops);
            for chart in &sim.step_charts {
                let StepChart::DanceSingle(chart) = chart else { continue };
                charts.push(SongChart {
                    title: sim.title.clone(),
                    difficulty: format!("{:?}", chart.difficulty),
                    rating: chart.rating.to_string(),
                    notes: to_timed_notes(&chart.measures, &timing),
                });
            }
        }
        charts
    }

    fn max_width() -> usize {
        std::env::var("DDR_TREE_WIDTH").map_or(5000, |width| width.parse().expect("a number"))
    }

    /// For the .sm files under `DDR_SONGS`, tab separated: a `SEG` line per segment (notes, whether
    /// it's exact, branches tried, cost, branches merged, most kept) and a `CHART` line per chart
    /// (title, difficulty, rating, notes, inexact segments, then each event and pattern in the
    /// order of `Events`, the positions of `Footing` last, then staircases, drills, drill notes,
    /// gallops and triples).
    #[test]
    #[ignore]
    fn report_songs() {
        use super::super::{SEGMENT_BREAK_SECONDS, segments};
        const PATTERNS: [Technique; 4] = [Technique::Staircase, Technique::Drill, Technique::Gallop, Technique::Triple];

        let max_width = max_width();
        for chart in song_charts() {
            let (mut events, mut found, mut drill_notes, mut inexact) = (Events::default(), [0; 4], 0, 0);
            for segment in segments(&chart.notes, SEGMENT_BREAK_SECONDS) {
                let solution = solve(segment, &super::COSTS, max_width);
                println!(
                    "SEG\t{}\t{}\t{}\t{}\t{}\t{}",
                    segment.len(),
                    solution.exact,
                    solution.nodes,
                    solution.cost,
                    solution.merged,
                    solution.widest
                );
                inexact += usize::from(!solution.exact);
                events += solution.events;
                for (count, technique) in found.iter_mut().zip(PATTERNS) {
                    *count += solution.count(technique);
                }
                drill_notes += solution.of(Technique::Drill).map(|drill| drill.notes).sum::<usize>();
            }
            let counts = [
                chart.notes.len(),
                inexact,
                events.jacks,
                events.vertical_footswitches,
                events.sideswitches,
                events.vertical_crossovers,
                events.doublesteps,
                events.brackets,
                events.freeze_switches,
                events.footing.horizontal_crossovers,
                events.footing.laterals,
                events.footing.half_spins,
                events.footing.full_spins,
                found[0],
                found[1],
                drill_notes,
                found[2],
                found[3],
            ];
            let counts: Vec<String> = counts.iter().map(usize::to_string).collect();
            println!("CHART\t{}\t{}\t{}\t{}", chart.title, chart.difficulty, chart.rating, counts.join("\t"));
        }
    }

    /// Times `solve` over the charts under `DDR_SONGS`, `DDR_RUNS` times (5). Prints one `TIME`
    /// line per chart, tab separated: title, difficulty, rating, notes, segments, branches tried,
    /// then each run's seconds. Then one `RUN` line per run: seconds for every chart.
    #[test]
    #[ignore]
    fn time_songs() {
        use super::super::{SEGMENT_BREAK_SECONDS, segments};
        use std::time::Instant;

        let runs: usize = std::env::var("DDR_RUNS").map_or(5, |runs| runs.parse().expect("a number"));
        let max_width = max_width();
        let charts = song_charts();
        let chart_segments: Vec<Vec<&[TimedNote]>> =
            charts.iter().map(|chart| segments(&chart.notes, SEGMENT_BREAK_SECONDS)).collect();
        // [chart][run]
        let mut seconds = vec![Vec::with_capacity(runs); charts.len()];
        let mut nodes = vec![0; charts.len()];
        for _ in 0..runs {
            for (chart, segments) in chart_segments.iter().enumerate() {
                let start = Instant::now();
                let mut tried = 0;
                for segment in segments {
                    tried += std::hint::black_box(solve(segment, &super::COSTS, max_width)).nodes;
                }
                seconds[chart].push(start.elapsed().as_secs_f64());
                nodes[chart] = tried;
            }
        }
        for (chart, song) in charts.iter().enumerate() {
            let runs: Vec<String> = seconds[chart].iter().map(|seconds| format!("{seconds:.6}")).collect();
            println!(
                "TIME\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                song.title,
                song.difficulty,
                song.rating,
                song.notes.len(),
                chart_segments[chart].len(),
                nodes[chart],
                runs.join("\t")
            );
        }
        for run in 0..runs {
            println!("RUN\t{}\t{:.3}", run + 1, seconds.iter().map(|chart| chart[run]).sum::<f64>());
        }
    }
}
