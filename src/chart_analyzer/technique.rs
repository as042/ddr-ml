//! What is looked for in a chart, and the record of each one found: where it is, how long, how fast.

use super::{TIME_EPSILON, TimedNote};

/// In the order of the statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Technique {
    // From the notes alone (see `sequences`)
    /// An arrow the note before had too, whichever foot steps it
    RepeatedArrow,
    /// 2+ arrows at once
    Jump,
    /// `Settings::min_stream_notes` or more evenly spaced notes
    Stream,
    /// A note under a hold or roll that started on an earlier note
    HeldStep,
    /// A note off the beat with no note on the next stronger position (see `sequences`)
    Syncopation,
    /// A note with uneven gaps before and after it
    RhythmChange,

    // Events of the foot simulation (see `footing::Events`)
    Jack,
    VerticalFootswitch,
    /// A vertical footswitch on Up
    UpFootswitch,
    /// A vertical footswitch on Down
    DownFootswitch,
    Sideswitch,
    HorizontalCrossover,
    /// A horizontal crossover by a foot that came from its own side's arrow
    FullCrossover,
    /// Any other horizontal crossover: by a foot that came from Up or Down
    HalfCrossover,
    VerticalCrossover,
    Doublestep,
    Bracket,
    FreezeSwitch,
    Lateral,
    HalfSpin,
    FullSpin,

    // Patterns on the simulation's footing (see `footing::patterns`)
    Staircase,
    Drill,
    Gallop,
    Triple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Notes,
    FootingEvent,
    FootingPattern,
}

impl Technique {
    pub const ALL: [Technique; 25] = [
        Technique::RepeatedArrow,
        Technique::Jump,
        Technique::Stream,
        Technique::HeldStep,
        Technique::Syncopation,
        Technique::RhythmChange,
        Technique::Jack,
        Technique::VerticalFootswitch,
        Technique::UpFootswitch,
        Technique::DownFootswitch,
        Technique::Sideswitch,
        Technique::HorizontalCrossover,
        Technique::FullCrossover,
        Technique::HalfCrossover,
        Technique::VerticalCrossover,
        Technique::Doublestep,
        Technique::Bracket,
        Technique::FreezeSwitch,
        Technique::Lateral,
        Technique::HalfSpin,
        Technique::FullSpin,
        Technique::Staircase,
        Technique::Drill,
        Technique::Gallop,
        Technique::Triple,
    ];

    /// Position in `ALL`.
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Technique::RepeatedArrow => "repeated arrow",
            Technique::Jump => "jump",
            Technique::Stream => "stream",
            Technique::HeldStep => "held step",
            Technique::Syncopation => "syncopation",
            Technique::RhythmChange => "rhythm change",
            Technique::Jack => "jack",
            Technique::VerticalFootswitch => "vertical footswitch",
            Technique::UpFootswitch => "up footswitch",
            Technique::DownFootswitch => "down footswitch",
            Technique::Sideswitch => "sideswitch",
            Technique::HorizontalCrossover => "horizontal crossover",
            Technique::FullCrossover => "full crossover",
            Technique::HalfCrossover => "half crossover",
            Technique::VerticalCrossover => "vertical crossover",
            Technique::Doublestep => "doublestep",
            Technique::Bracket => "bracket",
            Technique::FreezeSwitch => "freeze switch",
            Technique::Lateral => "lateral",
            Technique::HalfSpin => "half spin",
            Technique::FullSpin => "full spin",
            Technique::Staircase => "staircase",
            Technique::Drill => "drill",
            Technique::Gallop => "gallop",
            Technique::Triple => "triple",
        }
    }

    pub fn plural(self) -> String {
        let name = self.name();
        if name.ends_with("ch") { format!("{name}es") } else { format!("{name}s") }
    }

    /// "1 jack", "2 jacks".
    pub fn counted(self, count: usize) -> String {
        if count == 1 { format!("1 {}", self.name()) } else { format!("{count} {}", self.plural()) }
    }

    /// The name in snake_case, for statistic names.
    pub fn key(self) -> String {
        self.name().replace(' ', "_")
    }

    pub fn source(self) -> Source {
        match self {
            Technique::RepeatedArrow
            | Technique::Jump
            | Technique::Stream
            | Technique::HeldStep
            | Technique::Syncopation
            | Technique::RhythmChange => Source::Notes,
            Technique::Staircase | Technique::Drill | Technique::Gallop | Technique::Triple => Source::FootingPattern,
            _ => Source::FootingEvent,
        }
    }

    /// The techniques that each one of this is also one of.
    pub fn kinds(self) -> &'static [Technique] {
        match self {
            Technique::VerticalFootswitch => &[Technique::UpFootswitch, Technique::DownFootswitch],
            Technique::HorizontalCrossover => &[Technique::FullCrossover, Technique::HalfCrossover],
            _ => &[],
        }
    }

    /// Whether one can span more notes than another.
    pub fn varies_in_length(self) -> bool {
        matches!(self, Technique::Stream | Technique::Drill)
    }

    /// A position of the feet, as hard to leave fast as to enter fast.
    pub fn is_position(self) -> bool {
        matches!(
            self,
            Technique::HorizontalCrossover
                | Technique::FullCrossover
                | Technique::HalfCrossover
                | Technique::Lateral
                | Technique::HalfSpin
                | Technique::FullSpin
        )
    }

    /// Timed by the shorter of the gaps before and after its note: a position, or a rhythm
    /// change, which is both its gaps. The rest are timed by the gap before.
    pub fn timed_by_either_gap(self) -> bool {
        self.is_position() || self == Technique::RhythmChange
    }
}

/// One technique found in a segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Occurrence {
    pub technique: Technique,
    /// Its first note, counted within the segment
    pub note: usize,
    /// Notes it spans. 1 for what a single step does.
    pub notes: usize,
    /// Notes per second it is done at (see `Occurrence::at` and `Occurrence::over`)
    pub speed: f64,
}

impl Occurrence {
    /// What the step of note `i` does, at the speed of the gap before it, or of the shorter of
    /// the gaps before and after (see `Technique::timed_by_either_gap`). 0 with no such gap in
    /// the segment.
    pub fn at(technique: Technique, segment: &[TimedNote], i: usize) -> Occurrence {
        let (before, after) = gaps(segment, i);
        let gap = if technique.timed_by_either_gap() { before.min(after) } else { before };
        Occurrence { technique, note: i, notes: 1, speed: speed(gap) }
    }

    /// `notes` notes from note `first`, at their mean speed.
    pub fn over(technique: Technique, segment: &[TimedNote], first: usize, notes: usize) -> Occurrence {
        let seconds = segment[first + notes - 1].time - segment[first].time;
        let speed = if seconds > TIME_EPSILON { (notes - 1) as f64 / seconds } else { 0.0 };
        Occurrence { technique, note: first, notes, speed }
    }
}

/// Seconds from the note before note `i` to it, and from it to the note after. Infinite where
/// the segment has no such note.
pub fn gaps(segment: &[TimedNote], i: usize) -> (f64, f64) {
    let gap = |later: usize| match (later.checked_sub(1).and_then(|before| segment.get(before)), segment.get(later)) {
        (Some(before), Some(later)) => later.time - before.time,
        _ => f64::INFINITY,
    };
    (gap(i), gap(i + 1))
}

/// Notes per second of notes `gap` seconds apart. 0 for an infinite gap.
pub fn speed(gap: f64) -> f64 {
    if gap > TIME_EPSILON { 1.0 / gap } else { 0.0 }
}

/// Gaps are even if the longest is `tolerance` longer than the shortest, as a fraction, at most.
pub fn even(shortest: f64, longest: f64, tolerance: f64) -> bool {
    longest <= shortest * (1.0 + tolerance) + TIME_EPSILON
}

/// Runs of evenly spaced notes (see `even`), each as long as it goes, as (first note, notes). A
/// run may start on the last note of the one before.
///
/// `links(i)`: note `i` can follow note `i - 1` in a run. `extends(first, i)`: note `i` can join
/// the run from `first` that reaches note `i - 1`.
pub fn even_runs(
    segment: &[TimedNote],
    min_notes: usize,
    tolerance: f64,
    links: impl Fn(usize) -> bool,
    extends: impl Fn(usize, usize) -> bool,
) -> Vec<(usize, usize)> {
    let notes = segment.len();
    let gap = |i: usize| segment[i].time - segment[i - 1].time;
    let mut runs = Vec::new();
    let mut first = 0;
    let (mut shortest, mut longest) = (f64::INFINITY, 0.0_f64);
    for i in 1..=notes {
        let linked = i < notes && links(i);
        if linked {
            let (shorter, longer) = (shortest.min(gap(i)), longest.max(gap(i)));
            if extends(first, i) && even(shorter, longer, tolerance) {
                (shortest, longest) = (shorter, longer);
                continue;
            }
        }
        if i - first >= min_notes {
            runs.push((first, i - first));
        }
        (first, shortest, longest) = if linked { (i - 1, gap(i), gap(i)) } else { (i, f64::INFINITY, 0.0) };
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::super::SETTINGS;
    use super::super::footing::tests::segment;
    use super::*;

    const TOLERANCE: f64 = SETTINGS.even_tolerance;

    #[test]
    fn lists_every_technique_once_in_order() {
        for (i, technique) in Technique::ALL.into_iter().enumerate() {
            assert_eq!(technique.index(), i, "{technique:?}");
        }
        let mut keys: Vec<String> = Technique::ALL.iter().map(|technique| technique.key()).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), Technique::ALL.len());
    }

    #[test]
    fn names_techniques() {
        assert_eq!(Technique::VerticalFootswitch.key(), "vertical_footswitch");
        assert_eq!(Technique::VerticalFootswitch.counted(1), "1 vertical footswitch");
        assert_eq!(Technique::VerticalFootswitch.counted(2), "2 vertical footswitches");
        assert_eq!(Technique::Jack.counted(3), "3 jacks");
        assert_eq!(Technique::Jack.source(), Source::FootingEvent);
        assert_eq!(Technique::Stream.source(), Source::Notes);
        assert_eq!(Technique::Drill.source(), Source::FootingPattern);
        assert_eq!(Technique::HorizontalCrossover.kinds(), [Technique::FullCrossover, Technique::HalfCrossover]);
        assert_eq!(Technique::VerticalFootswitch.kinds(), [Technique::UpFootswitch, Technique::DownFootswitch]);
        assert_eq!(Technique::DownFootswitch.counted(2), "2 down footswitches");
        assert_eq!(Technique::HalfCrossover.source(), Source::FootingEvent);
        assert!(Technique::FullCrossover.is_position() && Technique::Jack.kinds().is_empty());
        assert_eq!(Technique::RhythmChange.key(), "rhythm_change");
        assert_eq!(Technique::Syncopation.counted(2), "2 syncopations");
        assert_eq!((Technique::Syncopation.source(), Technique::RhythmChange.source()), (Source::Notes, Source::Notes));
        assert!(Technique::RhythmChange.timed_by_either_gap() && !Technique::RhythmChange.is_position());
        assert!(Technique::Lateral.timed_by_either_gap() && !Technique::Syncopation.timed_by_either_gap());
    }

    #[test]
    fn times_a_step_by_the_gap_before_and_a_position_by_the_shorter_gap() {
        // 0.125 s, then 0.25 s
        let notes = segment("L D . U");
        assert_eq!(gaps(&notes, 0), (f64::INFINITY, 0.125));
        assert_eq!(gaps(&notes, 1), (0.125, 0.25));
        assert_eq!(gaps(&notes, 2), (0.25, f64::INFINITY));

        let at = |technique, i| Occurrence::at(technique, &notes, i);
        assert_eq!(at(Technique::Doublestep, 2), Occurrence { technique: Technique::Doublestep, note: 2, notes: 1, speed: 4.0 });
        assert_eq!(at(Technique::HorizontalCrossover, 2).speed, 4.0);
        assert_eq!(at(Technique::Doublestep, 1).speed, 8.0);
        assert_eq!(at(Technique::HorizontalCrossover, 1).speed, 8.0);
        assert_eq!((at(Technique::RhythmChange, 2).speed, at(Technique::Syncopation, 2).speed), (4.0, 4.0));
        assert_eq!((at(Technique::RhythmChange, 1).speed, at(Technique::RhythmChange, 0).speed), (8.0, 8.0));
        // No note before
        assert_eq!(at(Technique::Bracket, 0).speed, 0.0);
        assert_eq!(at(Technique::HorizontalCrossover, 0).speed, 8.0);
        assert_eq!(Occurrence::at(Technique::Bracket, &notes[..1], 0).speed, 0.0);
        assert_eq!(Occurrence::at(Technique::Lateral, &notes[..1], 0).speed, 0.0);
    }

    #[test]
    fn times_a_run_by_its_mean_speed() {
        // 3 notes over 0.375 s
        let notes = segment("L D . U");
        let run = Occurrence::over(Technique::Triple, &notes, 0, 3);
        assert_eq!((run.note, run.notes), (0, 3));
        assert!((run.speed - 2.0 / 0.375).abs() < 1e-9);
        assert_eq!(Occurrence::over(Technique::Stream, &notes, 1, 1).speed, 0.0);
    }

    #[test]
    fn finds_even_runs() {
        let any = |_: usize| true;
        let all = |_: usize, _: usize| true;
        // 5 notes 0.125 s apart, then 3 more 0.25 s apart, sharing a note
        let notes = segment("L D U R L . D . U . R");
        assert_eq!(even_runs(&notes, 2, TOLERANCE, any, all), [(0, 5), (4, 4)]);
        // Twice the gap is even enough at a tolerance of 1
        assert_eq!(even_runs(&notes, 2, 1.0, any, all), [(0, 8)]);
        assert_eq!(even_runs(&notes, 5, TOLERANCE, any, all), [(0, 5)]);
        assert_eq!(even_runs(&notes, 6, TOLERANCE, any, all), []);
        // A note that can't follow the one before starts a new run
        assert_eq!(even_runs(&notes, 2, TOLERANCE, |i| i != 2, all), [(0, 2), (2, 3), (4, 4)]);
        // One that can't join the run so far starts one from the note before
        assert_eq!(even_runs(&notes, 2, TOLERANCE, any, |first, i| i - first < 3), [(0, 3), (2, 3), (4, 3), (6, 2)]);
        assert_eq!(even_runs(&notes[..1], 1, TOLERANCE, any, all), [(0, 1)]);
        assert_eq!(even_runs(&[], 1, TOLERANCE, any, all), []);
    }
}
