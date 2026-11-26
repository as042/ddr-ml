//! Scoring the foot search's costs (`chart_analyzer::Costs`) by how close its counts come to a
//! chart spreadsheet's. The sheet's columns are ITGmania's tech counts, so ours are counted its
//! way (see `counts`).

use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use crate::chart_analyzer::{Costs, Events, Solution, Technique, TimedNote, foot_segment};
use crate::sim_loader::ChartDifficulty;

/// The counts compared, in the sheet's column names. The first `SCORED` make the score.
pub const COLUMNS: [&str; 6] = ["Crossovers", "Footswitches", "Sideswitches", "Doublesteps", "Jacks", "Brackets"];
pub const SCORED: usize = 5;

/// One chart's `COLUMNS`, ours or the sheet's.
pub type Counts = [f64; 6];

/// ITGmania (TechCounts.cpp) counts a jack, a footswitch or a doublestep only if its note is
/// less than this many seconds after the one before.
const JACK_CUTOFF_SECONDS: f64 = 0.176;
const FOOTSWITCH_CUTOFF_SECONDS: f64 = 0.3;
const DOUBLESTEP_CUTOFF_SECONDS: f64 = 0.235;

/// Every event of a kind the sheet has a column for. Its Footswitches are on Up and Down only.
fn all_counts(events: &Events) -> Counts {
    [
        events.footing.horizontal_crossovers,
        events.vertical_footswitches,
        events.sideswitches,
        events.doublesteps,
        events.jacks,
        events.brackets,
    ]
    .map(|count| count as f64)
}

/// A segment's footing counted as ITGmania would: jacks, footswitches and doublesteps under its
/// cutoffs only, and jacks and doublesteps only where the note and the one before have one
/// arrow each.
pub fn counts(notes: &[TimedNote], footing: &Solution) -> Counts {
    let single = |note: &TimedNote| (0..4).filter(|&arrow| note.steps(arrow)).count() == 1;
    let mut counts = [0.0; 6];
    for occurrence in &footing.occurrences {
        let i = occurrence.note;
        // The first note's gap is the break before the segment
        let before = i.checked_sub(1).map(|before| &notes[before]);
        let within = |cutoff: f64| before.is_some_and(|before| notes[i].time - before.time < cutoff);
        let singles = before.is_some_and(|before| single(&notes[i]) && single(before));
        let column = match occurrence.technique {
            Technique::HorizontalCrossover => 0,
            Technique::VerticalFootswitch if within(FOOTSWITCH_CUTOFF_SECONDS) => 1,
            Technique::Sideswitch if within(FOOTSWITCH_CUTOFF_SECONDS) => 2,
            Technique::Doublestep if singles && within(DOUBLESTEP_CUTOFF_SECONDS) => 3,
            Technique::Jack if singles && within(JACK_CUTOFF_SECONDS) => 4,
            Technique::Bracket => 5,
            _ => continue,
        };
        counts[column] += 1.0;
    }
    counts
}

/// Rows of comma-separated text. A field in double quotes can hold commas, line breaks and `""`
/// for a quote.
pub(crate) fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let (mut rows, mut row, mut field) = (Vec::new(), Vec::new(), String::new());
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => row.push(std::mem::take(&mut field)),
            '\n' if !quoted => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            '\r' if !quoted => {}
            c => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

/// Letters and digits only, in lower case: titles differ in spacing and punctuation.
pub(crate) fn title_key(title: &str) -> String {
    title.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

/// One folder of a chart spreadsheet.
pub struct Sheet {
    /// By (`title_key`, difficulty)
    charts: HashMap<(String, String), Counts>,
}

impl Sheet {
    /// The rows of `folder` (the sheet's Folder column). A title with `|` in it is two titles
    /// for one song. A count that isn't a number is 0.
    pub fn parse(text: &str, folder: &str) -> Result<Sheet, String> {
        let rows = parse_csv(text);
        let header = rows.first().ok_or("the sheet is empty")?;
        let column = |name: &str| {
            header.iter().position(|title| title == name).ok_or_else(|| format!("the sheet has no {name} column"))
        };
        let (folder_column, title, difficulty) = (column("Folder")?, column("Title")?, column("Difficulty")?);
        let mut counted = [0; 6];
        for (index, name) in COLUMNS.iter().enumerate() {
            counted[index] = column(name)?;
        }

        let mut charts = HashMap::new();
        let field = |row: &[String], column: usize| row.get(column).map_or("", String::as_str).to_string();
        for row in rows.iter().skip(1).filter(|row| field(row, folder_column) == folder) {
            // The sheet's names for the .sm format's difficulties
            let difficulty = match field(row, difficulty).as_str() {
                "Basic" => ChartDifficulty::Easy,
                "Difficult" => ChartDifficulty::Medium,
                "Expert" => ChartDifficulty::Hard,
                "Challenge" => ChartDifficulty::Challenge,
                "Beginner" => ChartDifficulty::Beginner,
                _ => continue,
            };
            let counts = counted.map(|column| field(row, column).parse().unwrap_or(0.0));
            for title in field(row, title).split('|') {
                charts.insert((title_key(title), format!("{difficulty:?}")), counts);
            }
        }
        if charts.is_empty() {
            return Err(format!("the sheet has no charts in the folder {folder}"));
        }
        Ok(Sheet { charts })
    }

    pub fn counts(&self, title: &str, difficulty: ChartDifficulty) -> Option<Counts> {
        self.charts.get(&(title_key(title), format!("{difficulty:?}"))).copied()
    }
}

/// A chart to foot, and what the sheet counts in it.
pub struct Chart {
    /// Which of the songs it is from, to keep a song's charts together in a split
    pub song: usize,
    /// See `chart_analyzer::footing_segments`
    pub segments: Vec<Vec<TimedNote>>,
    pub sheet: Counts,
}

pub struct Footed {
    /// Per chart, as `counts`
    pub counts: Vec<Counts>,
    /// Over all charts, with no cutoffs
    pub all: Counts,
    /// Segments whose footing may not be the cheapest
    pub inexact: usize,
}

/// Every chart footed with `costs`, on `threads` threads.
pub fn foot_all(charts: &[Chart], costs: &Costs, threads: usize) -> Footed {
    let next = AtomicUsize::new(0);
    let foot = || {
        let mut footed = Vec::new();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            let Some(chart) = charts.get(index) else { break };
            let (mut events, mut counted, mut inexact) = (Events::default(), [0.0; 6], 0);
            for segment in &chart.segments {
                let solution = foot_segment(segment, costs);
                for (total, count) in counted.iter_mut().zip(counts(segment, &solution)) {
                    *total += count;
                }
                events += solution.events;
                inexact += usize::from(!solution.exact);
            }
            footed.push((index, counted, all_counts(&events), inexact));
        }
        footed
    };
    let footed: Vec<(usize, Counts, Counts, usize)> = thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.max(1)).map(|_| scope.spawn(foot)).collect();
        workers.into_iter().flat_map(|worker| worker.join().expect("a footing thread panicked")).collect()
    });

    let mut all_footed = Footed { counts: vec![[0.0; 6]; charts.len()], all: [0.0; 6], inexact: 0 };
    for (index, counts, all, inexact) in footed {
        all_footed.counts[index] = counts;
        for (total, count) in all_footed.all.iter_mut().zip(all) {
            *total += count;
        }
        all_footed.inexact += inexact;
    }
    all_footed
}

pub struct Score {
    /// Per column: the charts' absolute differences from the sheet added up, over our total plus
    /// the sheet's. 0 is a match, 1 is nothing in common, and counting half as many everywhere
    /// is as far off as counting twice as many (a third).
    pub errors: Counts,
    /// Our totals
    pub totals: Counts,
    /// Mean of the first `SCORED` errors
    pub score: f64,
}

/// `ours`: per chart, as `Footed::counts`.
pub fn score(charts: &[Chart], ours: &[Counts]) -> Score {
    let (mut differences, mut sheet_totals, mut totals) = ([0.0; 6], [0.0; 6], [0.0; 6]);
    for (chart, ours) in charts.iter().zip(ours) {
        for column in 0..6 {
            differences[column] += (ours[column] - chart.sheet[column]).abs();
            sheet_totals[column] += chart.sheet[column];
            totals[column] += ours[column];
        }
    }
    let mut errors = [0.0; 6];
    for column in 0..6 {
        let both = totals[column] + sheet_totals[column];
        errors[column] = if both > 0.0 { differences[column] / both } else { 0.0 };
    }
    Score { errors, totals, score: errors[..SCORED].iter().sum::<f64>() / SCORED as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chart_analyzer::{TEST_COSTS as COSTS, foot_arrows};

    const SHEET: &str = "Folder,Title,Difficulty,Crossovers,Footswitches,Sideswitches,Jacks,Brackets,Doublesteps,Other\r\n\
        A3,\"Acid, Tribal & Dance\",Expert,4,3,2,1,0,5,x\n\
        A3,量子の海|Quantum Sea,Basic,1,0,0,7,,2,x\n\
        A3,Odd,Edit,9,9,9,9,9,9,x\n\
        A20,Elsewhere,Expert,9,9,9,9,9,9,x\n";

    #[test]
    fn parses_quoted_csv() {
        let rows = parse_csv("a,\"b, \"\"c\"\"\nd\",e\r\n,\nlast");
        assert_eq!(rows, [vec!["a", "b, \"c\"\nd", "e"], vec!["", ""], vec!["last"]]);
        assert!(parse_csv("").is_empty());
    }

    #[test]
    fn reads_one_folder_of_a_sheet() {
        let sheet = Sheet::parse(SHEET, "A3").unwrap();
        // Columns by name, into `COLUMNS` order
        assert_eq!(sheet.counts("ACID TRIBAL & DANCE", ChartDifficulty::Hard), Some([4.0, 3.0, 2.0, 5.0, 1.0, 0.0]));
        assert_eq!(sheet.counts("Acid, Tribal & Dance", ChartDifficulty::Medium), None);
        // Either title, and an empty count
        let both = Some([1.0, 0.0, 0.0, 2.0, 7.0, 0.0]);
        assert_eq!(sheet.counts("量子の海", ChartDifficulty::Easy), both);
        assert_eq!(sheet.counts("quantum-sea", ChartDifficulty::Easy), both);
        assert_eq!(sheet.counts("Elsewhere", ChartDifficulty::Hard), None);
        assert_eq!(sheet.charts.len(), 3);

        assert!(Sheet::parse(SHEET, "A4").is_err());
        assert!(Sheet::parse("Folder,Title\nA3,x\n", "A3").is_err());
    }

    fn chart(arrows: &str, sheet: Counts) -> Chart {
        Chart { song: 0, segments: vec![foot_arrows(arrows, &COSTS, 1.0).unwrap().notes], sheet }
    }

    #[test]
    fn scores_counts_against_the_sheet() {
        // A crossover, a vertical footswitch, nothing
        let charts = [
            chart("LDRDL", [1.0, 0.0, 0.0, 0.0, 9.0, 0.0]),
            chart("LDDR", [3.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            chart("LDUR", [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ];
        let footed = foot_all(&charts, &COSTS, 3);
        assert_eq!(footed.counts, [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0; 6]]);
        assert_eq!((footed.all, footed.inexact), ([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0));
        assert_eq!(foot_all(&charts, &COSTS, 1).counts, footed.counts);

        let score = score(&charts, &footed.counts);
        // Crossovers: 3 off, of our 1 and the sheet's 4. Footswitches: 1 off, of 1 and 2. No
        // sideswitches or doublesteps at all. Jacks: nothing in common.
        assert_eq!(score.errors, [3.0 / 5.0, 1.0 / 3.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(score.totals, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(score.score, (3.0 / 5.0 + 1.0 / 3.0 + 1.0) / 5.0);

        // Twice as many is as far off as half as many
        let (twice, half) = (chart("LR", [2.0; 6]), chart("LR", [0.5; 6]));
        let same_way_off = |chart: Chart| super::score(&[chart], &[[1.0; 6]]).score;
        assert_eq!(same_way_off(twice), 1.0 / 3.0);
        assert_eq!(same_way_off(half), 1.0 / 3.0);
    }

    #[test]
    fn counts_as_the_sheet_does() {
        let counted = |arrows: &str, speed: f64| {
            let footed = foot_arrows(arrows, &COSTS, speed).unwrap();
            (all_counts(&footed.footing.events), counts(&footed.notes, &footed.footing))
        };
        // A footswitch: counted under 0.3 s
        let footswitch = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(counted("LDDR", 1.0), (footswitch, footswitch));
        assert_eq!(counted("LDDR", 0.5), (footswitch, footswitch));
        assert_eq!(counted("LDDR", 0.4), (footswitch, [0.0; 6]));
        // Jacks: under 0.176 s
        let jacks = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        assert_eq!(counted("UUU", 1.0), (jacks, jacks));
        assert_eq!(counted("UUU", 0.5), (jacks, [0.0; 6]));
        // And not in or after a jump
        assert_eq!(counted("[LR][LR]", 1.0), (jacks, [0.0; 6]));
        // A doublestep: under 0.235 s
        let doublestep = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert_eq!(counted("RLDRL", 1.0), (doublestep, doublestep));
        assert_eq!(counted("RL  UR", 1.0), (doublestep, [0.0; 6]));
        // A crossover: at any speed
        let crossover = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(counted("LDRDL", 0.25), (crossover, crossover));
    }
}
