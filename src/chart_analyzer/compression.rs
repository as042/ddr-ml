//! How well a chart's notes compress: a stand-in for their entropy that sees order, where
//! `ChartStatistics::rhythm_entropy` only sees how common each gap is.

use std::collections::HashMap;

use super::TimedNote;
use super::statistics::ChartStatistics;
use crate::sim_loader::Note;

/// A beat is this many of a gap's units: StepMania's rows, 192 to a measure.
const GAP_UNITS_PER_BEAT: f64 = 48.0;

/// `notes`: the chart's, in order.
pub(super) fn count(notes: &[TimedNote], stats: &mut ChartStatistics) {
    let gaps: Vec<u8> = (0..notes.len()).map(|i| gap_byte(notes, i)).collect();
    let arrows: Vec<u8> = notes.iter().map(|note| arrows_byte(&note.row)).collect();
    let steps: Vec<u8> = gaps.iter().zip(&arrows).flat_map(|(&gap, &arrows)| [gap, arrows]).collect();
    stats.step_compression = fold(&steps);
    stats.arrow_compression = fold(&arrows);
    stats.rhythm_compression = fold(&gaps);
}

/// Beats from the note before note `i` to it, in `GAP_UNITS_PER_BEAT`, rounded: 255 at most, 0
/// for the first note.
fn gap_byte(notes: &[TimedNote], i: usize) -> u8 {
    match i.checked_sub(1) {
        Some(before) => ((notes[i].beat - notes[before].beat) * GAP_UNITS_PER_BEAT).round().clamp(0.0, 255.0) as u8,
        None => 0,
    }
}

/// What a row has to step on, 2 bits a panel from Left: 1 a tap, 2 a hold head, 3 a roll head.
fn arrows_byte(row: &[Note; 4]) -> u8 {
    let step = |note: Note| match note {
        Note::Tap => 1,
        Note::HoldStart => 2,
        Note::RollStart => 3,
        Note::Empty | Note::End | Note::Mine => 0,
    };
    row.iter().enumerate().fold(0, |byte, (panel, &note)| byte | step(note) << (2 * panel))
}

/// How many times smaller `lzw_bits` makes `bytes`. Under 1 if they come out larger. 0 for none.
fn fold(bytes: &[u8]) -> f64 {
    if bytes.is_empty() { 0.0 } else { (8 * bytes.len()) as f64 / lzw_bits(bytes) as f64 }
}

/// Bits LZW writes `bytes` in. Its dictionary starts as the 256 single bytes. Each code is the
/// longest phrase in it that comes next, in as many bits as it takes to number the phrases so
/// far, and adds that phrase with the byte after it. No clear or stop codes.
fn lzw_bits(bytes: &[u8]) -> usize {
    let Some((&first, rest)) = bytes.split_first() else { return 0 };
    // (phrase, next byte) to the phrase they make
    let mut longer: HashMap<(usize, u8), usize> = HashMap::new();
    let mut phrases = 256;
    let code_bits = |phrases: usize| (phrases - 1).ilog2() as usize + 1;
    let (mut bits, mut phrase) = (0, first as usize);
    for &byte in rest {
        match longer.get(&(phrase, byte)) {
            Some(&known) => phrase = known,
            None => {
                bits += code_bits(phrases);
                longer.insert((phrase, byte), phrases);
                phrases += 1;
                phrase = byte as usize;
            }
        }
    }
    bits + code_bits(phrases)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{assert_close, stats};
    use super::*;
    use crate::sim_loader::ChartDifficulty::*;

    #[test]
    fn counts_the_bits_lzw_writes() {
        assert_eq!(lzw_bits(&[]), 0);
        // One code, of the 256 there are
        assert_eq!(lzw_bits(&[7]), 8);
        // No phrase comes twice: a code a byte, 9 bits each once there are 257
        assert_eq!(lzw_bits(&[1, 2, 3, 4]), 8 + 3 * 9);
        // 1 4 1 4 1 4 1 4 as 1, 4, 1 4, 1 4 1, 4
        assert_eq!(lzw_bits(&[1, 4, 1, 4, 1, 4, 1, 4]), 8 + 4 * 9);
        // A run of 10 as 5, 5 5, 5 5 5, 5 5 5 5
        assert_eq!(lzw_bits(&[5; 10]), 8 + 3 * 9);
        // A run of 1 + 2 + ... + 257 is 257 codes, the last with 512 phrases to number: 9 bits.
        // A byte more is one more code, with 513: 10 bits
        let run = vec![5; 257 * 258 / 2 + 1];
        assert_eq!(lzw_bits(&run[1..]), 8 + 256 * 9);
        assert_eq!(lzw_bits(&run), 8 + 256 * 9 + 10);
        // Repeats make it smaller, and nothing to repeat makes it larger
        assert!(fold(&[5; 100]) > 5.0);
        assert_close(fold(&[1, 2, 3, 4]), 32.0 / 35.0, "four different bytes");
        assert_eq!(fold(&[]), 0.0);
    }

    #[test]
    fn writes_notes_as_bytes() {
        use Note::*;
        assert_eq!(arrows_byte(&[Tap, Empty, Empty, Empty]), 0b0000_0001);
        assert_eq!(arrows_byte(&[Empty, Tap, Empty, Tap]), 0b0100_0100);
        assert_eq!(arrows_byte(&[HoldStart, Empty, RollStart, Empty]), 0b0011_0010);
        // Not steps
        assert_eq!(arrows_byte(&[Mine, End, Empty, Tap]), 0b0100_0000);

        let tap = |beat: f64| TimedNote { time: beat, beat, heard_beat: beat, row: [Tap, Empty, Empty, Empty], hold_ends: [None; 4] };
        // A quarter note, an 8th, a 16th, a 12th, then gaps too long or too short to tell apart
        let notes = [tap(0.0), tap(1.0), tap(1.5), tap(1.75), tap(1.75 + 1.0 / 3.0), tap(8.0), tap(20.0), tap(20.001)];
        let gaps: Vec<u8> = (0..notes.len()).map(|i| gap_byte(&notes, i)).collect();
        assert_eq!(gaps, [0, 48, 24, 12, 16, 255, 255, 0]);
    }

    #[test]
    fn compresses_a_charts_notes() {
        // (difficulty, steps, arrows, rhythm), each (bytes, the bits LZW writes them in)
        let expected = [(Beginner, (0, 0), (0, 0), (0, 0)), (Medium, (24, 179), (12, 71), (12, 62))];
        for (difficulty, steps, arrows, rhythm) in expected {
            let s = stats("rhythm_stats.sm", difficulty);
            let fold = |(bytes, bits): (usize, usize)| if bits > 0 { (8 * bytes) as f64 / bits as f64 } else { 0.0 };
            assert_close(s.step_compression, fold(steps), &format!("{difficulty:?} steps"));
            assert_close(s.arrow_compression, fold(arrows), &format!("{difficulty:?} arrows"));
            assert_close(s.rhythm_compression, fold(rhythm), &format!("{difficulty:?} rhythm"));
        }
    }
}
