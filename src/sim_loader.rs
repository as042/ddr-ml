use std::{fs, io::{self, ErrorKind}, path::{Path, PathBuf}};

use serde::{Deserialize, Serialize};

#[derive(Default)]
pub struct SimFile {
    // Header. Missing tags load as "" or 0, like empty ones (`#BANNER:;`), as in StepMania.
    pub title: String,
    pub artist: String,
    artist_translit: String,
    banner: String,
    background: String,
    cd_title: String,
    music: String,
    /// Negated: `#OFFSET:-0.125` puts beat 0 at 0.125 s into the music.
    offset: f64,
    sample_start: f64,
    sample_length: f64,
    pub bpms: Vec<BPMChange>,
    /// In file order.
    pub stops: Vec<Stop>,
    bg_stage: String,

    /// Note information for each chart
    pub step_charts: Vec<StepChart>,
}

pub struct BPMChange {
    /// Can be fractional (`64.5=300`).
    pub beat: f64,
    pub bpm: f64,
}

pub struct Stop {
    /// Can be fractional.
    pub beat: f64,
    pub seconds: f64,
}

pub enum StepChart {
    DanceSingle(SingleChart),
    DanceDouble(DoubleChart),
}

pub struct SingleChart {
    pub difficulty: ChartDifficulty,
    pub rating: f64,
    pub measures: Vec<Measure>,
}

/// Not yet implemented.
#[allow(dead_code)]
pub struct DoubleChart {

}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartDifficulty {
    Beginner,
    Easy,
    Medium,
    Hard,
    Challenge,
}

pub struct Measure {
    /// Always 4 in .sm files (no time signatures). Rows split the beats evenly: 16 rows are
    /// 16th notes.
    pub num_beats: usize,
    pub rows: Vec<[Note; 4]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Note {
    Empty,
    Tap,
    HoldStart,
    /// Ends a hold or a roll (.sm files use `3` for both).
    End,
    RollStart,
    /// DDR's shock arrows
    Mine,
}

/// `path` itself, or for a song folder, the `.sm` file in it named after the folder.
pub(crate) fn simfile_path(path: impl AsRef<Path>) -> io::Result<PathBuf> {
    let mut file_path = None;
    if path.as_ref().is_dir() {
        let abs_dir = path.as_ref().canonicalize()?;
        let dir_name = abs_dir.file_name().ok_or(io::Error::other("Could not parse file name"))?;

        for entry in fs::read_dir(path)?.flatten() {
            let entry_path = entry.path();

            // The folder's audio and images are named after it too
            let is_sm = entry_path.extension().is_some_and(|extension| extension.eq_ignore_ascii_case("sm"));
            if entry_path.is_file() && is_sm && entry_path.file_stem() == Some(dir_name) {
                file_path = Some(entry_path);
            }
         }
    }
    else if path.as_ref().is_file() {
        file_path = Some(path.as_ref().to_path_buf());
    }
    file_path.ok_or(io::Error::new(ErrorKind::NotFound, "Failed to determine path"))
}

impl SimFile {
    /// Load the entire contents of a simfile (.sm) into memory. Can select a file or the containing directory if they share the same name.
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        SimFile::parse(&fs::read_to_string(simfile_path(path)?)?)
    }

    pub fn parse(content: &str) -> io::Result<Self> {
        let mut sim = SimFile::default();

        // A tag (`#NAME:value;`) can span lines, so they collect in `pending` until a `;`.
        let mut pending = String::new();
        for line in content.lines() {
            // Strip `//` comments
            let line = line.split("//").next().unwrap_or_default();
            // No value has a line starting with `#`: the tag before has lost its `;`, and would
            // take this one for part of its value
            if line.trim_start().starts_with('#')
                && let Some(open) = pending.find('#')
            {
                let name = pending[open + 1..].split(':').next().unwrap_or_default().trim();
                return Err(invalid(format!("#{name}: missing ';' (the next tag starts before it ends)")));
            }
            pending.push_str(line);
            pending.push('\n');

            while let Some(end) = pending.find(';') {
                let tag: String = pending.drain(..=end).collect();
                // Skip text outside a tag
                let Some(start) = tag.find('#') else { continue };
                let tag = &tag[start + 1..end];
                let (name, value) = tag.split_once(':').unwrap_or((tag, ""));
                let value = value.trim();

                match name.trim().to_ascii_uppercase().as_str() {
                    "TITLE" => sim.title = value.to_string(),
                    "ARTIST" => sim.artist = value.to_string(),
                    "ARTISTTRANSLIT" => sim.artist_translit = value.to_string(),
                    "BANNER" => sim.banner = value.to_string(),
                    "BACKGROUND" => sim.background = value.to_string(),
                    "CDTITLE" => sim.cd_title = value.to_string(),
                    "MUSIC" => sim.music = value.to_string(),
                    "OFFSET" => sim.offset = parse_number_or_zero("#OFFSET", value)?,
                    "SAMPLESTART" => sim.sample_start = parse_number_or_zero("#SAMPLESTART", value)?,
                    "SAMPLELENGTH" => sim.sample_length = parse_number_or_zero("#SAMPLELENGTH", value)?,
                    "BPMS" => {
                        let bpms = parse_beat_pairs("#BPMS", value)?;
                        if let Some((beat, bpm)) = bpms.iter().find(|&&(_, bpm)| bpm <= 0.0) {
                            return Err(invalid(format!("#BPMS: a BPM must be above 0, got {bpm} on beat {beat}{WARP}")));
                        }
                        sim.bpms = bpms.into_iter().map(|(beat, bpm)| BPMChange { beat, bpm }).collect()
                    }
                    "STOPS" => {
                        let stops = parse_beat_pairs("#STOPS", value)?;
                        if let Some((beat, seconds)) = stops.iter().find(|&&(_, seconds)| seconds < 0.0) {
                            return Err(invalid(format!("#STOPS: a stop can't be negative, got {seconds} on beat {beat}{WARP}")));
                        }
                        sim.stops = stops.into_iter().map(|(beat, seconds)| Stop { beat, seconds }).collect()
                    }
                    "BGSTAGE" => sim.bg_stage = value.to_string(),
                    "NOTES" => sim.step_charts.extend(parse_chart(value)?),
                    _ => {} // Unused tags, such as #SUBTITLE
                }
            }
        }

        if pending.contains('#') {
            return Err(invalid("the file ends in the middle of a tag (missing ';')"));
        }
        if sim.bpms.is_empty() {
            return Err(invalid("missing #BPMS"));
        }
        Ok(sim)
    }
}

/// Some files skip part of the song with a negative BPM or stop.
const WARP: &str = " (a warp, which isn't supported)";

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

/// A finite number: `inf` and `NaN` parse as f64s, but nothing here can use them.
fn parse_number(what: &str, value: &str) -> io::Result<f64> {
    let number = value.trim().parse().ok().filter(|number: &f64| number.is_finite());
    number.ok_or_else(|| invalid(format!("{what}: expected a number, got {value:?}")))
}

/// 0 for an empty value, as for a missing tag.
fn parse_number_or_zero(what: &str, value: &str) -> io::Result<f64> {
    if value.trim().is_empty() { Ok(0.0) } else { parse_number(what, value) }
}

/// Comma-separated `beat=value` pairs, as in `#BPMS` (`0=150,64.5=300`) and `#STOPS`.
fn parse_beat_pairs(tag: &str, value: &str) -> io::Result<Vec<(f64, f64)>> {
    value
        .split(',')
        .filter(|pair| !pair.trim().is_empty())
        .map(|pair| {
            let (beat, value) = pair
                .split_once('=')
                .ok_or_else(|| invalid(format!("{tag}: expected beat=value, got {:?}", pair.trim())))?;
            Ok((parse_number(tag, beat)?, parse_number(tag, value)?))
        })
        .collect()
}

/// One `#NOTES` value: `step type:description:difficulty:meter:groove radar values:note data`.
/// `None` for charts `StepChart` can't hold: other step types and Edit charts.
fn parse_chart(value: &str) -> io::Result<Option<StepChart>> {
    let fields: Vec<&str> = value.splitn(6, ':').map(str::trim).collect();
    let &[step_type, _description, difficulty_name, meter, _radar, note_data] = fields.as_slice() else {
        return Err(invalid(format!("#NOTES: expected 6 ':'-separated fields, got {}", fields.len())));
    };

    // Before the difficulty: other games have difficulties of their own
    if !matches!(step_type, "dance-single" | "dance-double") {
        return Ok(None);
    }
    let Some(difficulty) = parse_difficulty(difficulty_name)? else { return Ok(None) };

    let chart = match step_type {
        "dance-single" => StepChart::DanceSingle(SingleChart {
            difficulty,
            rating: parse_number("#NOTES meter", meter)?,
            measures: parse_measures(note_data)
                .map_err(|e| invalid(format!("#NOTES ({step_type} {difficulty_name}): {e}")))?,
        }),
        _ => StepChart::DanceDouble(DoubleChart {}),
    };
    Ok(Some(chart))
}

/// A `#NOTES` difficulty, by its name or one of the old names StepMania still reads, as
/// StepMania maps them (its "expert" is Challenge, not DDR's Expert). `None` for Edit.
fn parse_difficulty(name: &str) -> io::Result<Option<ChartDifficulty>> {
    let difficulty = match name.to_ascii_lowercase().as_str() {
        "beginner" => ChartDifficulty::Beginner,
        "easy" | "basic" | "light" => ChartDifficulty::Easy,
        "medium" | "another" | "trick" | "standard" | "difficult" => ChartDifficulty::Medium,
        "hard" | "ssr" | "maniac" | "heavy" => ChartDifficulty::Hard,
        "challenge" | "smaniac" | "expert" | "oni" => ChartDifficulty::Challenge,
        "edit" => return Ok(None),
        _ => return Err(invalid(format!("#NOTES: unknown difficulty {name:?}"))),
    };
    Ok(Some(difficulty))
}

/// Measures are separated by `,` and hold one row per line.
fn parse_measures(note_data: &str) -> Result<Vec<Measure>, String> {
    note_data
        .split(',')
        .enumerate()
        .map(|(m, measure)| {
            let rows = measure
                .split_whitespace()
                .map(|row| {
                    parse_row(row).ok_or_else(|| {
                        format!("measure {m}: can't read row {row:?} (expected 4 of 0, 1, 2, 3, 4 or M)")
                    })
                })
                .collect::<Result<_, _>>()?;
            Ok(Measure { num_beats: 4, rows })
        })
        .collect()
}

/// One dance-single row, such as `1002`: left, down, up, right.
fn parse_row(row: &str) -> Option<[Note; 4]> {
    let notes: Vec<Note> = row
        .chars()
        .map(|c| match c {
            '0' => Some(Note::Empty),
            '1' => Some(Note::Tap),
            '2' => Some(Note::HoldStart),
            '3' => Some(Note::End),
            '4' => Some(Note::RollStart),
            'M' => Some(Note::Mine),
            _ => None, // lifts (L), fakes (F), keysounds, etc. have no Note variant yet
        })
        .collect::<Option<_>>()?;
    notes.try_into().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ChartDifficulty::*;

    /// Loads a file or song folder from `test_resources/`. The valid ones come from
    /// `make_test_simfiles.py`.
    fn load(name: &str) -> SimFile {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_resources").join(name);
        SimFile::from_file(&path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()))
    }

    fn singles(sim: &SimFile) -> Vec<&SingleChart> {
        sim.step_charts
            .iter()
            .filter_map(|chart| match chart {
                StepChart::DanceSingle(single) => Some(single),
                StepChart::DanceDouble(_) => None,
            })
            .collect()
    }

    /// The chart's notes in .sm notation, with rows separated by spaces and measures by ", ".
    fn notes_text(chart: &SingleChart) -> String {
        let row_text = |row: &[Note; 4]| -> String {
            row.iter()
                .map(|note| match note {
                    Note::Empty => '0',
                    Note::Tap => '1',
                    Note::HoldStart => '2',
                    Note::End => '3',
                    Note::RollStart => '4',
                    Note::Mine => 'M',
                })
                .collect()
        };
        let measure_text = |measure: &Measure| measure.rows.iter().map(row_text).collect::<Vec<_>>().join(" ");
        chart.measures.iter().map(measure_text).collect::<Vec<_>>().join(", ")
    }

    #[test]
    fn parses_every_note_kind() {
        use Note::*;
        assert_eq!(parse_row("0123"), Some([Empty, Tap, HoldStart, End]));
        assert_eq!(parse_row("4M03"), Some([RollStart, Mine, Empty, End]));
        assert_eq!(parse_row("L000"), None, "lifts aren't supported yet");
        assert_eq!(parse_row("000"), None, "a dance-single row has 4 columns");
    }

    #[test]
    fn loads_every_feature() {
        let sim = load("Feature Test");
        assert_eq!(sim.title, "Feature Test");
        assert_eq!(sim.artist, "テスト");
        assert_eq!(sim.artist_translit, "Tesuto");
        assert_eq!(sim.banner, "Feature Test.png");
        assert_eq!(sim.background, "Feature Test-bg.png");
        assert_eq!(sim.cd_title, "./CDTitles/Feature Test.png");
        assert_eq!(sim.music, "Feature Test.ogg");
        assert_eq!(sim.offset, -0.125);
        assert_eq!((sim.sample_start, sim.sample_length), (12.345, 15.0));
        assert_eq!(sim.bg_stage, "7");
        assert_eq!(
            sim.bpms.iter().map(|b| (b.beat, b.bpm)).collect::<Vec<_>>(),
            [(0.0, 120.0), (16.0, 240.0), (32.5, 60.0), (48.0, 120.0)]
        );
        assert_eq!(sim.stops.iter().map(|s| (s.beat, s.seconds)).collect::<Vec<_>>(), [(8.0, 0.5), (24.25, 0.125)]);

        // The Edit and pump-single charts are skipped, leaving 5 singles and then 2 doubles.
        assert_eq!(sim.step_charts.len(), 7);
        assert!(matches!(&sim.step_charts[5..], [StepChart::DanceDouble(_), StepChart::DanceDouble(_)]));
        let singles = singles(&sim);
        assert_eq!(
            singles.iter().map(|c| (c.difficulty, c.rating)).collect::<Vec<_>>(),
            [(Beginner, 1.0), (Easy, 3.0), (Medium, 5.0), (Hard, 8.0), (Challenge, 11.0)]
        );
        assert!(singles.iter().flat_map(|c| &c.measures).all(|m| m.num_beats == 4));

        // Beginner: taps and jumps.
        assert_eq!(notes_text(singles[0]), "1000 0100 0010 0001, 1001 0110 0000 0000");
        // Easy: holds.
        assert_eq!(notes_text(singles[1]), "2000 0000 3000 0000, 0200 0020 0000 0000 0300 0030 0000 0000");
        // Medium: rolls and mines.
        assert_eq!(notes_text(singles[2]), "4000 0M00 0000 3000, 00M0 000M 0004 0003");
        // Hard: a measure of each size, with a tap on the left first and on the right last.
        let sizes = [4, 8, 12, 16, 24, 32, 48, 64, 96, 192];
        let expected: Vec<String> = sizes.iter().map(|&rows| format!("1000 {}0001", "0000 ".repeat(rows - 2))).collect();
        assert_eq!(notes_text(singles[3]), expected.join(", "));
        // Challenge: all of the above mixed together.
        assert_eq!(
            notes_text(singles[4]),
            "1000 0000 0100 0000 2001 0000 0010 0000 3000 0000 0M00 0000 0004 0000 0000 0000, \
             0000 M000 0000 0003 1100 0000 0011 0000"
        );
    }

    #[test]
    fn loads_a_song_folder_or_the_file_in_it() {
        let from_folder = load("Feature Test");
        let from_file = load("Feature Test/Feature Test.sm");
        assert_eq!(from_folder.title, from_file.title);
        let notes = |sim: &SimFile| singles(sim).into_iter().map(notes_text).collect::<Vec<_>>();
        assert_eq!(notes(&from_folder), notes(&from_file));
    }

    #[test]
    fn loads_a_song_folder_with_its_audio_and_images() {
        let folder = std::env::temp_dir().join(format!("ddr-trainer-song-folder-{}", std::process::id())).join("Song");
        fs::create_dir_all(&folder).unwrap();
        let resources = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_resources");
        fs::copy(resources.join("chart_stats.sm"), folder.join("Song.SM")).unwrap();
        // Named after the folder too, and not text
        for other in ["Song.avi", "Song.ogg", "Song.png", "Song-jacket.png"] {
            fs::write(folder.join(other), [0xff, 0xfe, 0x00, 0x80]).unwrap();
        }
        let loaded = SimFile::from_file(&folder);
        let without = SimFile::from_file(folder.parent().unwrap());
        fs::remove_dir_all(folder.parent().unwrap()).unwrap();
        assert_eq!(loaded.unwrap().title, load("chart_stats.sm").title);
        // No simfile named after that folder
        assert_eq!(without.err().map(|e| e.kind()), Some(ErrorKind::NotFound));
    }

    #[test]
    fn loads_formatting_quirks() {
        let sim = load("quirks.sm");
        // ':' and '#' inside a value, and two tags on one line.
        assert_eq!(sim.title, "Re:Quirks #2");
        assert_eq!(sim.artist, "Nobody");
        // Spaces around a value are trimmed.
        assert_eq!(sim.music, "quirks.ogg");
        // Tags the file leaves out.
        for missing in [&sim.artist_translit, &sim.banner, &sim.background, &sim.cd_title, &sim.bg_stage] {
            assert_eq!(missing, "");
        }
        assert_eq!((sim.offset, sim.sample_start, sim.sample_length), (0.0, 0.0, 0.0));
        // #BPMS over several lines with the commas at the start of each line, and an empty #STOPS.
        assert_eq!(
            sim.bpms.iter().map(|b| (b.beat, b.bpm)).collect::<Vec<_>>(),
            [(0.0, 150.0), (64.0, 300.0), (96.5, 150.0)]
        );
        assert!(sim.stops.is_empty());

        // A tab-indented chart header, a lowercase difficulty, a description with spaces, a row
        // with trailing spaces, a blank line between rows, and comments after ',' and ';'.
        assert_eq!(sim.step_charts.len(), 1);
        let singles = singles(&sim);
        assert_eq!((singles[0].difficulty, singles[0].rating), (Challenge, 10.0));
        assert_eq!(notes_text(singles[0]), "1000 0100 0010 0001, 1111 0000");
    }

    /// (file in `test_resources/invalid/`, part of the error it must fail with). The files come
    /// from `make_invalid_simfiles.py`, and each one's first line says what's broken.
    const INVALID_FILES: [(&str, &str); 15] = [
        ("bad_offset.sm", r#"#OFFSET: expected a number, got "abc""#),
        ("bpm_without_equals.sm", r#"#BPMS: expected beat=value, got "64.000""#),
        ("bad_stop_length.sm", r#"#STOPS: expected a number, got "long""#),
        ("missing_bpms.sm", "missing #BPMS"),
        ("unterminated_tag.sm", "the file ends in the middle of a tag"),
        ("tag_without_semicolon.sm", "#OFFSET: missing ';' (the next tag starts before it ends)"),
        ("negative_bpm.sm", "#BPMS: a BPM must be above 0, got -120 on beat 16 (a warp"),
        ("negative_stop.sm", "#STOPS: a stop can't be negative, got -0.5 on beat 8 (a warp"),
        ("infinite_bpm.sm", r#"#BPMS: expected a number, got "inf""#),
        ("too_few_chart_fields.sm", "#NOTES: expected 6 ':'-separated fields, got 5"),
        ("unknown_difficulty.sm", r#"#NOTES: unknown difficulty "Impossible""#),
        ("bad_meter.sm", r#"#NOTES meter: expected a number, got "eight""#),
        ("unsupported_note.sm", r#"#NOTES (dance-single Hard): measure 1: can't read row "0L00""#),
        ("wrong_row_width.sm", r#"#NOTES (dance-single Hard): measure 0: can't read row "10000""#),
        ("not_utf8.sm", "valid UTF-8"),
    ];

    fn load_error(name: &str) -> io::Error {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_resources").join(name);
        SimFile::from_file(&path).err().unwrap_or_else(|| panic!("{name} loaded, but it should have failed"))
    }

    #[test]
    fn rejects_invalid_files() {
        for (name, expected) in INVALID_FILES {
            let error = load_error(&format!("invalid/{name}"));
            assert_eq!(error.kind(), ErrorKind::InvalidData, "{name}: {error}");
            assert!(error.to_string().contains(expected), "{name}: expected an error containing {expected:?}, got {error:?}");
        }

        // Every file in invalid/ must be listed above
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_resources/invalid");
        let mut files: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.is_file())
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        files.sort();
        let mut listed: Vec<&str> = INVALID_FILES.iter().map(|(name, _)| *name).collect();
        listed.sort();
        assert_eq!(files, listed);
    }

    /// A file that loads, with `old` replaced by `new`.
    fn changed(old: &str, new: &str) -> io::Result<SimFile> {
        let valid = "#TITLE:t;\n#DISPLAYBPM:120;\n#BPMS:0=120;\n#STOPS:4=0.5;\n#SUBTITLE:s;\n#NOTES:dance-single::Hard:8::1000\n0000\n;\n";
        assert!(SimFile::parse(valid).is_ok());
        assert_eq!(valid.matches(old).count(), 1, "{old:?}");
        SimFile::parse(&valid.replace(old, new))
    }

    #[test]
    fn loads_empty_header_numbers_as_zero() {
        let sim = changed("#DISPLAYBPM:120;", "#OFFSET:;\n#SAMPLESTART: ;\n#SAMPLELENGTH:\n;").unwrap();
        assert_eq!((sim.offset, sim.sample_start, sim.sample_length), (0.0, 0.0, 0.0));
        let sim = changed("#DISPLAYBPM:120;", "#OFFSET:-0.5;\n#SAMPLESTART:12;").unwrap();
        assert_eq!((sim.offset, sim.sample_start, sim.sample_length), (-0.5, 12.0, 0.0));
        // Not a number at all is still an error, and so is a chart with no meter
        assert!(changed("#DISPLAYBPM:120;", "#OFFSET:abc;").is_err());
        assert!(changed("Hard:8:", "Hard::").is_err());
    }

    #[test]
    fn reads_the_step_type_before_the_difficulty() {
        let chart = |step_type: &str, difficulty: &str| parse_chart(&format!("{step_type}::{difficulty}:8::1000\n0000\n"));
        let single = |difficulty: &str| match chart("dance-single", difficulty) {
            Ok(Some(StepChart::DanceSingle(single))) => Some(single.difficulty),
            _ => None,
        };
        // Another game's chart is skipped whatever its difficulty is called
        for other in ["pump-single", "lights-cabinet", "kb7-single"] {
            for difficulty in ["Hard", "Crazy", "Nightmare", ""] {
                assert!(matches!(chart(other, difficulty), Ok(None)), "{other} {difficulty:?}");
            }
        }
        // A dance chart's has to be one StepMania reads
        assert!(chart("dance-single", "Crazy").is_err());
        assert!(chart("dance-double", "Crazy").is_err());
        assert!(matches!(chart("dance-double", "Oni"), Ok(Some(StepChart::DanceDouble(_)))));
        assert!(matches!(chart("dance-single", "Edit"), Ok(None)));
        assert!(matches!(chart("dance-double", "edit"), Ok(None)));

        // Old names, in any case
        let expected = [
            (Beginner, &["Beginner", "beginner"][..]),
            (Easy, &["Easy", "Basic", "LIGHT"][..]),
            (Medium, &["Medium", "Another", "Trick", "Standard", "Difficult"][..]),
            (Hard, &["Hard", "SSR", "Maniac", "Heavy"][..]),
            (Challenge, &["Challenge", "SManiac", "Expert", "Oni"][..]),
        ];
        for (difficulty, names) in expected {
            for name in names {
                assert_eq!(single(name), Some(difficulty), "{name}");
            }
        }
    }

    #[test]
    fn rejects_a_tag_that_runs_into_the_next() {
        let error = |old: &str, new: &str| changed(old, new).err().map(|e| e.to_string()).unwrap_or_default();
        // Would have taken the stops for part of an unused tag's value
        assert!(error("#DISPLAYBPM:120;", "#DISPLAYBPM:120").starts_with("#DISPLAYBPM: missing ';'"));
        // Would have taken the chart
        assert!(error("#SUBTITLE:s;", "#SUBTITLE:s").starts_with("#SUBTITLE: missing ';'"));
        assert!(error("#BPMS:0=120;", "#BPMS:0=120").starts_with("#BPMS: missing ';'"));
        // The next tag indented, or after a blank line and a comment
        assert!(error("#TITLE:t;\n", "#TITLE:t\n\n// note\n  \t").starts_with("#TITLE: missing ';'"));

        // A `#` that doesn't start a line is part of the value, and a tag can still span lines
        let sim = changed("#TITLE:t;", "#TITLE:Re:Quirks #2;").unwrap();
        assert_eq!(sim.title, "Re:Quirks #2");
        let sim = changed("#BPMS:0=120;", "#BPMS:0=120\n,8=240\n;#ARTIST:a;").unwrap();
        assert_eq!((sim.bpms.len(), sim.artist.as_str()), (2, "a"));
        // Text outside any tag is still skipped
        assert!(changed("#TITLE:t;", "stray text\n#TITLE:t;").is_ok());
    }

    #[test]
    fn rejects_timing_that_would_run_backwards() {
        let error = |old: &str, new: &str| changed(old, new).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(error("#BPMS:0=120;", "#BPMS:0=120,8=-240;").contains("a BPM must be above 0, got -240 on beat 8"));
        assert!(error("#BPMS:0=120;", "#BPMS:0=0;").contains("a BPM must be above 0, got 0 on beat 0"));
        assert!(error("#STOPS:4=0.5;", "#STOPS:4=-0.25;").contains("a stop can't be negative, got -0.25 on beat 4"));
        // A stop of no length is harmless
        assert!(changed("#STOPS:4=0.5;", "#STOPS:4=0;").is_ok());
        for not_finite in ["inf", "-inf", "NaN", "infinity"] {
            assert!(error("#BPMS:0=120;", &format!("#BPMS:0={not_finite};")).contains("expected a number"), "{not_finite}");
            assert!(error("#STOPS:4=0.5;", &format!("#STOPS:{not_finite}=1;")).contains("expected a number"), "{not_finite}");
            assert!(parse_number("x", not_finite).is_err());
        }
        assert_eq!(parse_number("x", " -1.5e2 ").unwrap(), -150.0);
    }

    #[test]
    fn rejects_paths_with_no_simfile() {
        // The file loads by itself, but its folder doesn't: their names don't match.
        load("invalid/Mismatched Folder/other name.sm");
        for name in ["invalid/Mismatched Folder", "invalid/does not exist.sm"] {
            let error = load_error(name);
            assert_eq!(error.kind(), ErrorKind::NotFound, "{name}: {error:?}");
        }
    }
}