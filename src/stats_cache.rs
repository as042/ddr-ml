//! A cache of chart statistics, so a simfile is only analyzed again when it or the analyzer
//! changes.
//!
//! Entries are keyed by a hash of the simfile's contents, and record the analyzer's fingerprint
//! (a hash of its source, see `FINGERPRINTED`). An entry with another fingerprint is out of date.

use std::{
    collections::HashMap,
    fs,
    io::{self, BufWriter, ErrorKind},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::chart_analyzer::{AnalyzedChart, analyze_singles};
use crate::sim_loader::{SimFile, simfile_path};

/// The source files that decide a simfile's statistics, with their paths from `src/`. Any change
/// to them, even to a comment, makes every cached entry out of date.
const FINGERPRINTED: [(&str, &str); 4] = [
    ("sim_loader.rs", include_str!("sim_loader.rs")),
    ("chart_analyzer.rs", include_str!("chart_analyzer.rs")),
    ("chart_analyzer/footing.rs", include_str!("chart_analyzer/footing.rs")),
    ("stats_cache.rs", include_str!("stats_cache.rs")),
];

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// 64-bit FNV-1a, continuing from `hash`. These hashes are saved, and std's hasher can change
/// between Rust releases.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn analyzer_fingerprint() -> u64 {
    FINGERPRINTED
        .iter()
        .fold(FNV_OFFSET_BASIS, |hash, (path, source)| fnv1a(fnv1a(hash, path.as_bytes()), source.as_bytes()))
}

/// A simfile's cache key: a hash of its contents, as 16 hex digits.
fn key(content: &str) -> String {
    format!("{:016x}", fnv1a(FNV_OFFSET_BASIS, content.as_bytes()))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct CacheEntry {
    /// For people reading the cache file. The key is what identifies the simfile.
    title: String,
    artist: String,
    /// Seconds since the Unix epoch
    analyzed_at: u64,
    /// The analyzer's fingerprint at the time
    fingerprint: u64,
    charts: Vec<AnalyzedChart>,
}

/// Chart statistics, keyed by a hash of each simfile's contents. Keep it between runs with
/// `save` and `load`.
#[derive(Debug)]
pub struct StatsCache {
    entries: HashMap<String, CacheEntry>,
    /// The running analyzer's. Entries with any other are out of date.
    fingerprint: u64,
    hits: usize,
    misses: usize,
}

/// File layout, for saving.
#[derive(Serialize)]
struct CacheFile<'a> {
    entries: &'a HashMap<String, CacheEntry>,
}

/// File layout, for loading. Entries stay raw so one that doesn't parse can be dropped alone.
#[derive(Deserialize)]
struct RawCacheFile {
    entries: HashMap<String, serde_json::Value>,
}

impl StatsCache {
    pub fn new() -> Self {
        StatsCache { entries: HashMap::new(), fingerprint: analyzer_fingerprint(), hits: 0, misses: 0 }
    }

    /// A missing file gives an empty cache. Entries that don't parse (an older format, or a NaN
    /// saved as `null`) are dropped. A file that isn't a cache is an error, so `save` can't
    /// quietly overwrite it.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        let mut cache = StatsCache::new();
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(cache),
            Err(e) => return Err(e),
        };
        let raw: RawCacheFile = serde_json::from_str(&text).map_err(|e| {
            io::Error::new(ErrorKind::InvalidData, format!("{} isn't a stats cache: {e}", path.display()))
        })?;
        cache.entries = raw
            .entries
            .into_iter()
            .filter_map(|(key, entry)| Some((key, serde_json::from_value(entry).ok()?)))
            .collect();
        Ok(cache)
    }

    /// Writes JSON to a file beside `path`, then moves it into place, so a partial write can't
    /// leave half a cache.
    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let mut temp = path.as_os_str().to_owned();
        temp.push(".tmp");
        let temp = PathBuf::from(temp);

        let mut writer = BufWriter::new(fs::File::create(&temp)?);
        serde_json::to_writer(&mut writer, &CacheFile { entries: &self.entries })?;
        writer.into_inner()?.sync_all()?;
        fs::rename(&temp, path)
    }

    /// Every dance-single chart's statistics for a simfile or song folder, from the cache when it
    /// has them for this content and analyzer. A file that doesn't load is an error, and isn't
    /// cached.
    pub fn analyze_file(&mut self, path: impl AsRef<Path>) -> io::Result<&[AnalyzedChart]> {
        let content = fs::read_to_string(simfile_path(path)?)?;
        let key = key(&content);
        if self.entries.get(&key).is_some_and(|entry| entry.fingerprint == self.fingerprint) {
            self.hits += 1;
        } else {
            let sim = SimFile::parse(&content)?;
            let entry = CacheEntry {
                title: sim.title.clone(),
                artist: sim.artist.clone(),
                analyzed_at: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |since| since.as_secs()),
                fingerprint: self.fingerprint,
                charts: analyze_singles(&sim),
            };
            self.entries.insert(key.clone(), entry);
            self.misses += 1;
        }
        Ok(&self.entries[&key].charts)
    }

    /// `analyze_file` calls answered from the cache.
    pub fn hits(&self) -> usize {
        self.hits
    }

    /// `analyze_file` calls that had to analyze the file.
    pub fn misses(&self) -> usize {
        self.misses
    }
}

impl Default for StatsCache {
    fn default() -> Self {
        StatsCache::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("test_resources").join(name)
    }

    /// Deleted on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(test: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("ddr-trainer-stats-cache-{test}-{}", std::process::id()));
            fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn hashes_stably() {
        // FNV-1a's published test vectors
        assert_eq!(fnv1a(FNV_OFFSET_BASIS, b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a(FNV_OFFSET_BASIS, b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a(FNV_OFFSET_BASIS, b"foobar"), 0x85944171f73967e8);
        assert_eq!(key("foobar"), "85944171f73967e8");
    }

    #[test]
    fn reuses_cached_stats() {
        let mut cache = StatsCache::new();
        let expected = analyze_singles(&SimFile::from_file(resource("chart_stats.sm")).unwrap());
        assert_eq!(cache.analyze_file(resource("chart_stats.sm")).unwrap(), expected.as_slice());
        assert_eq!((cache.hits(), cache.misses()), (0, 1));
        assert_eq!(cache.analyze_file(resource("chart_stats.sm")).unwrap(), expected.as_slice());
        assert_eq!((cache.hits(), cache.misses()), (1, 1));

        // A song folder and the file in it share an entry
        let expected = analyze_singles(&SimFile::from_file(resource("Feature Test")).unwrap());
        assert_eq!(cache.analyze_file(resource("Feature Test")).unwrap(), expected.as_slice());
        assert_eq!(cache.analyze_file(resource("Feature Test/Feature Test.sm")).unwrap(), expected.as_slice());
        assert_eq!((cache.hits(), cache.misses()), (2, 2));
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn round_trips_through_a_file() {
        let dir = TempDir::new("round-trip");
        let path = dir.0.join("stats.json");
        let files = ["chart_stats.sm", "bpm_stats.sm", "nps_stats.sm", "pattern_stats.sm", "footing_stats.sm", "Feature Test"];
        let mut cache = StatsCache::new();
        for file in files {
            cache.analyze_file(resource(file)).unwrap();
        }
        cache.save(&path).unwrap();
        assert!(!dir.0.join("stats.json.tmp").exists(), "the temporary file should be moved into place");

        // Statistics round-trip exactly, so every file is a hit
        let mut loaded = StatsCache::load(&path).unwrap();
        assert_eq!(loaded.entries, cache.entries);
        for file in files {
            loaded.analyze_file(resource(file)).unwrap();
        }
        assert_eq!((loaded.hits(), loaded.misses()), (files.len(), 0));
    }

    #[test]
    fn analyzes_outdated_and_edited_files_again() {
        let dir = TempDir::new("outdated");
        let original = fs::read_to_string(resource("chart_stats.sm")).unwrap();
        let (file, copy, edited) = (dir.0.join("file.sm"), dir.0.join("copy.sm"), dir.0.join("edited.sm"));
        fs::write(&file, &original).unwrap();
        fs::write(&copy, &original).unwrap();
        fs::write(&edited, format!("{original}// an edit\r\n")).unwrap();

        // A copy shares the entry. An edit gets a new one.
        let mut cache = StatsCache::new();
        cache.analyze_file(&file).unwrap();
        cache.analyze_file(&copy).unwrap();
        assert_eq!((cache.hits(), cache.misses()), (1, 1));
        cache.analyze_file(&edited).unwrap();
        assert_eq!((cache.hits(), cache.misses()), (1, 2));
        assert_eq!(cache.entries.len(), 2);

        // As if analyzed by different code, long ago
        let entry = cache.entries.get_mut(&key(&original)).unwrap();
        entry.fingerprint ^= 1;
        entry.analyzed_at = 0;
        cache.analyze_file(&file).unwrap();
        assert_eq!((cache.hits(), cache.misses()), (1, 3));
        let entry = &cache.entries[&key(&original)];
        assert_eq!(entry.fingerprint, cache.fingerprint);
        assert!(entry.analyzed_at > 0);
    }

    #[test]
    fn drops_entries_it_cant_read() {
        let dir = TempDir::new("unreadable");
        let path = dir.0.join("stats.json");
        let mut cache = StatsCache::new();
        cache.analyze_file(resource("chart_stats.sm")).unwrap();
        let entry = cache.entries.values().next().unwrap().clone();

        // A NaN statistic (saved as null), and an entry from before a field was added
        let good = serde_json::to_value(&entry).unwrap();
        let mut null_stat = good.clone();
        null_stat["charts"][0]["stats"]["chart_dur"] = serde_json::Value::Null;
        let mut missing_field = good.clone();
        missing_field.as_object_mut().unwrap().remove("fingerprint");
        let file = serde_json::json!({
            "entries": { "good": good, "null stat": null_stat, "missing field": missing_field }
        });
        fs::write(&path, file.to_string()).unwrap();

        let loaded = StatsCache::load(&path).unwrap();
        assert_eq!(loaded.entries.keys().collect::<Vec<_>>(), ["good"]);
        assert_eq!(loaded.entries["good"], entry);
    }

    #[test]
    fn loads_only_caches() {
        let dir = TempDir::new("not-a-cache");
        assert!(StatsCache::load(dir.0.join("missing.json")).unwrap().entries.is_empty());
        for (name, contents) in [("model_config.json", r#"{"d_input": 3}"#), ("garbage.json", "not json")] {
            fs::write(dir.0.join(name), contents).unwrap();
            let error = StatsCache::load(dir.0.join(name)).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidData, "{name}: {error}");
        }
    }

    #[test]
    fn fingerprints_all_the_analyzer() {
        // Every analyzer file must be listed, each under the path it's included from.
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut expected: Vec<String> = ["sim_loader.rs", "chart_analyzer.rs", "stats_cache.rs"].map(String::from).into();
        for entry in fs::read_dir(src.join("chart_analyzer")).unwrap() {
            expected.push(format!("chart_analyzer/{}", entry.unwrap().file_name().to_string_lossy()));
        }
        expected.sort();
        let mut listed: Vec<String> = FINGERPRINTED.iter().map(|(path, _)| path.to_string()).collect();
        listed.sort();
        assert_eq!(listed, expected);
        for (path, source) in FINGERPRINTED {
            assert_eq!(fs::read_to_string(src.join(path)).unwrap(), source, "{path}");
        }
    }
}
