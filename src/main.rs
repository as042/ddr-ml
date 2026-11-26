#[macro_use]
mod out;
mod dataset;
mod device;
mod model;
mod pca;
mod train;
mod sim_loader;
mod chart_analyzer;
mod cma_es;
mod cost_fit;
mod stats_cache;
mod tools;

#[cfg(test)]
mod tests;

use std::{error::Error, process::ExitCode};

const USAGE: &str = "\
usage: ddr-trainer <tool> [arguments]

tools:
  train [songs]                train a rating model on a folder of song folders and save it
  pca [songs]                  fit a PCA of the chart statistics and save it
  infer <simfile>...           rate each chart with the saved model
  feet <arrows> [speed]        foot one segment, e.g. UDDRLRURDL (a jump in brackets: L[UD]R;
                               a hold from D+ to D-, or to the end: D+RUD-L; a space is a
                               pause of one row: \"LR  UD\"). A row is a 16th at 120 BPM, or
                               speed times as fast, e.g. 2
  feet <simfile>               list a simfile's charts and what their footings add up to
  feet <simfile> <difficulty>  foot a chart, e.g. hard
  analyze <simfile> <difficulty> [--footing] [--<cost or setting> <value>]...
                               run the whole chart analyzer on one chart and print its
                               statistics as JSON, for other programs. Any cost or setting can
                               be set (no arguments lists them). A simfile of - is read from
                               stdin. --footing adds every note's footing and every technique
                               found, by note
  score-costs <sheet> [songs]  score the foot search's costs against a chart spreadsheet (.csv)
  scan-costs <sheet> [songs]   score them, then again with each cost changed on its own
  fit-costs <sheet> [songs]    search for the costs closest to the spreadsheet, on three songs in
                               four, and score them on the rest (about 40 minutes for DDR A3)

songs defaults to the dataset, the packs in data/packs from DDR A to WORLD, and for the cost
tools to its DDR A3 pack, which the sheet counts. Run from the repo. A simfile is a .sm or .ssc
file, or its song folder.";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The first argument picks the tool, the rest are its own.
fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    let Some((tool, args)) = args.split_first() else {
        say!("{USAGE}");
        return Ok(());
    };
    match tool.as_str() {
        "train" => tools::train(args),
        "pca" => tools::pca(args),
        "infer" => tools::infer(args),
        "feet" => tools::feet(args),
        "analyze" => tools::analyze(args),
        "score-costs" => tools::score_costs(args),
        "scan-costs" => tools::scan_costs(args),
        "fit-costs" => tools::fit_costs(args),
        "help" | "-h" | "--help" => {
            say!("{USAGE}");
            Ok(())
        }
        _ => Err(format!("unknown tool `{tool}`\n\n{USAGE}").into()),
    }
}
