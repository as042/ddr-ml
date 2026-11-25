//! Printing that outlives its reader.

/// `println!`, but if the reader has gone (`ddr-trainer feet song | head`) the run ends quietly,
/// where `println!` panics.
macro_rules! say {
    () => { $crate::out::line(format_args!("")) };
    ($($arg:tt)*) => { $crate::out::line(format_args!($($arg)*)) };
}

pub fn line(text: std::fmt::Arguments) {
    // A direct write would get past the test harness's capture
    if cfg!(test) {
        println!("{text}");
        return;
    }
    use std::io::{ErrorKind, Write as _};
    match writeln!(std::io::stdout(), "{text}") {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::BrokenPipe => std::process::exit(0),
        Err(e) => panic!("failed printing to stdout: {e}"),
    }
}
