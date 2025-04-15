use std::sync::Mutex;

trait ThreadSafeReporter: Send {
    fn report(&mut self, progress: f32);
}
impl<F: FnMut(f32) + Send> ThreadSafeReporter for F {
    fn report(&mut self, progress: f32) {
        self(progress);
    }
}

pub(crate) trait Report {
    fn report(&mut self, progress: f32);
}

enum ProgressInner<'a> {
    Send(&'a mut dyn ThreadSafeReporter),
    Bound(&'a mut dyn FnMut(f32)),
}

#[derive(Clone, Copy)]
pub(crate) struct ProgressRange {
    pub start: f32,
    pub length: f32,
}
impl ProgressRange {
    pub const FULL: Self = Self {
        start: 0.0,
        length: 1.0,
    };

    pub fn from_to(from: f32, to: f32) -> Self {
        debug_assert!(from <= to);
        Self {
            start: from,
            length: to - from,
        }
    }

    pub fn sub_range(&self, other: Self) -> Self {
        Self {
            start: self.start + other.start * self.length,
            length: other.length * self.length,
        }
    }

    pub fn project(&self, progress: f32) -> f32 {
        debug_assert!((0.0..=1.0).contains(&progress));
        self.start + self.length * progress
    }
}
impl Default for ProgressRange {
    fn default() -> Self {
        Self::FULL
    }
}

/// A progress reporter used by [`crate::Encoder`] and [`crate::encode()`].
///
/// This structure is just a wrapper around a function that handles progress
/// reports. A progress report is a single `f32` value between 0 to 1
/// representing the percentage of the task that has been completed.
///
/// Encoders will generally try to make 10 to 50 progress reports per second.
/// This is a best effort basis, so reports may be more or less frequent. In
/// certain scenarios, there may be dozens of reports within 1 millisecond.
///
/// Reports are guaranteed to only increase or stay the same. I.e. after 40%
/// has been reported, the next report may be 40% again, 42%, or similar, but
/// never 39% or lower. In particular, there may be multiple reports for 100%.
///
/// ### Multi-threaded progress reporting
///
/// Encoding may happen in parallel across multiple threads. As such, progress
/// reports may come from multiple threads at the same time.
///
/// The main constructor [`Progress::new()`] takes a report function that can
/// be called from multiple threads. This is the **recommended** constructor.
///
/// If you cannot provide a thread-safe function, use
/// [`Progress::new_single_threaded()`]. The main downside of this is that
/// progress reports will be less frequent. In particular, formats that encoded
/// in parallel (e.g. BC1) will suddenly jump from 0% to 100% without any
/// progress in between.
///
/// Note that single-threaded progress reporters will **not** prevent or
/// interfere with parallel encoding in any way. There's no performance
/// penalty.
pub struct Progress<'a> {
    reporter: ProgressInner<'a>,
    range: ProgressRange,
}
impl<'a> Progress<'a> {
    /// Creates a new progress reporter.
    ///
    /// This reporter supports multi-threaded progress reporting. See the
    /// documentation of [`Progress`] for more details.
    pub fn new<F: FnMut(f32) + Send>(reporter: &'a mut F) -> Self {
        Self {
            reporter: ProgressInner::Send(reporter),
            range: ProgressRange::FULL,
        }
    }
    /// Creates a new progress reporter.
    ///
    /// This reporter does **not** support multi-threaded progress reporting.
    /// See the documentation of [`Progress`] for more details.
    pub fn new_single_threaded<F: FnMut(f32)>(reporter: &'a mut F) -> Self {
        Self {
            reporter: ProgressInner::Bound(reporter),
            range: ProgressRange::FULL,
        }
    }

    /// Calls the underlying reporter function with the given progress.
    ///
    /// ### Panics
    ///
    /// If the function is called with a value outside the range of `0.0..=1.0`,
    /// the function will panic **if debug assertions are enabled**.
    ///
    /// The underlying reporter may also panic.
    pub fn report(&mut self, progress: f32) {
        let progress = self.range.project(progress);
        match &mut self.reporter {
            ProgressInner::Send(report) => report.report(progress),
            ProgressInner::Bound(report) => report(progress),
        }
    }
}
impl Report for Progress<'_> {
    fn report(&mut self, progress: f32) {
        self.report(progress);
    }
}
impl Report for Option<&mut Progress<'_>> {
    fn report(&mut self, progress: f32) {
        if let Some(reporter) = self {
            reporter.report(progress);
        }
    }
}

pub(crate) struct SubProgress<'a, 'b> {
    progress: Option<&'a mut Progress<'b>>,
    original_range: ProgressRange,
}
impl<'a, 'b> SubProgress<'a, 'b> {
    pub fn new(progress: Option<&'a mut Progress<'b>>) -> Self {
        let original_range = progress.as_ref().map(|p| p.range).unwrap_or_default();
        Self {
            progress,
            original_range,
        }
    }
    pub fn get(&mut self) -> Option<&'_ mut Progress<'b>> {
        self.progress.as_deref_mut()
    }

    pub fn set_range(&mut self, range: ProgressRange) {
        if let Some(progress) = self.progress.as_mut() {
            progress.range = self.original_range.sub_range(range);
        }
    }
    pub fn set_original_range(&mut self) {
        self.set_range(self.original_range);
    }
}
impl Drop for SubProgress<'_, '_> {
    fn drop(&mut self) {
        if let Some(progress) = self.progress.as_mut() {
            progress.range = self.original_range;
        }
    }
}
impl Report for SubProgress<'_, '_> {
    fn report(&mut self, progress: f32) {
        if let Some(reporter) = self.progress.as_mut() {
            reporter.report(progress);
        }
    }
}

/// The current progress and a function to report it.
type InnerState<'a> = (u64, &'a mut dyn ThreadSafeReporter);
pub(crate) struct ParallelProgress<'a> {
    progress: Option<Mutex<InnerState<'a>>>,
    total: u64,
    range: ProgressRange,
}
impl<'a> ParallelProgress<'a> {
    pub fn new(progress: &'a mut Option<&mut Progress>, total: u64) -> Self {
        let range = progress.as_ref().map(|p| p.range).unwrap_or_default();

        Self {
            progress: match progress {
                Some(Progress {
                    reporter: ProgressInner::Send(f),
                    ..
                }) => Some(Mutex::new((0, *f))),
                _ => None,
            },
            total,
            range,
        }
    }

    pub fn submit(&self, progress: u64) {
        if let Some(mutex) = self.progress.as_ref() {
            let mut guard = mutex.lock().unwrap();
            guard.0 += progress;
            let progress = self.range.project(guard.0 as f32 / self.total as f32);
            guard.1.report(progress);
        }
    }
}
