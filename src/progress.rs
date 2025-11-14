use std::sync::{atomic::AtomicBool, Arc, Mutex};

use crate::EncodingError;

trait ThreadSafeReporter: Send {
    fn report(&mut self, progress: f32);
}
impl<F: FnMut(f32) + Send> ThreadSafeReporter for F {
    fn report(&mut self, progress: f32) {
        self(progress);
    }
}

enum ProgressInner<'a> {
    Send(&'a mut dyn ThreadSafeReporter),
    Bound(&'a mut dyn FnMut(f32)),
    None,
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

/// A progress reporter used by [`Encoder`](crate::Encoder) and
/// [`encode()`](crate::encode()).
///
/// This structure is just a wrapper around a function that handles progress
/// reports. A progress report is a single `f32` value between 0 and 1
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
/// ## Multithreaded progress reporting
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
///
/// ## Cancellation
///
/// A progress reporter may optionally be initialized with a
/// [`CancellationToken`] using [`with_cancellation()`](Progress::with_cancellation).
/// This allows operations that receive a [`Progress`] to be cancelled
/// cooperatively and concurrently.
pub struct Progress<'a> {
    reporter: ProgressInner<'a>,
    range: ProgressRange,
    cancel: Option<CancellationToken>,
}
impl<'a> Progress<'a> {
    /// Creates a new progress reporter.
    ///
    /// This reporter supports multithreaded progress reporting. See the
    /// documentation of [`Progress`] for more details.
    pub fn new<F: FnMut(f32) + Send>(reporter: &'a mut F) -> Self {
        Self {
            reporter: ProgressInner::Send(reporter),
            range: ProgressRange::FULL,
            cancel: None,
        }
    }
    /// Creates a new progress reporter.
    ///
    /// This reporter does **not** support multithreaded progress reporting.
    /// See the documentation of [`Progress`] for more details.
    pub fn new_single_threaded<F: FnMut(f32)>(reporter: &'a mut F) -> Self {
        Self {
            reporter: ProgressInner::Bound(reporter),
            range: ProgressRange::FULL,
            cancel: None,
        }
    }
    /// Creates a progress reporter that does nothing when reporting progress.
    ///
    /// This can be used in combination with [`with_cancellation`](Progress::with_cancellation)
    /// to create a progress reporter that only handles cancellation.
    pub fn none() -> Self {
        Self {
            reporter: ProgressInner::None,
            range: ProgressRange::FULL,
            cancel: None,
        }
    }

    /// Sets a cancellation token that can be used to cancel an operation given
    /// this progress reporter.
    pub fn with_cancellation(mut self, token: &CancellationToken) -> Self {
        self.cancel = Some(token.clone());
        self
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
            ProgressInner::None => {}
        }
    }

    /// Returns whether the operation has been cancelled.
    ///
    /// If no [`CancellationToken`] was set, this will always return `false`.
    /// If a token was set, this will return [`CancellationToken::is_cancelled`].
    pub fn is_cancelled(&self) -> bool {
        if let Some(cancel) = &self.cancel {
            cancel.is_cancelled()
        } else {
            false
        }
    }

    pub(crate) fn sub_range(&mut self, range: ProgressRange) -> Progress<'_> {
        let range = self.range.sub_range(range);
        Progress {
            reporter: match &mut self.reporter {
                ProgressInner::Send(f) => ProgressInner::Send(*f),
                ProgressInner::Bound(f) => ProgressInner::Bound(*f),
                ProgressInner::None => ProgressInner::None,
            },
            range,
            cancel: self.cancel.clone(),
        }
    }

    pub(crate) fn check_cancelled(&self) -> Result<(), EncodingError> {
        if self.is_cancelled() {
            Err(EncodingError::Cancelled)
        } else {
            Ok(())
        }
    }
    pub(crate) fn checked_report(&mut self, progress: f32) -> Result<(), EncodingError> {
        self.check_cancelled()?;
        self.report(progress);
        Ok(())
    }
    /// Always checks if the operation has been cancelled, and reports progress if the condition is true.
    pub(crate) fn checked_report_if(
        &mut self,
        cond: bool,
        progress: f32,
    ) -> Result<(), EncodingError> {
        if cond {
            self.checked_report(progress)
        } else {
            self.check_cancelled()
        }
    }
}

/// A token that can be used to cooperatively cancel an operation.
///
/// A `CancellationToken` can be shared between multiple threads. When one
/// thread calls [`cancel()`](CancellationToken::cancel), other threads can
/// check the cancellation state using [`is_cancelled()`](CancellationToken::is_cancelled).
///
/// ## Linked tokens
///
/// Cloning a `CancellationToken` will result in a new token that shares the
/// same cancellation state. Cancelling one token will cancel all clones and
/// vice versa. This behavior can be used to cancel multiple operations at
/// once.
///
/// ## See also
///
/// - [`Progress::with_cancellation()`]
/// - [`Progress::is_cancelled()`]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}
impl CancellationToken {
    /// Creates a new cancellation token that is not cancelled.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Sets the cancellation state to cancelled.
    ///
    /// Any operation that checks this cancellation state will be cancelled,
    /// unless the cancellation state has been [reset](Self::reset) before the
    /// check.
    pub fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Resets the cancellation state to not cancelled.
    ///
    /// If operations that have previously been cancelled are still running
    /// while the cancellation state is reset, they may continue running til
    /// completion or be cancelled again. To cancel operations properly, only
    /// reset after all operations checking the cancellation state have
    /// completed.
    pub fn reset(&self) {
        self.cancelled
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Returns whether the operation has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }
}
impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}
impl Clone for CancellationToken {
    /// Creates a clone of this [`CancellationToken`], which will be cancelled
    /// if this token gets cancelled and vice versa. The cancellation state is
    /// shared between all clones.
    fn clone(&self) -> Self {
        CancellationToken {
            cancelled: self.cancelled.clone(),
        }
    }
}
const _: () = {
    // Ensure CancellationToken is Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}
    let _ = assert_send_sync::<CancellationToken>;
};

/// The current progress and a function to report it.
type InnerState<'a> = (u64, &'a mut dyn ThreadSafeReporter);
pub(crate) struct ParallelProgress<'a> {
    progress: Option<Mutex<InnerState<'a>>>,
    total: u64,
    range: ProgressRange,
    cancel: Option<CancellationToken>,
}
impl<'a> ParallelProgress<'a> {
    pub fn new(progress: &'a mut &mut Progress, total: u64) -> Self {
        Self {
            progress: match &mut progress.reporter {
                ProgressInner::Send(f) => Some(Mutex::new((0, *f))),
                _ => None,
            },
            total,
            range: progress.range,
            cancel: progress.cancel.clone(),
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

    pub fn is_cancelled(&self) -> bool {
        self.cancel
            .as_ref()
            .map(|c| c.is_cancelled())
            .unwrap_or(false)
    }
    pub fn check_cancelled(&self) -> Result<(), EncodingError> {
        if self.is_cancelled() {
            Err(EncodingError::Cancelled)
        } else {
            Ok(())
        }
    }
}
