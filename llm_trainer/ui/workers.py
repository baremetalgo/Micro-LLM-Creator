from __future__ import annotations

from queue import Queue
from threading import Event
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal


class TaskWorker(QObject):
    """Background worker used for long-running UI tasks."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        fn: Any,
        *args: Any,
        progress_queue: Optional[Queue] = None,
        with_progress: bool = False,
        stop_event: Optional[Event] = None,
    ) -> None:
        """Create a worker.

        Args:
            fn: Callable to execute in the worker thread.
            *args: Positional arguments passed to ``fn``.
            progress_queue: Optional queue for progress events.
            with_progress: Whether to pass a progress callback to ``fn``.
            stop_event: Optional event used for cooperative cancellation.
        """

        super().__init__()
        self.fn = fn
        self.args = args
        self.progress_queue = progress_queue
        self.with_progress = with_progress
        self.stop_event = stop_event

    def run(self) -> None:
        """Execute the worker function and emit completion or failure."""

        try:
            if self.with_progress:
                self.finished.emit(self.fn(*self.args, progress=self._queue_progress, should_stop=self._should_stop))
            else:
                self.finished.emit(self.fn(*self.args))
        except Exception as exc:
            self.failed.emit(str(exc))

    def _queue_progress(self, event: Any) -> None:
        if self.progress_queue is not None:
            self.progress_queue.put(event)

    def _should_stop(self) -> bool:
        """Return whether the active task has been asked to stop.

        Returns:
            True when the cooperative stop event is set.
        """

        return bool(self.stop_event and self.stop_event.is_set())
