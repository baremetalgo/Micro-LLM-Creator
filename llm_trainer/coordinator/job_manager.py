from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from llm_trainer.backends.base import ProgressCallback, StopCallback, TrainerBackend
from llm_trainer.backends.registry import DEFAULT_BACKEND_REGISTRY, BackendRegistry
from llm_trainer.contracts import (
    BackendKind,
    ClaimJobRequest,
    ClaimJobResponse,
    CompleteJobRequest,
    CompleteJobResponse,
    FailJobRequest,
    FailJobResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    JobStatus,
    ProgressReportRequest,
    ProgressReportResponse,
    ProtocolStatus,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    TrainingMetrics,
    TrainingJobSpec,
    TrainingResultSpec,
    WorkerAvailability,
    utc_now_iso,
)
from llm_trainer.coordinator.state_store import JobStateStore
from llm_trainer.training import TrainingResult


class WorkerStatus(str, Enum):
    """Worker availability state."""

    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class WorkerDescriptor:
    """Training worker registered with the job manager.

    Attributes:
        worker_id: Stable worker identifier.
        backend: Backend kind this worker can execute.
        status: Current availability state.
        device: Device advertised by the worker.
        hostname: Optional worker host name.
        capabilities: Free-form hardware/runtime capabilities.
    """

    worker_id: str
    backend: BackendKind = BackendKind.LOCAL
    status: WorkerStatus = WorkerStatus.AVAILABLE
    device: str = "auto"
    hostname: Optional[str] = None
    capabilities: dict[str, Any] = field(default_factory=dict)
    last_heartbeat_at: Optional[str] = None

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the worker descriptor to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            "worker_id": self.worker_id,
            "backend": self.backend.value,
            "status": self.status.value,
            "device": self.device,
            "hostname": self.hostname,
            "capabilities": self.capabilities,
            "last_heartbeat_at": self.last_heartbeat_at,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "WorkerDescriptor":
        """Create a worker descriptor from JSON-friendly values.

        Args:
            data: Serialized worker data.

        Returns:
            Worker descriptor.
        """

        return cls(
            worker_id=data["worker_id"],
            backend=BackendKind(data.get("backend", BackendKind.LOCAL.value)),
            status=WorkerStatus(data.get("status", WorkerStatus.AVAILABLE.value)),
            device=data.get("device", "auto"),
            hostname=data.get("hostname"),
            capabilities=dict(data.get("capabilities") or {}),
            last_heartbeat_at=data.get("last_heartbeat_at"),
        )


@dataclass
class WorkerHeartbeat:
    """Heartbeat reported by a training worker.

    Attributes:
        worker_id: Worker identifier.
        status: Worker availability state.
        backend: Backend kind the worker can execute.
        active_job_id: Job currently running on the worker.
        device: Worker device.
        metrics: Runtime metrics reported by the worker.
        timestamp: UTC heartbeat timestamp.
    """

    worker_id: str
    status: WorkerStatus
    backend: BackendKind = BackendKind.LOCAL
    active_job_id: Optional[str] = None
    device: str = "auto"
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the heartbeat to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "backend": self.backend.value,
            "active_job_id": self.active_job_id,
            "device": self.device,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


@dataclass
class ManagedJob:
    """Job state tracked by the manager.

    Attributes:
        spec: Training job contract.
        assigned_worker_id: Worker currently assigned to this job.
        result: Serializable result metadata when finished.
        error: Error text when failed.
    """

    spec: TrainingJobSpec
    assigned_worker_id: Optional[str] = None
    result: Optional[TrainingResultSpec] = None
    error: Optional[str] = None
    latest_metrics: Optional[TrainingMetrics] = None
    updated_at: str = field(default_factory=utc_now_iso)

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the managed job to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            "spec": self.spec.to_jsonable(),
            "assigned_worker_id": self.assigned_worker_id,
            "result": _jsonable_result(self.result) if self.result else None,
            "error": self.error,
            "latest_metrics": self.latest_metrics.__dict__ if self.latest_metrics else None,
            "updated_at": self.updated_at,
        }


class JobManager:
    """Coordinates training jobs across available backends.

    The first implementation runs local jobs in-process. The API is intentionally
    shaped like a distributed coordinator so remote clients can be added without
    rewriting the desktop app's training calls.
    """

    def __init__(
        self,
        registry: Optional[BackendRegistry] = None,
        state_store: Optional[JobStateStore] = None,
    ) -> None:
        """Create a job manager.

        Args:
            registry: Backend registry used to resolve job backends.
            state_store: Optional persistent state store.
        """

        self.registry = registry or DEFAULT_BACKEND_REGISTRY
        self.state_store = state_store or JobStateStore()
        self._jobs: dict[str, ManagedJob] = {}
        self._workers: dict[str, WorkerDescriptor] = {}
        self._stop_requested: set[str] = set()
        self._restore_state()
        self.register_worker(
            WorkerDescriptor(
                worker_id="local",
                backend=BackendKind.LOCAL,
                status=WorkerStatus.AVAILABLE,
                device="auto",
                hostname="localhost",
            )
        )

    def register_worker(self, worker: WorkerDescriptor) -> None:
        """Register or update a worker.

        Args:
            worker: Worker descriptor.
        """

        self._workers[worker.worker_id] = worker
        self._persist_worker(worker)

    def register_remote_worker(self, request: RegisterWorkerRequest) -> RegisterWorkerResponse:
        """Register a remote worker from a protocol request.

        Args:
            request: Register worker request.

        Returns:
            Register worker response.
        """

        if not request.worker_id.strip():
            return RegisterWorkerResponse(
                worker_id=request.worker_id,
                accepted=False,
                status=ProtocolStatus.REJECTED,
                message="worker_id is required",
            )
        capabilities = request.capabilities.to_jsonable()
        if request.labels:
            capabilities["labels"] = request.labels
        self.register_worker(
            WorkerDescriptor(
                worker_id=request.worker_id,
                backend=request.backend,
                status=WorkerStatus.AVAILABLE,
                device=request.device,
                hostname=request.capabilities.hostname,
                capabilities=capabilities,
                last_heartbeat_at=utc_now_iso(),
            )
        )
        return RegisterWorkerResponse(
            worker_id=request.worker_id,
            accepted=True,
            status=ProtocolStatus.OK,
            heartbeat_interval_seconds=10,
            message="worker registered",
        )

    def record_heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        """Record a worker heartbeat and update worker availability.

        Args:
            heartbeat: Worker heartbeat.
        """

        worker = self._workers.get(
            heartbeat.worker_id,
            WorkerDescriptor(
                worker_id=heartbeat.worker_id,
                backend=heartbeat.backend,
                device=heartbeat.device,
            ),
        )
        worker.backend = heartbeat.backend
        worker.status = heartbeat.status
        worker.device = heartbeat.device
        worker.last_heartbeat_at = heartbeat.timestamp
        self._workers[worker.worker_id] = worker
        self._persist_worker(worker)
        self.state_store.record_heartbeat(worker.worker_id, heartbeat.to_jsonable())

    def handle_heartbeat(self, request: HeartbeatRequest) -> HeartbeatResponse:
        """Handle a protocol heartbeat request from a worker.

        Args:
            request: Heartbeat request.

        Returns:
            Heartbeat response.
        """

        if not request.worker_id.strip():
            return HeartbeatResponse(
                status=ProtocolStatus.REJECTED,
                should_stop_job=True,
                message="worker_id is required",
            )
        heartbeat = WorkerHeartbeat(
            worker_id=request.worker_id,
            status=_availability_to_worker_status(request.availability),
            backend=request.backend,
            active_job_id=request.active_job_id,
            device=request.device,
            metrics=request.metrics,
            timestamp=request.sent_at,
        )
        self.record_heartbeat(heartbeat)
        should_stop = bool(request.active_job_id and self._should_stop_remote_job(request.active_job_id))
        should_pause = bool(request.active_job_id and self._should_pause_remote_job(request.active_job_id))
        return HeartbeatResponse(
            status=ProtocolStatus.OK,
            should_stop_job=should_stop,
            should_pause_job=should_pause,
            message=_control_message(should_stop, should_pause, "heartbeat accepted"),
        )

    def handle_claim_job(self, request: ClaimJobRequest) -> ClaimJobResponse:
        """Assign a queued job to a worker when compatible work exists.

        Args:
            request: Claim job request.

        Returns:
            Claim job response containing an assigned job when available.
        """

        if not request.worker_id.strip():
            return ClaimJobResponse(status=ProtocolStatus.REJECTED, message="worker_id is required")
        worker = self._workers.get(
            request.worker_id,
            WorkerDescriptor(
                worker_id=request.worker_id,
                backend=request.backend,
                status=WorkerStatus.AVAILABLE,
                hostname=request.capabilities.hostname,
                capabilities=request.capabilities.to_jsonable(),
            ),
        )
        worker.backend = request.backend
        worker.status = WorkerStatus.AVAILABLE
        worker.hostname = request.capabilities.hostname or worker.hostname
        worker.capabilities.update(request.capabilities.to_jsonable())
        worker.last_heartbeat_at = utc_now_iso()
        self._workers[worker.worker_id] = worker
        self._persist_worker(worker)

        for managed in self._jobs.values():
            job = managed.spec
            if not self._worker_can_claim_job(worker, job):
                continue
            managed.assigned_worker_id = worker.worker_id
            worker.status = WorkerStatus.BUSY
            job.status = JobStatus.ASSIGNED
            self._persist_worker(worker)
            self._persist_job(job.job_id)
            return ClaimJobResponse(job=job, status=ProtocolStatus.OK, message="job assigned")
        return ClaimJobResponse(status=ProtocolStatus.OK, message="no compatible queued job")

    def handle_progress_report(self, request: ProgressReportRequest) -> ProgressReportResponse:
        """Handle progress metrics from a worker.

        Args:
            request: Progress report request.

        Returns:
            Progress report response.
        """

        managed = self._jobs.get(request.job_id)
        if managed is None:
            return ProgressReportResponse(
                status=ProtocolStatus.REJECTED,
                should_stop_job=True,
                message="unknown job",
            )
        if managed.assigned_worker_id and managed.assigned_worker_id != request.worker_id:
            return ProgressReportResponse(
                status=ProtocolStatus.REJECTED,
                should_stop_job=True,
                message="job is assigned to a different worker",
            )
        managed.assigned_worker_id = request.worker_id
        managed.latest_metrics = request.metrics
        managed.updated_at = request.sent_at
        if managed.spec.status == JobStatus.ASSIGNED:
            managed.spec.status = JobStatus.RUNNING
        worker = self._workers.get(request.worker_id)
        if worker:
            worker.status = WorkerStatus.BUSY
            worker.last_heartbeat_at = request.sent_at
            self._persist_worker(worker)
        self._persist_job(request.job_id)
        should_stop = self._should_stop_remote_job(request.job_id)
        should_pause = self._should_pause_remote_job(request.job_id)
        return ProgressReportResponse(
            status=ProtocolStatus.OK,
            should_stop_job=should_stop,
            should_pause_job=should_pause,
            message=_control_message(should_stop, should_pause, "progress accepted"),
        )

    def handle_complete_job(self, request: CompleteJobRequest) -> CompleteJobResponse:
        """Handle successful remote job completion.

        Args:
            request: Complete job request.

        Returns:
            Complete job response.
        """

        if request.result is None:
            return CompleteJobResponse(status=ProtocolStatus.REJECTED, message="result is required")
        managed = self._jobs.get(request.result.job_id)
        if managed is None:
            return CompleteJobResponse(status=ProtocolStatus.REJECTED, message="unknown job")
        managed.assigned_worker_id = request.worker_id
        managed.result = request.result
        managed.spec.status = request.result.status
        managed.updated_at = request.sent_at
        managed.error = request.result.error
        worker = self._workers.get(request.worker_id)
        if worker:
            worker.status = WorkerStatus.AVAILABLE
            worker.last_heartbeat_at = request.sent_at
            self._persist_worker(worker)
        self._persist_job(request.result.job_id)
        self._stop_requested.discard(request.result.job_id)
        return CompleteJobResponse(status=ProtocolStatus.OK, message="job completion accepted")

    def handle_fail_job(self, request: FailJobRequest) -> FailJobResponse:
        """Handle remote job failure.

        Args:
            request: Fail job request.

        Returns:
            Fail job response.
        """

        managed = self._jobs.get(request.job_id)
        if managed is None:
            return FailJobResponse(status=ProtocolStatus.REJECTED, message="unknown job")
        managed.assigned_worker_id = None if request.retryable else request.worker_id
        managed.spec.status = JobStatus.QUEUED if request.retryable else JobStatus.FAILED
        managed.error = request.error
        managed.updated_at = request.sent_at
        worker = self._workers.get(request.worker_id)
        if worker:
            worker.status = WorkerStatus.AVAILABLE
            worker.last_heartbeat_at = request.sent_at
            self._persist_worker(worker)
        self._persist_job(request.job_id)
        self._stop_requested.discard(request.job_id)
        return FailJobResponse(
            status=ProtocolStatus.OK,
            message="job requeued after failure" if request.retryable else "job failure accepted",
        )

    def mark_stale_workers_offline(self, timeout_seconds: int = 30) -> list[str]:
        """Mark workers offline when their last heartbeat is too old.

        Args:
            timeout_seconds: Age in seconds after which a worker is stale.

        Returns:
            Worker IDs marked offline.
        """

        marked: list[str] = []
        cutoff = datetime.now().astimezone() - timedelta(seconds=timeout_seconds)
        for worker in self._workers.values():
            if worker.worker_id == "local" or worker.status == WorkerStatus.OFFLINE:
                continue
            heartbeat_at = _parse_timestamp(worker.last_heartbeat_at)
            if heartbeat_at and heartbeat_at < cutoff:
                worker.status = WorkerStatus.OFFLINE
                self._persist_worker(worker)
                marked.append(worker.worker_id)
        return marked

    def list_workers(self) -> list[WorkerDescriptor]:
        """Return known workers.

        Returns:
            Registered workers.
        """

        return list(self._workers.values())

    def submit(self, job: TrainingJobSpec) -> str:
        """Submit a job to the manager queue.

        Args:
            job: Training job contract.

        Returns:
            Job identifier.
        """

        job.status = JobStatus.QUEUED
        self._jobs[job.job_id] = ManagedJob(spec=job)
        self._persist_job(job.job_id)
        return job.job_id

    def get_job(self, job_id: str) -> ManagedJob:
        """Return a managed job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            Managed job.

        Raises:
            KeyError: If the job is unknown.
        """

        return self._jobs[job_id]

    def list_jobs(self) -> list[ManagedJob]:
        """Return tracked jobs.

        Returns:
            Managed jobs.
        """

        return list(self._jobs.values())

    def cancel(self, job_id: str) -> None:
        """Request cooperative cancellation for a job.

        Args:
            job_id: Job identifier.
        """

        managed = self.get_job(job_id)
        managed.spec.status = JobStatus.STOPPING
        self._stop_requested.add(job_id)
        self._persist_job(job_id)

    def stop_all_jobs(self) -> int:
        """Request cooperative stop for all active jobs.

        Returns:
            Number of jobs marked for stopping.
        """

        count = 0
        for managed in self._jobs.values():
            if managed.spec.status in {JobStatus.QUEUED, JobStatus.ASSIGNED, JobStatus.RUNNING, JobStatus.PAUSED}:
                managed.spec.status = JobStatus.STOPPING
                self._stop_requested.add(managed.spec.job_id)
                self._persist_job(managed.spec.job_id)
                count += 1
        return count

    def pause_all_jobs(self) -> int:
        """Pause queued or active remote jobs.

        Returns:
            Number of jobs marked paused.
        """

        count = 0
        for managed in self._jobs.values():
            if managed.spec.status in {JobStatus.QUEUED, JobStatus.ASSIGNED, JobStatus.RUNNING}:
                managed.spec.status = JobStatus.PAUSED
                self._persist_job(managed.spec.job_id)
                count += 1
        return count

    def resume_all_jobs(self) -> int:
        """Resume paused jobs by returning them to the queue.

        Returns:
            Number of jobs resumed.
        """

        count = 0
        for managed in self._jobs.values():
            if managed.spec.status == JobStatus.PAUSED:
                managed.spec.status = JobStatus.QUEUED
                managed.assigned_worker_id = None
                self._persist_job(managed.spec.job_id)
                count += 1
        return count

    def run_next(
        self,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> TrainingResult:
        """Run the next queued job.

        Args:
            progress: Optional progress callback.
            should_stop: Optional external stop callback.

        Returns:
            Training result.

        Raises:
            ValueError: If no queued job is available.
        """

        for managed in self._jobs.values():
            if managed.spec.status == JobStatus.QUEUED:
                return self.run_job(managed.spec.job_id, progress=progress, should_stop=should_stop)
        raise ValueError("No queued training jobs are available")

    def run_job(
        self,
        job_id: str,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> TrainingResult:
        """Run one job through an available backend.

        Args:
            job_id: Job identifier.
            progress: Optional progress callback.
            should_stop: Optional external stop callback.

        Returns:
            Training result.
        """

        managed = self.get_job(job_id)
        job = managed.spec
        worker = self._select_worker(job.runtime.backend)
        managed.assigned_worker_id = worker.worker_id
        worker.status = WorkerStatus.BUSY
        job.status = JobStatus.ASSIGNED
        self._persist_worker(worker)
        self._persist_job(job_id)
        backend = self.registry.get(job.runtime.backend)
        try:
            result = backend.run(
                job,
                progress=progress,
                should_stop=lambda: self._should_stop(job_id, should_stop),
            )
            managed.result = self._result_spec(job, result)
            job.status = managed.result.status
            self._persist_job(job_id)
            return result
        except Exception as exc:
            job.status = JobStatus.FAILED
            managed.error = str(exc)
            self._persist_job(job_id)
            raise
        finally:
            worker.status = WorkerStatus.AVAILABLE
            worker.last_heartbeat_at = utc_now_iso()
            self._persist_worker(worker)
            self._stop_requested.discard(job_id)

    def _select_worker(self, backend: BackendKind) -> WorkerDescriptor:
        """Select an available worker for a backend.

        Args:
            backend: Backend kind required by the job.

        Returns:
            Available worker.

        Raises:
            ValueError: If no worker is available.
        """

        for worker in self._workers.values():
            if worker.backend == backend and worker.status == WorkerStatus.AVAILABLE:
                return worker
        raise ValueError(f"No available worker for backend {backend.value}")

    def _worker_can_claim_job(self, worker: WorkerDescriptor, job: TrainingJobSpec) -> bool:
        """Return whether a remote worker can claim a job.

        Args:
            worker: Worker descriptor.
            job: Training job contract.

        Returns:
            Whether the worker can claim the job.
        """

        if job.status != JobStatus.QUEUED:
            return False
        if job.runtime.backend != worker.backend:
            return False
        if job.runtime.preferred_worker_id and job.runtime.preferred_worker_id != worker.worker_id:
            return False
        if job.runtime.min_vram_gb is not None:
            worker_vram = _worker_total_vram_gb(worker)
            if worker_vram is None or worker_vram < job.runtime.min_vram_gb:
                return False
        if job.runtime.tags:
            worker_labels = set(worker.capabilities.get("labels") or [])
            if not set(job.runtime.tags).issubset(worker_labels):
                return False
        return True

    def _should_stop(self, job_id: str, external_stop: Optional[StopCallback]) -> bool:
        """Return whether a job should stop.

        Args:
            job_id: Job identifier.
            external_stop: Optional external stop callback.

        Returns:
            Whether the job should stop.
        """

        return job_id in self._stop_requested or bool(external_stop and external_stop())

    def _should_stop_remote_job(self, job_id: str) -> bool:
        """Return whether a remote worker should stop a job.

        Args:
            job_id: Job identifier.

        Returns:
            Whether the worker should stop the job.
        """

        if job_id in self._stop_requested:
            return True
        managed = self._jobs.get(job_id)
        return bool(managed and managed.spec.status in {JobStatus.STOPPING, JobStatus.CANCELLED, JobStatus.FAILED})

    def _should_pause_remote_job(self, job_id: str) -> bool:
        """Return whether a remote worker should pause a job.

        Args:
            job_id: Job identifier.

        Returns:
            Whether the worker should pause the job.
        """

        managed = self._jobs.get(job_id)
        return bool(managed and managed.spec.status == JobStatus.PAUSED)

    def _result_spec(self, job: TrainingJobSpec, result: TrainingResult) -> TrainingResultSpec:
        """Create a serializable result contract from a training result.

        Args:
            job: Training job contract.
            result: Concrete training result.

        Returns:
            Serializable result specification.
        """

        return TrainingResultSpec(
            job_id=job.job_id,
            status=JobStatus.CANCELLED if result.stopped else JobStatus.COMPLETED,
            checkpoint_path=result.checkpoint_path,
            summary_path=result.summary_path,
            final_train_loss=result.final_train_loss,
            final_val_loss=result.final_val_loss,
            stopped=result.stopped,
        )

    def _restore_state(self) -> None:
        """Restore persisted jobs and workers from the state store."""

        for worker_data in self.state_store.load_workers():
            worker = WorkerDescriptor.from_jsonable(worker_data)
            if worker.status == WorkerStatus.BUSY:
                worker.status = WorkerStatus.OFFLINE
            self._workers[worker.worker_id] = worker
        for job_data in self.state_store.load_jobs():
            managed = _managed_job_from_jsonable(job_data)
            if managed.spec.status in {JobStatus.ASSIGNED, JobStatus.RUNNING, JobStatus.STOPPING}:
                managed.spec.status = JobStatus.QUEUED
                managed.assigned_worker_id = None
                managed.error = "Recovered after app restart before job completion."
            self._jobs[managed.spec.job_id] = managed
            self._persist_job(managed.spec.job_id)

    def _persist_job(self, job_id: str) -> None:
        """Persist a managed job.

        Args:
            job_id: Job identifier.
        """

        managed = self._jobs.get(job_id)
        if managed:
            self.state_store.save_job(job_id, managed.spec.status.value, managed.to_jsonable())

    def _persist_worker(self, worker: WorkerDescriptor) -> None:
        """Persist a worker descriptor.

        Args:
            worker: Worker descriptor.
        """

        self.state_store.save_worker(worker.worker_id, worker.status.value, worker.to_jsonable())


def run_local_job(
    job: TrainingJobSpec,
    backend: Optional[TrainerBackend] = None,
    progress: Optional[ProgressCallback] = None,
    should_stop: Optional[StopCallback] = None,
) -> TrainingResult:
    """Run a job through a temporary local manager.

    Args:
        job: Training job contract.
        backend: Optional backend override for tests or embedded use.
        progress: Optional progress callback.
        should_stop: Optional cooperative cancellation callback.

    Returns:
        Training result.
    """

    registry = BackendRegistry()
    if backend is not None:
        registry.register(job.runtime.backend, backend)
    manager = JobManager(registry=registry)
    manager.submit(job)
    return manager.run_job(job.job_id, progress=progress, should_stop=should_stop)


def _jsonable_result(result: TrainingResultSpec) -> dict[str, Any]:
    """Convert a result spec to JSON-friendly values.

    Args:
        result: Training result specification.

    Returns:
        Serializable dictionary.
    """

    output: dict[str, Any] = {}
    for key, value in result.__dict__.items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, Enum):
            output[key] = value.value
        else:
            output[key] = value
    return output


def _managed_job_from_jsonable(data: dict[str, Any]) -> ManagedJob:
    """Create a managed job from JSON-friendly values.

    Args:
        data: Serialized managed job payload.

    Returns:
        Managed job.
    """

    result_data = data.get("result")
    return ManagedJob(
        spec=TrainingJobSpec.from_jsonable(data["spec"]),
        assigned_worker_id=data.get("assigned_worker_id"),
        result=_result_from_jsonable(result_data) if result_data else None,
        error=data.get("error"),
        latest_metrics=TrainingMetrics(**dict(data.get("latest_metrics") or {})) if data.get("latest_metrics") else None,
        updated_at=data.get("updated_at", utc_now_iso()),
    )


def _result_from_jsonable(data: dict[str, Any]) -> TrainingResultSpec:
    """Create a training result spec from JSON-friendly values.

    Args:
        data: Serialized result data.

    Returns:
        Training result specification.
    """

    checkpoint_path = data.get("checkpoint_path")
    summary_path = data.get("summary_path")
    return TrainingResultSpec(
        job_id=data["job_id"],
        status=JobStatus(data["status"]),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        summary_path=Path(summary_path) if summary_path else None,
        final_train_loss=data.get("final_train_loss"),
        final_val_loss=data.get("final_val_loss"),
        stopped=bool(data.get("stopped")),
        error=data.get("error"),
        artifact_bundle_url=data.get("artifact_bundle_url"),
    )


def _availability_to_worker_status(availability: WorkerAvailability) -> WorkerStatus:
    """Convert protocol availability to manager worker status.

    Args:
        availability: Protocol worker availability.

    Returns:
        Manager worker status.
    """

    if availability == WorkerAvailability.BUSY:
        return WorkerStatus.BUSY
    if availability == WorkerAvailability.OFFLINE:
        return WorkerStatus.OFFLINE
    return WorkerStatus.AVAILABLE


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO timestamp.

    Args:
        value: ISO timestamp.

    Returns:
        Parsed timestamp when valid.
    """

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed


def _worker_total_vram_gb(worker: WorkerDescriptor) -> Optional[float]:
    """Return worker total VRAM in GB when known.

    Args:
        worker: Worker descriptor.

    Returns:
        Total VRAM in GB.
    """

    value = worker.capabilities.get("total_vram_gb")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _control_message(should_stop: bool, should_pause: bool, default: str) -> str:
    """Return a human-readable control message.

    Args:
        should_stop: Whether stop was requested.
        should_pause: Whether pause was requested.
        default: Default message.

    Returns:
        Control message.
    """

    if should_stop:
        return "stop requested"
    if should_pause:
        return "pause requested"
    return default
