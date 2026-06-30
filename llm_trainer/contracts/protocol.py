from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from llm_trainer.contracts.jobs import BackendKind, TrainingJobSpec, TrainingMetrics, TrainingResultSpec, utc_now_iso


class ProtocolMessageKind(str, Enum):
    """Coordinator protocol message kind."""

    REGISTER_WORKER_REQUEST = "register_worker_request"
    REGISTER_WORKER_RESPONSE = "register_worker_response"
    HEARTBEAT_REQUEST = "heartbeat_request"
    HEARTBEAT_RESPONSE = "heartbeat_response"
    CLAIM_JOB_REQUEST = "claim_job_request"
    CLAIM_JOB_RESPONSE = "claim_job_response"
    PROGRESS_REPORT_REQUEST = "progress_report_request"
    PROGRESS_REPORT_RESPONSE = "progress_report_response"
    COMPLETE_JOB_REQUEST = "complete_job_request"
    COMPLETE_JOB_RESPONSE = "complete_job_response"
    FAIL_JOB_REQUEST = "fail_job_request"
    FAIL_JOB_RESPONSE = "fail_job_response"


class ProtocolStatus(str, Enum):
    """Coordinator protocol response status."""

    OK = "ok"
    REJECTED = "rejected"
    ERROR = "error"


class WorkerAvailability(str, Enum):
    """Remote worker availability status."""

    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class ProtocolEnvelope:
    """Base metadata for coordinator protocol messages.

    Attributes:
        message_id: Unique protocol message identifier.
        kind: Protocol message kind.
        sent_at: UTC timestamp when the message was created.
        protocol_version: Protocol version string.
    """

    kind: ProtocolMessageKind
    message_id: str = field(default_factory=lambda: f"msg_{uuid4().hex}")
    sent_at: str = field(default_factory=utc_now_iso)
    protocol_version: str = "0.1"

    def envelope_json(self) -> dict[str, Any]:
        """Return JSON-friendly envelope fields.

        Returns:
            Serializable envelope dictionary.
        """

        return {
            "message_id": self.message_id,
            "kind": self.kind.value,
            "sent_at": self.sent_at,
            "protocol_version": self.protocol_version,
        }


@dataclass
class WorkerCapabilities:
    """Hardware and runtime capabilities advertised by a worker.

    Attributes:
        hostname: Optional host name.
        platform: Operating system or runtime platform label.
        cpu_count: Logical CPU count.
        system_ram_gb: System RAM in GB.
        gpu_names: GPU names visible to the worker.
        total_vram_gb: Total visible GPU VRAM in GB.
        supports_cuda: Whether CUDA is available.
        supports_bf16: Whether BF16 is available.
        supports_fp16: Whether FP16 is available.
        extra: Free-form implementation details.
    """

    hostname: Optional[str] = None
    platform: Optional[str] = None
    cpu_count: Optional[int] = None
    system_ram_gb: Optional[float] = None
    gpu_names: list[str] = field(default_factory=list)
    total_vram_gb: Optional[float] = None
    supports_cuda: bool = False
    supports_bf16: bool = False
    supports_fp16: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        """Convert capabilities to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "system_ram_gb": self.system_ram_gb,
            "gpu_names": self.gpu_names,
            "total_vram_gb": self.total_vram_gb,
            "supports_cuda": self.supports_cuda,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "extra": self.extra,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "WorkerCapabilities":
        """Create capabilities from JSON-friendly values.

        Args:
            data: Serialized capabilities.

        Returns:
            Worker capabilities.
        """

        return cls(
            hostname=data.get("hostname"),
            platform=data.get("platform"),
            cpu_count=data.get("cpu_count"),
            system_ram_gb=data.get("system_ram_gb"),
            gpu_names=list(data.get("gpu_names") or []),
            total_vram_gb=data.get("total_vram_gb"),
            supports_cuda=bool(data.get("supports_cuda")),
            supports_bf16=bool(data.get("supports_bf16")),
            supports_fp16=bool(data.get("supports_fp16")),
            extra=dict(data.get("extra") or {}),
        )


@dataclass
class RegisterWorkerRequest(ProtocolEnvelope):
    """Request sent by a worker to join the coordinator."""

    worker_id: str = ""
    backend: BackendKind = BackendKind.REMOTE_CLIENT
    device: str = "auto"
    capabilities: WorkerCapabilities = field(default_factory=WorkerCapabilities)
    labels: list[str] = field(default_factory=list)

    def __init__(
        self,
        worker_id: str,
        backend: BackendKind = BackendKind.REMOTE_CLIENT,
        device: str = "auto",
        capabilities: Optional[WorkerCapabilities] = None,
        labels: Optional[list[str]] = None,
    ) -> None:
        """Create a register worker request.

        Args:
            worker_id: Worker identifier.
            backend: Backend kind the worker can run.
            device: Preferred runtime device.
            capabilities: Worker hardware/runtime capabilities.
            labels: Free-form worker labels for scheduling.
        """

        super().__init__(ProtocolMessageKind.REGISTER_WORKER_REQUEST)
        self.worker_id = worker_id
        self.backend = backend
        self.device = device
        self.capabilities = capabilities or WorkerCapabilities()
        self.labels = labels or []

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "backend": self.backend.value,
            "device": self.device,
            "capabilities": self.capabilities.to_jsonable(),
            "labels": self.labels,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "RegisterWorkerRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Register worker request.
        """

        request = cls(
            worker_id=data["worker_id"],
            backend=BackendKind(data.get("backend", BackendKind.REMOTE_CLIENT.value)),
            device=data.get("device", "auto"),
            capabilities=WorkerCapabilities.from_jsonable(data.get("capabilities") or {}),
            labels=list(data.get("labels") or []),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class RegisterWorkerResponse(ProtocolEnvelope):
    """Response returned after worker registration."""

    status: ProtocolStatus = ProtocolStatus.OK
    worker_id: str = ""
    accepted: bool = True
    heartbeat_interval_seconds: int = 10
    message: str = ""

    def __init__(
        self,
        worker_id: str,
        accepted: bool = True,
        status: ProtocolStatus = ProtocolStatus.OK,
        heartbeat_interval_seconds: int = 10,
        message: str = "",
    ) -> None:
        """Create a register worker response.

        Args:
            worker_id: Worker identifier.
            accepted: Whether registration was accepted.
            status: Protocol response status.
            heartbeat_interval_seconds: Requested heartbeat interval.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.REGISTER_WORKER_RESPONSE)
        self.status = status
        self.worker_id = worker_id
        self.accepted = accepted
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "worker_id": self.worker_id,
            "accepted": self.accepted,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "message": self.message,
        }


@dataclass
class HeartbeatRequest(ProtocolEnvelope):
    """Heartbeat request sent by a worker."""

    worker_id: str = ""
    availability: WorkerAvailability = WorkerAvailability.AVAILABLE
    backend: BackendKind = BackendKind.REMOTE_CLIENT
    active_job_id: Optional[str] = None
    device: str = "auto"
    metrics: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        worker_id: str,
        availability: WorkerAvailability = WorkerAvailability.AVAILABLE,
        backend: BackendKind = BackendKind.REMOTE_CLIENT,
        active_job_id: Optional[str] = None,
        device: str = "auto",
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create a heartbeat request.

        Args:
            worker_id: Worker identifier.
            availability: Current worker availability.
            backend: Backend kind the worker can run.
            active_job_id: Active job identifier when busy.
            device: Runtime device.
            metrics: Worker metrics.
        """

        super().__init__(ProtocolMessageKind.HEARTBEAT_REQUEST)
        self.worker_id = worker_id
        self.availability = availability
        self.backend = backend
        self.active_job_id = active_job_id
        self.device = device
        self.metrics = metrics or {}

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "availability": self.availability.value,
            "backend": self.backend.value,
            "active_job_id": self.active_job_id,
            "device": self.device,
            "metrics": self.metrics,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "HeartbeatRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Heartbeat request.
        """

        request = cls(
            worker_id=data["worker_id"],
            availability=WorkerAvailability(data.get("availability", WorkerAvailability.AVAILABLE.value)),
            backend=BackendKind(data.get("backend", BackendKind.REMOTE_CLIENT.value)),
            active_job_id=data.get("active_job_id"),
            device=data.get("device", "auto"),
            metrics=dict(data.get("metrics") or {}),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class HeartbeatResponse(ProtocolEnvelope):
    """Response returned after a worker heartbeat."""

    status: ProtocolStatus = ProtocolStatus.OK
    should_stop_job: bool = False
    should_pause_job: bool = False
    message: str = ""

    def __init__(
        self,
        status: ProtocolStatus = ProtocolStatus.OK,
        should_stop_job: bool = False,
        should_pause_job: bool = False,
        message: str = "",
    ) -> None:
        """Create a heartbeat response.

        Args:
            status: Protocol response status.
            should_stop_job: Whether active job should stop.
            should_pause_job: Whether active job should pause.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.HEARTBEAT_RESPONSE)
        self.status = status
        self.should_stop_job = should_stop_job
        self.should_pause_job = should_pause_job
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "should_stop_job": self.should_stop_job,
            "should_pause_job": self.should_pause_job,
            "message": self.message,
        }


@dataclass
class ClaimJobRequest(ProtocolEnvelope):
    """Request sent by a worker asking for a job."""

    worker_id: str = ""
    backend: BackendKind = BackendKind.REMOTE_CLIENT
    capabilities: WorkerCapabilities = field(default_factory=WorkerCapabilities)

    def __init__(
        self,
        worker_id: str,
        backend: BackendKind = BackendKind.REMOTE_CLIENT,
        capabilities: Optional[WorkerCapabilities] = None,
    ) -> None:
        """Create a claim job request.

        Args:
            worker_id: Worker identifier.
            backend: Backend kind requested by the worker.
            capabilities: Latest worker capabilities.
        """

        super().__init__(ProtocolMessageKind.CLAIM_JOB_REQUEST)
        self.worker_id = worker_id
        self.backend = backend
        self.capabilities = capabilities or WorkerCapabilities()

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "backend": self.backend.value,
            "capabilities": self.capabilities.to_jsonable(),
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "ClaimJobRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Claim job request.
        """

        request = cls(
            worker_id=data["worker_id"],
            backend=BackendKind(data.get("backend", BackendKind.REMOTE_CLIENT.value)),
            capabilities=WorkerCapabilities.from_jsonable(data.get("capabilities") or {}),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class ClaimJobResponse(ProtocolEnvelope):
    """Response with an assigned job or empty assignment."""

    status: ProtocolStatus = ProtocolStatus.OK
    job: Optional[TrainingJobSpec] = None
    message: str = ""

    def __init__(
        self,
        job: Optional[TrainingJobSpec] = None,
        status: ProtocolStatus = ProtocolStatus.OK,
        message: str = "",
    ) -> None:
        """Create a claim job response.

        Args:
            job: Assigned job contract when available.
            status: Protocol response status.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.CLAIM_JOB_RESPONSE)
        self.status = status
        self.job = job
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "job": self.job.to_jsonable() if self.job else None,
            "message": self.message,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "ClaimJobResponse":
        """Create a response from JSON-friendly values.

        Args:
            data: Serialized response.

        Returns:
            Claim job response.
        """

        job_data = data.get("job")
        response = cls(
            job=TrainingJobSpec.from_jsonable(job_data) if job_data else None,
            status=ProtocolStatus(data.get("status", ProtocolStatus.OK.value)),
            message=data.get("message", ""),
        )
        _restore_envelope(response, data)
        return response


@dataclass
class ProgressReportRequest(ProtocolEnvelope):
    """Progress update sent by a worker for a running job."""

    worker_id: str = ""
    job_id: str = ""
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)

    def __init__(self, worker_id: str, job_id: str, metrics: Optional[TrainingMetrics] = None) -> None:
        """Create a progress report request.

        Args:
            worker_id: Worker identifier.
            job_id: Job identifier.
            metrics: Training metrics.
        """

        super().__init__(ProtocolMessageKind.PROGRESS_REPORT_REQUEST)
        self.worker_id = worker_id
        self.job_id = job_id
        self.metrics = metrics or TrainingMetrics()

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "metrics": self.metrics.__dict__,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "ProgressReportRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Progress report request.
        """

        request = cls(
            worker_id=data["worker_id"],
            job_id=data["job_id"],
            metrics=TrainingMetrics(**dict(data.get("metrics") or {})),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class ProgressReportResponse(ProtocolEnvelope):
    """Response returned after a progress report."""

    status: ProtocolStatus = ProtocolStatus.OK
    should_stop_job: bool = False
    should_pause_job: bool = False
    message: str = ""

    def __init__(
        self,
        status: ProtocolStatus = ProtocolStatus.OK,
        should_stop_job: bool = False,
        should_pause_job: bool = False,
        message: str = "",
    ) -> None:
        """Create a progress report response.

        Args:
            status: Protocol response status.
            should_stop_job: Whether the worker should stop the active job.
            should_pause_job: Whether the worker should pause the active job.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.PROGRESS_REPORT_RESPONSE)
        self.status = status
        self.should_stop_job = should_stop_job
        self.should_pause_job = should_pause_job
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "should_stop_job": self.should_stop_job,
            "should_pause_job": self.should_pause_job,
            "message": self.message,
        }


@dataclass
class CompleteJobRequest(ProtocolEnvelope):
    """Completion report sent by a worker."""

    worker_id: str = ""
    result: Optional[TrainingResultSpec] = None

    def __init__(self, worker_id: str, result: TrainingResultSpec) -> None:
        """Create a complete job request.

        Args:
            worker_id: Worker identifier.
            result: Training result specification.
        """

        super().__init__(ProtocolMessageKind.COMPLETE_JOB_REQUEST)
        self.worker_id = worker_id
        self.result = result

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "result": _result_to_jsonable(self.result) if self.result else None,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "CompleteJobRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Complete job request.
        """

        request = cls(
            worker_id=data["worker_id"],
            result=_result_from_jsonable(dict(data["result"])),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class CompleteJobResponse(ProtocolEnvelope):
    """Response returned after job completion."""

    status: ProtocolStatus = ProtocolStatus.OK
    message: str = ""

    def __init__(self, status: ProtocolStatus = ProtocolStatus.OK, message: str = "") -> None:
        """Create a complete job response.

        Args:
            status: Protocol response status.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.COMPLETE_JOB_RESPONSE)
        self.status = status
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "message": self.message,
        }


@dataclass
class FailJobRequest(ProtocolEnvelope):
    """Failure report sent by a worker."""

    worker_id: str = ""
    job_id: str = ""
    error: str = ""
    retryable: bool = False

    def __init__(self, worker_id: str, job_id: str, error: str, retryable: bool = False) -> None:
        """Create a fail job request.

        Args:
            worker_id: Worker identifier.
            job_id: Job identifier.
            error: Failure text.
            retryable: Whether the coordinator may retry the job.
        """

        super().__init__(ProtocolMessageKind.FAIL_JOB_REQUEST)
        self.worker_id = worker_id
        self.job_id = job_id
        self.error = error
        self.retryable = retryable

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the request to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "worker_id": self.worker_id,
            "job_id": self.job_id,
            "error": self.error,
            "retryable": self.retryable,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> "FailJobRequest":
        """Create a request from JSON-friendly values.

        Args:
            data: Serialized request.

        Returns:
            Fail job request.
        """

        request = cls(
            worker_id=data["worker_id"],
            job_id=data["job_id"],
            error=data.get("error", ""),
            retryable=bool(data.get("retryable")),
        )
        _restore_envelope(request, data)
        return request


@dataclass
class FailJobResponse(ProtocolEnvelope):
    """Response returned after job failure report."""

    status: ProtocolStatus = ProtocolStatus.OK
    message: str = ""

    def __init__(self, status: ProtocolStatus = ProtocolStatus.OK, message: str = "") -> None:
        """Create a fail job response.

        Args:
            status: Protocol response status.
            message: Human-readable response message.
        """

        super().__init__(ProtocolMessageKind.FAIL_JOB_RESPONSE)
        self.status = status
        self.message = message

    def to_jsonable(self) -> dict[str, Any]:
        """Convert the response to JSON-friendly values.

        Returns:
            Serializable dictionary.
        """

        return {
            **self.envelope_json(),
            "status": self.status.value,
            "message": self.message,
        }


def _restore_envelope(message: ProtocolEnvelope, data: dict[str, Any]) -> None:
    """Restore envelope fields on a protocol message.

    Args:
        message: Protocol message.
        data: Serialized message data.
    """

    message.message_id = data.get("message_id", message.message_id)
    message.sent_at = data.get("sent_at", message.sent_at)
    message.protocol_version = data.get("protocol_version", message.protocol_version)


def _result_to_jsonable(result: TrainingResultSpec) -> dict[str, Any]:
    """Convert a result spec to JSON-friendly values.

    Args:
        result: Training result specification.

    Returns:
        Serializable dictionary.
    """

    output: dict[str, Any] = {}
    for key, value in result.__dict__.items():
        if hasattr(value, "value"):
            output[key] = value.value
        elif value is None:
            output[key] = None
        else:
            output[key] = str(value) if key.endswith("_path") else value
    return output


def _result_from_jsonable(data: dict[str, Any]) -> TrainingResultSpec:
    """Create a result spec from JSON-friendly values.

    Args:
        data: Serialized result.

    Returns:
        Training result specification.
    """

    from pathlib import Path
    from llm_trainer.contracts.jobs import JobStatus

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
