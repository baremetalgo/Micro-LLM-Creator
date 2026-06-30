from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import ModelConfig, TrainingConfig
from .services import train_from_dataset
from .training import TrainingResult


ProgressCallback = Callable[[Any], None]
StopCallback = Callable[[], bool]


@dataclass
class TrainingJobRequest:
    """Request payload for a training service job.

    Args:
        dataset_dir: Prepared dataset directory.
        model_config: Model architecture settings.
        training_config: Training runtime and optimizer settings.
    """

    dataset_dir: Path
    model_config: ModelConfig
    training_config: TrainingConfig


class LocalTrainerService:
    """Local trainer service used by the desktop coordinator.

    This service is intentionally small: the GUI depends on this API boundary
    instead of calling the trainer directly, which leaves room for a future
    cloud or multi-machine implementation behind the same contract.
    """

    def run(
        self,
        request: TrainingJobRequest,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCallback] = None,
    ) -> TrainingResult:
        """Run one local training job.

        Args:
            request: Training job request.
            progress: Optional progress callback.
            should_stop: Optional cooperative cancellation callback.

        Returns:
            Training result produced by the local trainer.
        """

        return train_from_dataset(
            request.dataset_dir,
            request.model_config,
            request.training_config,
            progress=progress,
            should_stop=should_stop,
        )


def run_training_job(
    dataset_dir: Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    progress: Optional[ProgressCallback] = None,
    should_stop: Optional[StopCallback] = None,
) -> TrainingResult:
    """Run a training job through the configured training service.

    Args:
        dataset_dir: Prepared dataset directory.
        model_config: Model architecture settings.
        training_config: Training runtime and optimizer settings.
        progress: Optional progress callback.
        should_stop: Optional cooperative cancellation callback.

    Returns:
        Training result from the active trainer service.
    """

    service = LocalTrainerService()
    return service.run(
        TrainingJobRequest(dataset_dir, model_config, training_config),
        progress=progress,
        should_stop=should_stop,
    )
