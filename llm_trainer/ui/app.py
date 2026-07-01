from __future__ import annotations

import ctypes
from datetime import datetime
import json
import logging
import os
from queue import Empty, Queue
import re
import shutil
import signal
import sqlite3
import sys
from pathlib import Path
from threading import Event, Thread
from typing import Any, Optional, Union

import torch
from PySide6.QtCore import QPoint, Qt, QThread, QTimer, Slot, qInstallMessageHandler
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from llm_trainer.app_logging import qt_message_handler, setup_logging
from llm_trainer.config import DatasetConfig, ModelConfig, TrainingConfig
from llm_trainer.conversation_datasets import CONVERSATION_DATASET_PRESETS, dataset_ids_for_stage, dataset_stage_label
from llm_trainer.contracts import BackendKind
from llm_trainer.contracts.jobs import RuntimeSpec, TrainingJobSpec
from llm_trainer.coordinator import CoordinatorApiServer, JobManager, create_job_artifact_bundle
from llm_trainer.evaluation import DEFAULT_BENCHMARK_PROMPTS, evaluate_checkpoint, normalize_prompts
from llm_trainer.export import export_gguf_with_llama_cpp, export_hf_microgpt_package, export_project_bundle, quantize_checkpoint
from llm_trainer.fine_tuning_service import run_fine_tuning_job
from llm_trainer.llama_chat import LlamaChatSession, load_llama_chat_session, stream_chat_reply
from llm_trainer.microgpt_chat import load_microgpt_chat_session, stream_microgpt_chat_reply
from llm_trainer.services import build_dataset, check_project_health, scan_dataset_preview
from llm_trainer.telemetry_store import initialize_store, insert_metric, latest_run, rows_until, telemetry_db_path
from llm_trainer.training import check_resume_compatibility, latest_checkpoint
from llm_trainer.training_planning import estimate_training_resources, format_bytes
from llm_trainer.training_service import run_training_job
from llm_trainer.ui.chat_widgets import ChatMessageWidget
from llm_trainer.ui.markdown_renderer import markdown_to_html
from llm_trainer.ui.workers import TaskWorker
from llm_trainer.ui.tabs.benchmark_tab import build_benchmark_tab
from llm_trainer.ui.tabs.chat_tab import build_chat_tab
from llm_trainer.ui.tabs.dataset_tab import build_dataset_tab
from llm_trainer.ui.tabs.live_tab import build_live_training_tab
from llm_trainer.ui.tabs.training_tab import build_training_tab
from llm_trainer.ui.tabs.export_tab import build_export_tab
from llm_trainer.ui.tabs.fine_tuning_tab import build_fine_tuning_tab
from llm_trainer.ui.tabs.job_manager_tab import build_job_manager_tab, set_table_rows

try:
    import psutil
except ImportError:
    psutil = None


WINDOWS_APP_ID = "MicroLLMCreator.Lightning"
LOGGER = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main PySide6 window for Micro LLM Creator."""

    def __init__(self) -> None:
        """Create the main application window."""

        super().__init__()
        self.log_file_path = setup_logging()
        LOGGER.info("Creating Micro LLM Creator main window")
        if QApplication.instance():
            QApplication.instance().setFont(QFont("Arial", 10))
        self.setWindowTitle("Micro LLM Creator")
        self.setWindowIcon(self._lightning_icon())
        self._windows_icon_handles: list[int] = []
        self.resize(1240, 820)
        self.thread: Optional[QThread] = None
        self.worker: Optional[TaskWorker] = None
        self.stop_event: Optional[Event] = None
        self.progress_queue: Optional[Queue] = None
        self.active_log: Optional[QTextEdit] = None
        self.active_progress_bar: Optional[QProgressBar] = None
        self.active_button: Optional[QPushButton] = None
        self.active_stop_button: Optional[QPushButton] = None
        self.active_button_text = ""
        self.active_button_restore_text = ""
        self.current_project_file: Optional[Path] = None
        self.telemetry_db_path: Optional[Path] = None
        self.telemetry_run_id = ""
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_scrub_active = False
        self.hardware_meter_labels: dict[int, QLabel] = {}
        self.training_cards: list[QWidget] = []
        self.training_controls_grid: Optional[QGridLayout] = None
        self.training_controls_columns = 3
        self.active_training_log: Optional[QTextEdit] = None
        self.active_training_progress: Optional[QProgressBar] = None
        self.active_training_final_button_text = "Start Training"
        self.interrupt_count = 0
        self.chat_session: Optional[LlamaChatSession] = None
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self.current_assistant_browser: Optional[QTextBrowser] = None
        self.current_assistant_meta: Optional[QLabel] = None
        self.current_assistant_message: Optional[ChatMessageWidget] = None
        self.pending_user_message = ""
        self.spinner_index = 0
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self._tick_spinner)
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._drain_progress_queue)
        self.job_manager = JobManager()
        self.coordinator_server: Optional[CoordinatorApiServer] = None
        self.coordinator_thread: Optional[Thread] = None
        self.job_manager_timer = QTimer(self)
        self.job_manager_timer.setInterval(2500)
        self.job_manager_timer.timeout.connect(self.refresh_job_manager_tab)

        self._apply_style()

        shell = self._build_shell()
        self.setCentralWidget(shell)
        self._install_ui_event_logging(shell)
        self.job_manager_timer.start()

    def _install_ui_event_logging(self, root: QWidget) -> None:
        """Log user-facing widget actions and parameter changes.

        Args:
            root: Root widget to scan for child controls.
        """

        for widget in root.findChildren(QWidget):
            if isinstance(widget, QAbstractButton):
                if widget.isCheckable():
                    widget.toggled.connect(
                        lambda checked, item=widget: self._log_ui_event("toggled", item, checked)
                    )
                else:
                    widget.clicked.connect(
                        lambda checked=False, item=widget: self._log_ui_event("clicked", item, checked)
                    )
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(
                    lambda value, item=widget: self._log_ui_event("changed", item, value)
                )
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(
                    lambda value, item=widget: self._log_ui_event("changed", item, value)
                )
            elif isinstance(widget, QDoubleSpinBox):
                widget.valueChanged.connect(
                    lambda value, item=widget: self._log_ui_event("changed", item, value)
                )
            elif isinstance(widget, QLineEdit):
                widget.editingFinished.connect(
                    lambda item=widget: self._log_ui_event("edited", item, item.text())
                )

    def _log_ui_event(self, action: str, widget: QWidget, value: Any) -> None:
        """Log a UI action or parameter value.

        Args:
            action: Event label.
            widget: Widget that emitted the event.
            value: Current value.
        """

        LOGGER.info("UI %s: %s = %s", action, self._widget_log_name(widget), value)

    @staticmethod
    def _widget_log_name(widget: QWidget) -> str:
        """Return a useful log label for a widget.

        Args:
            widget: Widget to describe.

        Returns:
            Human-readable widget label.
        """

        if isinstance(widget, QAbstractButton) and widget.text():
            return widget.text().replace("\n", " ")
        if isinstance(widget, QLineEdit) and widget.placeholderText():
            return widget.placeholderText()
        if widget.objectName():
            return widget.objectName()
        return widget.__class__.__name__

    def _apply_style(self) -> None:
        """Load the application stylesheet from the QSS module file."""

        qss_path = Path(__file__).with_name("styles.qss")
        self.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    def _build_shell(self) -> QWidget:
        """Build the top-level dashboard shell.

        Returns:
            Root shell widget.
        """

        shell = QWidget()
        shell.setObjectName("AppShell")
        root = QVBoxLayout(shell)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        top = QWidget()
        top.setObjectName("TopBar")
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(16, 8, 16, 8)
        top_layout.setSpacing(8)
        logo = QLabel("ML")
        logo.setObjectName("Logo")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Project name...")
        self.search_box.setMaximumWidth(260)
        self._tip(self.search_box, "Project name used when saving or reopening a Micro LLM Creator project.")
        self.new_project_button = QPushButton("New Project")
        self.new_project_button.setMaximumWidth(130)
        self.new_project_button.clicked.connect(self.new_project)
        self._tip(self.new_project_button, "Start a fresh Micro LLM Creator project with default paths and settings.")
        self.save_project_button = QPushButton("Save Project")
        self.save_project_button.setMaximumWidth(130)
        self.save_project_button.clicked.connect(self.save_project)
        self._tip(self.save_project_button, "Save all current paths and settings into a project.json file.")
        self.open_project_button = QPushButton("Open Project")
        self.open_project_button.setMaximumWidth(130)
        self.open_project_button.clicked.connect(self.open_project)
        self._tip(self.open_project_button, "Open a saved project.json file and restore the UI settings.")
        self.dataset_status = QLabel("Dataset: not prepared")
        self.train_status = QLabel("Training: idle")
        self.export_status = QLabel("Export: waiting")
        self.chat_status = QLabel("Chat: no model loaded")
        for label in (self.dataset_status, self.train_status, self.export_status, self.chat_status):
            label.setObjectName("TopStatus")
            label.setMinimumWidth(0)
            label.setMaximumWidth(180)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            label.setWordWrap(False)
        self.project_state = QLabel("Ready")
        self.project_state.setObjectName("Metric")
        self.project_state.setMinimumWidth(0)
        self.project_state.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        top_layout.addWidget(logo)
        top_layout.addSpacing(12)
        top_layout.addWidget(self.search_box)
        top_layout.addWidget(self.new_project_button)
        top_layout.addWidget(self.save_project_button)
        top_layout.addWidget(self.open_project_button)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.dataset_status)
        top_layout.addWidget(self.train_status)
        top_layout.addWidget(self.export_status)
        top_layout.addWidget(self.chat_status)
        top_layout.addStretch(1)
        top_layout.addWidget(self.project_state)
        root.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        rail = QWidget()
        rail.setObjectName("SideRail")
        rail.setFixedWidth(82)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(12, 18, 12, 18)
        rail_layout.setSpacing(12)
        self.dataset_nav = self._nav_button("⇩\nIN")
        self.training_nav = self._nav_button("✦\nAI")
        self.training_nav.setText("AI")
        self.fine_tune_nav = self._nav_button("FT")
        self.live_nav = self._nav_button("LIVE")
        self.jobs_nav = self._nav_button("JOB")
        self.benchmark_nav = self._nav_button("◷\nBench")
        self.export_nav = self._nav_button("⇧\nX")
        self.chat_nav = self._nav_button("◌\nChat")
        self._tip(self.dataset_nav, "Open dataset preparation: load text/PDF files and build tokenizer data.")
        self._tip(self.training_nav, "Open model training: configure architecture and optimization settings.")
        self._tip(self.fine_tune_nav, "Open fine-tuning: adapt checkpoints with instruction, conversation, or LoRA settings.")
        self._tip(self.live_nav, "Open the live training tracker with model flow, charts, metrics, and telemetry.")
        self._tip(self.jobs_nav, "Open Job Manager: monitor workers, remote connections, assignments, and job controls.")
        self._tip(self.benchmark_nav, "Open benchmark prompts: test checkpoint quality with repeatable prompts.")
        self._tip(self.export_nav, "Open export tools: bundle or quantize the trained model artifacts.")
        self._tip(self.chat_nav, "Open Chat: load a GGUF or native MicroGPT model once and send prompts.")
        self.dataset_nav.setChecked(True)
        self.dataset_nav.clicked.connect(lambda: self._switch_page(0))
        self.training_nav.clicked.connect(lambda: self._switch_page(1))
        self.fine_tune_nav.clicked.connect(lambda: self._switch_page(2))
        self.live_nav.clicked.connect(lambda: self._switch_page(3))
        self.jobs_nav.clicked.connect(lambda: self._switch_page(4))
        self.benchmark_nav.clicked.connect(lambda: self._switch_page(5))
        self.export_nav.clicked.connect(lambda: self._switch_page(6))
        self.chat_nav.clicked.connect(lambda: self._switch_page(7))
        rail_layout.addWidget(self.dataset_nav)
        rail_layout.addWidget(self.training_nav)
        rail_layout.addWidget(self.fine_tune_nav)
        rail_layout.addWidget(self.live_nav)
        rail_layout.addWidget(self.jobs_nav)
        rail_layout.addWidget(self.benchmark_nav)
        rail_layout.addWidget(self.export_nav)
        rail_layout.addWidget(self.chat_nav)
        rail_layout.addStretch(1)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_dataset_tab())
        self.pages.addWidget(self._build_training_tab())
        self.pages.addWidget(self._build_fine_tuning_tab())
        self.pages.addWidget(self._build_live_training_tab())
        self.pages.addWidget(self._build_job_manager_tab())
        self.pages.addWidget(self._build_benchmark_tab())
        self.pages.addWidget(self._build_export_tab())
        self.pages.addWidget(self._build_chat_tab())

        body.addWidget(rail)
        body.addWidget(self.pages, 1)
        root.addLayout(body, 1)
        return shell

    def _nav_button(self, text: str) -> QPushButton:
        """Create a left-rail navigation button.

        Args:
            text: Button label.

        Returns:
            Configured navigation button.
        """

        button = QPushButton(text)
        button.setObjectName("NavButton")
        button.setCheckable(True)
        return button

    def _switch_page(self, index: int) -> None:
        """Switch the visible page.

        Args:
            index: Page index in the stacked widget.
        """

        self.pages.setCurrentIndex(index)
        buttons = [
            self.dataset_nav,
            self.training_nav,
            self.fine_tune_nav,
            self.live_nav,
            self.jobs_nav,
            self.benchmark_nav,
            self.export_nav,
            self.chat_nav,
        ]
        for button_index, button in enumerate(buttons):
            button.setChecked(button_index == index)
        self._refresh_training_layout()
        if index == 4:
            self.refresh_job_manager_tab()

    def resizeEvent(self, event: Any) -> None:
        """Refresh responsive layouts when the main window changes size.

        Args:
            event: Qt resize event.
        """

        super().resizeEvent(event)
        self._refresh_training_layout()

    def _refresh_training_layout(self) -> None:
        """Apply responsive card columns on the training page."""

        if not self.training_cards or self.training_controls_grid is None:
            return
        width = self.pages.width() if hasattr(self, "pages") else self.width()
        if width >= 900:
            columns = 2
        else:
            columns = 1
        if columns == self.training_controls_columns:
            return
        self._set_training_card_columns(columns)

    def _set_training_card_columns(self, columns: int) -> None:
        """Reflow the training cards into the requested column count.

        Args:
            columns: Number of columns to use.
        """

        if self.training_controls_grid is None:
            return
        while self.training_controls_grid.count():
            self.training_controls_grid.takeAt(0)
        for index, card in enumerate(self.training_cards):
            row = index // columns
            column = index % columns
            self.training_controls_grid.addWidget(card, row, column)
        for column in range(2):
            self.training_controls_grid.setColumnStretch(column, 1 if column < columns else 0)
        self.training_controls_columns = columns

    def _build_dataset_tab(self) -> QWidget:
        """Build the dataset preparation page.

        Returns:
            Dataset page widget.
        """

        return build_dataset_tab(self)

    def _build_training_tab(self) -> QWidget:
        """Build the training configuration page.

        Returns:
            Training page widget.
        """

        return build_training_tab(self)

    def _build_fine_tuning_tab(self) -> QWidget:
        """Build the fine-tuning page.

        Returns:
            Fine-tuning page widget.
        """

        return build_fine_tuning_tab(self)

    def _build_live_training_tab(self) -> QWidget:
        """Build the live training tracker page.

        Returns:
            Live training tracker page widget.
        """

        return build_live_training_tab(self)

    def _build_job_manager_tab(self) -> QWidget:
        """Build the distributed job manager page.

        Returns:
            Job manager page widget.
        """

        return build_job_manager_tab(self)

    def refresh_job_manager_tab(self) -> None:
        """Refresh the job manager dashboard tables."""

        if not hasattr(self, "job_worker_table"):
            return
        workers = self.job_manager.list_workers()
        jobs = self.job_manager.list_jobs()
        heartbeats = self.job_manager.state_store.latest_heartbeats()
        worker_rows = []
        for worker in workers:
            heartbeat = heartbeats.get(worker.worker_id, {})
            metrics = heartbeat.get("metrics") or {}
            active_job = heartbeat.get("active_job_id") or self._active_job_for_worker(worker.worker_id)
            capabilities = worker.capabilities or {}
            cpu_ram_gpu = (
                f"CPU {capabilities.get('cpu_count', '-')}, "
                f"RAM {capabilities.get('system_ram_gb', '-')} GB, "
                f"VRAM {capabilities.get('total_vram_gb', '-')} GB"
            )
            if metrics:
                cpu_ram_gpu = f"{cpu_ram_gpu}, util {metrics.get('gpu_util', metrics.get('gpu_memory_percent', '-'))}"
            worker_rows.append(
                [
                    worker.worker_id,
                    worker.status.value,
                    worker.backend.value,
                    worker.device,
                    worker.last_heartbeat_at or "-",
                    active_job or "-",
                    cpu_ram_gpu,
                    ", ".join(capabilities.get("labels") or []) or "-",
                ]
            )
        set_table_rows(self.job_worker_table, worker_rows)

        job_rows = []
        for managed in jobs:
            job = managed.spec
            metrics = managed.latest_metrics
            job_rows.append(
                [
                    job.job_id,
                    job.status.value,
                    managed.assigned_worker_id or "-",
                    job.runtime.backend.value,
                    self._metric_pair(metrics.epoch if metrics else None, metrics.total_epochs if metrics else None),
                    self._metric_pair(metrics.step if metrics else None, metrics.total_steps if metrics else None),
                    str(job.training.batch_size),
                    str(job.model.config.layer_count),
                    self._metric_float(metrics.train_loss if metrics else None),
                    self._metric_float(metrics.tokens_per_second if metrics else None, suffix=" tok/s"),
                    managed.updated_at,
                ]
            )
        set_table_rows(self.job_table, job_rows)
        active_count = sum(1 for item in jobs if item.spec.status.value in {"assigned", "running", "paused", "stopping"})
        queued_count = sum(1 for item in jobs if item.spec.status.value == "queued")
        self.job_worker_count_label.setText(f"Workers: {len(workers)}")
        self.job_active_count_label.setText(f"Active jobs: {active_count}")
        self.job_queue_count_label.setText(f"Queued jobs: {queued_count}")
        self.job_db_label.setText(f"State DB: {self.job_manager.state_store.db_path}")
        self.job_manager_progress.setValue(100)

    def pause_all_managed_jobs(self) -> None:
        """Pause all managed jobs."""

        count = self.job_manager.pause_all_jobs()
        self.job_manager_log.append(f"Pause requested for {count} job(s).")
        self.refresh_job_manager_tab()

    def resume_all_managed_jobs(self) -> None:
        """Resume all paused managed jobs."""

        count = self.job_manager.resume_all_jobs()
        self.job_manager_log.append(f"Resumed {count} job(s).")
        self.refresh_job_manager_tab()

    def stop_all_managed_jobs(self) -> None:
        """Stop all managed jobs."""

        count = self.job_manager.stop_all_jobs()
        self.job_manager_log.append(f"Stop requested for {count} job(s).")
        self.refresh_job_manager_tab()

    def mark_stale_workers_offline(self) -> None:
        """Mark stale remote workers offline."""

        workers = self.job_manager.mark_stale_workers_offline()
        if workers:
            self.job_manager_log.append(f"Marked offline: {', '.join(workers)}")
        else:
            self.job_manager_log.append("No stale remote workers found.")
        self.refresh_job_manager_tab()

    def start_coordinator_server(self) -> None:
        """Start the coordinator API used by remote workers."""

        if self.coordinator_server is not None:
            self.job_manager_log.append("Coordinator API is already running.")
            return
        host = self.coordinator_host.text().strip() or "0.0.0.0"
        port = self.coordinator_port.value()
        artifact_root = Path(self.coordinator_artifact_root.text().strip()).expanduser()
        artifact_root.mkdir(parents=True, exist_ok=True)
        try:
            self.coordinator_server = CoordinatorApiServer(
                manager=self.job_manager,
                host=host,
                port=port,
                artifact_root=artifact_root,
            )
            self.coordinator_thread = Thread(target=self.coordinator_server.serve_forever, daemon=True)
            self.coordinator_thread.start()
        except Exception as exc:
            self.coordinator_server = None
            self.coordinator_thread = None
            QMessageBox.warning(self, "Coordinator failed", f"Could not start coordinator API:\n{exc}")
            return
        public_url = self.coordinator_public_url.text().strip() or f"http://127.0.0.1:{port}"
        self.coordinator_public_url.setText(public_url.rstrip("/"))
        self.coordinator_status_label.setText(f"Coordinator: running at {public_url.rstrip('/')}")
        self.coordinator_start_button.setEnabled(False)
        self.coordinator_stop_button.setEnabled(True)
        self.project_state.setText("Coordinator running")
        self.job_manager_log.append(f"Coordinator API started on {host}:{port}.")
        self.job_manager_log.append(f"Artifact sync root: {artifact_root}")

    def stop_coordinator_server(self) -> None:
        """Stop the coordinator API."""

        if self.coordinator_server is None:
            return
        self.coordinator_server.shutdown()
        if self.coordinator_thread is not None:
            self.coordinator_thread.join(timeout=3)
        self.coordinator_server = None
        self.coordinator_thread = None
        self.coordinator_status_label.setText("Coordinator: stopped")
        self.coordinator_start_button.setEnabled(True)
        self.coordinator_stop_button.setEnabled(False)
        self.project_state.setText("Coordinator stopped")
        self.job_manager_log.append("Coordinator API stopped.")

    def publish_remote_training_job(self) -> None:
        """Bundle the current training setup and queue it for remote workers."""

        if self.coordinator_server is None:
            self.start_coordinator_server()
            if self.coordinator_server is None:
                return
        try:
            job = self._current_remote_training_job()
            artifact_root = Path(self.coordinator_artifact_root.text().strip()).expanduser()
            base_url = f"{self.coordinator_public_url.text().strip().rstrip('/')}/artifacts"
            bundle_path = create_job_artifact_bundle(job, artifact_root=artifact_root, base_url=base_url)
            self.job_manager.submit(job)
        except Exception as exc:
            QMessageBox.warning(self, "Publish failed", f"Could not publish remote job:\n{exc}")
            return
        self.job_manager_log.append(f"Published remote job: {job.job_id}")
        self.job_manager_log.append(f"Input bundle: {bundle_path}")
        self.job_manager_log.append(f"Worker download URL: {job.metadata.get('artifact_bundle_url')}")
        self.project_state.setText("Remote job queued")
        self.refresh_job_manager_tab()

    def _current_remote_training_job(self) -> TrainingJobSpec:
        """Build a remote-worker job from current training controls.

        Returns:
            Complete training job spec ready to bundle and queue.

        Raises:
            FileNotFoundError: If the prepared dataset is missing.
            ValueError: If model or training options are invalid.
        """

        dataset_dir = Path(self.train_data_dir.text().strip())
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Prepared dataset folder does not exist: {dataset_dir}")
        for file_name in ("tokenizer.json", "train_tokens.json", "val_tokens.json"):
            if not (dataset_dir / file_name).exists():
                raise FileNotFoundError(f"Prepared dataset is missing {file_name}. Prepare the dataset first.")
        vocab_size = self._current_training_vocab_size(dataset_dir)
        if vocab_size <= 0:
            raise ValueError("Could not determine tokenizer vocabulary size from the prepared dataset.")
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        if resume_path is None and self.resume_training.isChecked():
            resume_path = latest_checkpoint(Path(self.model_dir.text()) / "checkpoints")
        model_config = self._current_model_config(vocab_size=vocab_size)
        training_config = self._current_training_config(resume_path, training_mode="pretrain")
        model_config.validate()
        training_config.validate()
        job = TrainingJobSpec.local(
            dataset_dir,
            model_config,
            training_config,
            metadata={
                "project_name": self.search_box.text().strip(),
                "submitted_from": "desktop_ui",
                "coordinator_url": self.coordinator_public_url.text().strip().rstrip("/"),
            },
        )
        job.runtime = RuntimeSpec(
            backend=BackendKind.REMOTE_CLIENT,
            device=training_config.device,
            tags=[training_config.device, "remote"],
        )
        return job

    def _active_job_for_worker(self, worker_id: str) -> str:
        """Return the active job ID for a worker.

        Args:
            worker_id: Worker identifier.

        Returns:
            Active job ID or empty string.
        """

        for managed in self.job_manager.list_jobs():
            if managed.assigned_worker_id == worker_id and managed.spec.status.value in {"assigned", "running", "paused", "stopping"}:
                return managed.spec.job_id
        return ""

    @staticmethod
    def _metric_pair(value: Optional[int], total: Optional[int]) -> str:
        """Format a metric pair.

        Args:
            value: Current value.
            total: Total value.

        Returns:
            Display text.
        """

        if value is None:
            return "-"
        if total is None:
            return str(value)
        return f"{value}/{total}"

    @staticmethod
    def _metric_float(value: Optional[float], suffix: str = "") -> str:
        """Format a floating-point metric.

        Args:
            value: Metric value.
            suffix: Optional suffix.

        Returns:
            Display text.
        """

        if value is None:
            return "-"
        return f"{value:.4g}{suffix}"

    def _init_telemetry_store(self, model_dir: Path) -> None:
        """Create or reset the SQLite telemetry store for a training run.

        Args:
            model_dir: Model output directory.
        """

        self.telemetry_db_path = initialize_store(model_dir)
        self.telemetry_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_timeline_label.setText("Timeline: live")
        self.live_scrub_active = False

    def _record_live_metric(self, event: dict[str, Any]) -> None:
        """Persist one live training metric event to SQLite.

        Args:
            event: Training progress event.
        """

        if self.telemetry_db_path is None or not self.telemetry_run_id or event.get("step") is None:
            return
        self.telemetry_latest_id = insert_metric(self.telemetry_db_path, self.telemetry_run_id, event)
        self.telemetry_latest_index += 1
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, self.telemetry_latest_index)
        if not self.live_scrub_active:
            self.live_time_slider.setValue(self.telemetry_latest_index)
            self.live_timeline_label.setText("Timeline: live")
        self.live_time_slider.blockSignals(False)

    def _load_existing_telemetry(self, model_dir: Path) -> None:
        """Load the latest saved telemetry run for an opened project.

        Args:
            model_dir: Model output directory that may contain ``training_telemetry.sqlite``.
        """

        db_path = telemetry_db_path(model_dir)
        self.telemetry_db_path = db_path if db_path.exists() else None
        self.telemetry_run_id = ""
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_scrub_active = False
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_time_slider.blockSignals(False)
        self.live_timeline_label.setText("Timeline: no saved telemetry")
        self.live_sample_text.setText("Training text: -")
        if self.telemetry_db_path is None:
            return
        try:
            run_row = latest_run(self.telemetry_db_path)
            if run_row is None:
                self.live_timeline_label.setText("Timeline: no samples")
                return
            self.telemetry_run_id = str(run_row["run_id"])
            self.telemetry_latest_index = int(run_row["sample_count"] or 0)
            self.telemetry_latest_id = int(run_row["latest_id"] or 0)
        except sqlite3.Error as exc:
            self.live_timeline_label.setText("Timeline: could not load")
            self.training_log.append(f"Telemetry load warning: {exc}")
            return
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, self.telemetry_latest_index)
        self.live_time_slider.setValue(self.telemetry_latest_index)
        self.live_time_slider.blockSignals(False)
        if self.telemetry_latest_index:
            rows = self._timeline_rows_until(self.telemetry_latest_index)
            if rows:
                self._apply_timeline_rows(rows)

    def _timeline_rows_until(self, sample_index: int) -> list[sqlite3.Row]:
        """Load telemetry rows up to a selected sample index.

        Args:
            sample_index: Maximum number of samples to load for the active run.

        Returns:
            Ordered telemetry rows for the active run.
        """

        if self.telemetry_db_path is None or not self.telemetry_run_id or sample_index <= 0:
            return []
        return rows_until(self.telemetry_db_path, self.telemetry_run_id, sample_index)

    def _begin_live_scrub(self) -> None:
        """Pause live auto-follow while the timeline slider is being dragged."""

        self.live_scrub_active = True

    def _end_live_scrub(self) -> None:
        """Apply the selected timeline snapshot after slider drag."""

        self._scrub_live_timeline(self.live_time_slider.value())

    def _jump_live_timeline_to_latest(self) -> None:
        """Return timeline display to the latest live point."""

        self.live_scrub_active = False
        self.live_time_slider.setValue(self.telemetry_latest_index)
        self._scrub_live_timeline(self.telemetry_latest_index)
        self.live_timeline_label.setText("Timeline: live")

    def _scrub_live_timeline(self, sample_index: int) -> None:
        """Replay charts and live visual widgets to a selected telemetry point.

        Args:
            sample_index: Timeline sample selected by the slider.
        """

        rows = self._timeline_rows_until(sample_index)
        if not rows:
            return
        self._apply_timeline_rows(rows)

    def _apply_timeline_rows(self, rows: list[sqlite3.Row]) -> None:
        """Apply historical telemetry rows to charts and live widgets.

        Args:
            rows: Ordered SQLite telemetry rows.
        """

        def series(name: str) -> list[tuple[int, float]]:
            return [(int(row["step"]), float(row[name])) for row in rows if row[name] is not None]

        latest = rows[-1]
        self.loss_chart.set_points(series("train_loss"), series("val_loss"))
        self.optimization_chart.set_points(series("learning_rate"), series("grad_norm"))
        self.stability_chart.set_points(series("weight_norm"), series("update_ratio"))
        self.throughput_chart.set_points(series("tokens_per_second"), series("samples_per_second"))
        self.memory_chart.set_points(series("vram_allocated_gb"), series("vram_reserved_gb"))
        snapshot = {key: latest[key] for key in latest.keys()}
        sample_text = str(snapshot.get("sample_text") or "").strip()
        if sample_text:
            self.live_sample_text.setText(f"Training text: {self._compact_preview_text(sample_text, 220)}")
        else:
            self.live_sample_text.setText("Training text: -")
        self._update_live_training_metrics(
            int(latest["step"]),
            snapshot,
            snapshot.get("train_loss"),
            snapshot.get("learning_rate"),
            snapshot.get("grad_norm"),
            snapshot.get("update_ratio"),
            snapshot.get("tokens_per_second"),
            snapshot.get("samples_per_second"),
            snapshot.get("vram_allocated_gb"),
            snapshot.get("vram_reserved_gb"),
            snapshot.get("gpu_memory_percent"),
            snapshot.get("system_cpu_percent"),
            snapshot.get("system_ram_percent"),
            snapshot.get("data_loader_workers"),
        )
        timestamp = datetime.fromtimestamp(float(latest["recorded_at"])).strftime("%H:%M:%S")
        self.live_timeline_label.setText(f"Timeline: step {int(latest['step']):,} @ {timestamp}")

    @staticmethod
    def _compact_preview_text(text: str, limit: int = 220) -> str:
        """Normalize a training preview into a compact single line.

        Args:
            text: Raw decoded preview text.
            limit: Maximum number of displayed characters.

        Returns:
            Single-line text preview.
        """

        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 3)].rstrip() + "..."

    def _build_export_tab(self) -> QWidget:
        """Build the export page.

        Returns:
            Export page widget.
        """

        return build_export_tab(self)

    def _build_benchmark_tab(self) -> QWidget:
        """Build the benchmark prompt page.

        Returns:
            Benchmark page widget.
        """

        return build_benchmark_tab(self)

    def _build_chat_tab(self) -> QWidget:
        """Build the model test chat page.

        Returns:
            Chat page widget.
        """

        return build_chat_tab(self)

    def _panel(self) -> QWidget:
        """Create a base page panel.

        Returns:
            Panel widget.
        """

        page = QWidget()
        page.setObjectName("Panel")
        return page

    def _page_title(self, text: str) -> QLabel:
        """Create a page title label.

        Args:
            text: Title text.

        Returns:
            Label configured as a page title.
        """

        label = QLabel(text)
        label.setObjectName("PageTitle")
        return label

    def _metric_chip(self, text: str, tooltip: str) -> QLabel:
        """Create a compact metric display label.

        Args:
            text: Initial metric text.
            tooltip: User-facing explanation.

        Returns:
            Configured metric label.
        """

        label = QLabel(text)
        label.setObjectName("MetricChip")
        label.setMinimumWidth(150)
        label.setMinimumHeight(28)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(label, tooltip)
        return label

    def _hardware_meter(self, name: str) -> QProgressBar:
        """Create a slider-like hardware utilization meter.

        Args:
            name: Display name for the meter.

        Returns:
            Configured progress bar.
        """

        meter = QProgressBar()
        meter.setObjectName("HardwareMeter")
        meter.setRange(0, 100)
        meter.setValue(0)
        meter.setTextVisible(False)
        meter.setFixedHeight(8)
        meter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(meter, f"Live {name} utilization.")
        return meter

    def _set_meter(self, meter: QProgressBar, name: str, value: Optional[float]) -> None:
        """Update a hardware utilization meter.

        Args:
            meter: Meter to update.
            name: Display name for the meter.
            value: Utilization percentage.
        """

        if value is None:
            meter.setValue(0)
            label = self.hardware_meter_labels.get(id(meter))
            if label is not None:
                label.setText(f"{name}: -")
            return
        bounded = max(0.0, min(100.0, float(value)))
        meter.setValue(int(round(bounded)))
        label = self.hardware_meter_labels.get(id(meter))
        if label is not None:
            label.setText(f"{name}: {bounded:.1f}%")

    def _update_dataset_quality_report(self, summary: dict[str, Any]) -> None:
        """Update dataset quality chips from a summary dictionary.

        Args:
            summary: Dataset summary fields.
        """

        document_count = int(summary.get("document_count", 0) or 0)
        token_count = int(summary.get("token_count", 0) or 0)
        character_count = int(summary.get("character_count", 0) or 0)
        vocab_size = int(summary.get("tokenizer_vocab_size", summary.get("vocab_size", 0)) or 0)
        code_count = int(summary.get("code_sample_count", 0) or 0)
        prose_count = int(summary.get("prose_sample_count", 0) or 0)
        conversation_count = int(summary.get("conversation_sample_count", 0) or 0)
        cached_count = int(summary.get("cached_file_count", 0) or 0)
        processed_count = int(summary.get("processed_file_count", 0) or 0)
        skipped_count = int(summary.get("skipped_file_count", 0) or 0)
        failed_count = int(summary.get("failed_file_count", 0) or 0)
        warning = str(summary.get("warning") or "none")
        self.dataset_quality_samples.setText(f"Samples: {document_count:,}")
        self.dataset_quality_tokens.setText(f"Tokens: {token_count:,}")
        self.dataset_quality_vocab.setText(f"Vocab: {vocab_size:,}" if vocab_size else "Vocab: -")
        self.dataset_quality_code.setText(f"Code/prose/chat: {code_count:,}/{prose_count:,}/{conversation_count:,}")
        self.dataset_quality_balance.setText("Balance: prepared")
        self.dataset_quality_readiness.setText("Readiness: preview needed")
        self.dataset_quality_cache.setText(f"Files: {processed_count:,} ok, {cached_count:,} cached, {skipped_count:,} skipped, {failed_count:,} failed")
        self.dataset_quality_warning.setText(f"Warnings: {warning}")
        self._tip(self.dataset_quality_samples, f"{character_count:,} source characters across prepared samples.")

    def _reset_dataset_quality_report(self) -> None:
        """Reset dataset quality chips to their empty state."""

        self.dataset_quality_samples.setText("Samples: -")
        self.dataset_quality_tokens.setText("Tokens: -")
        self.dataset_quality_vocab.setText("Vocab: -")
        self.dataset_quality_code.setText("Code/prose: -")
        self.dataset_quality_balance.setText("Balance: -")
        self.dataset_quality_readiness.setText("Readiness: -")
        self.dataset_quality_cache.setText("Cache: -")
        self.dataset_quality_duplicates.setText("Duplicates: -")
        self.dataset_quality_extraction.setText("Extraction: -")
        self.dataset_quality_warning.setText("Warnings: none")
        if hasattr(self, "dataset_advisor"):
            self.dataset_advisor.setPlainText("Run Preview Dataset to get cleanup suggestions.")

    def _card(self, title: str, content_layout: Union[QVBoxLayout, QFormLayout, QGridLayout, QHBoxLayout]) -> QWidget:
        """Create a neon module card.

        Args:
            title: Card heading.
            content_layout: Layout to place inside the card.

        Returns:
            Card widget.
        """

        card = QWidget()
        card.setObjectName("Card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("SectionLabel")
        layout.addWidget(title_label)
        layout.addLayout(content_layout)
        return card

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        """Create a bounded integer input.

        Args:
            minimum: Minimum value.
            maximum: Maximum value.
            value: Initial value.

        Returns:
            Configured spin box.
        """

        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setMaximumWidth(220)
        return spin

    def _double_spin(self, minimum: float, maximum: float, value: float, step: float, decimals: int) -> QDoubleSpinBox:
        """Create a bounded float input.

        Args:
            minimum: Minimum value.
            maximum: Maximum value.
            value: Initial value.
            step: Increment step.
            decimals: Number of displayed decimal places.

        Returns:
            Configured double spin box.
        """

        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setMaximumWidth(220)
        return spin

    def _path_row(self, field: QLineEdit, directory: bool = True, file_filter: str = "Checkpoints (*.pt)") -> QWidget:
        """Create a path field with a browse button.

        Args:
            field: Path input widget.
            directory: Whether the browse dialog selects folders.
            file_filter: File dialog filter used when ``directory`` is false.

        Returns:
            Row widget containing the path input and button.
        """

        row = QWidget()
        row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        browse = QPushButton("Browse")
        browse.setFixedWidth(88)
        self._tip(browse, "Open a file/folder picker for this path.")
        field.setMinimumWidth(180)
        field.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        browse.clicked.connect(lambda: self._browse(field, directory, file_filter))
        layout.addWidget(field, 1)
        layout.addWidget(browse)
        return row

    def _configure_form(self, form: QFormLayout) -> None:
        """Apply common form spacing and growth policy.

        Args:
            form: Form layout to configure.
        """

        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)

    def _configure_device_options(self) -> None:
        """Populate training device choices without duplicate CPU entries."""

        self.device.clear()
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self.device.addItem("cuda")
            self.device.addItem("cpu")
            self.device_info.setText(f"CUDA ready: {device_name}")
            self.use_amp_default = True
        else:
            self.device.addItem("cpu")
            cuda_build = getattr(torch.backends, "cuda", None)
            built_with_cuda = bool(cuda_build and torch.backends.cuda.is_built())
            if built_with_cuda:
                detail = "CUDA build found, but no usable NVIDIA GPU/driver was detected."
            else:
                detail = "CUDA is not available in this PyTorch install."
            self.device_info.setText(detail)
            self.use_amp_default = False

    def _thin_progress(self) -> QProgressBar:
        """Create a thin bottom progress bar.

        Returns:
            Configured progress bar.
        """

        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setTextVisible(False)
        progress.setFixedHeight(4)
        progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._tip(progress, "Progress indicator for the current page operation.")
        return progress

    def _tip(self, widget: QWidget, text: str) -> None:
        """Attach tooltip and status tip text.

        Args:
            widget: Widget receiving the tip.
            text: Tooltip text.
        """

        widget.setToolTip(text)
        widget.setStatusTip(text)

    def _render_chat_markdown(self, markdown_text: str) -> None:
        """Render chat Markdown with highlighted fenced code blocks when possible.

        Args:
            markdown_text: Markdown transcript to render.
        """

        if not hasattr(self, "current_assistant_message") or self.current_assistant_message is None:
            return
        self.current_assistant_message.set_content(markdown_text)

    def _add_chat_message(
        self,
        role: str,
        content: str,
        metrics: str = "",
        resend_prompt: Optional[str] = None,
    ) -> QTextBrowser:
        """Add one chat bubble.

        Args:
            role: Message role, either ``user`` or ``assistant``.
            content: Markdown message content.
            metrics: Optional metric text shown under assistant replies.
            resend_prompt: Prompt to resend from the bubble.

        Returns:
            Text browser used by the bubble.
        """

        should_follow = self._is_chat_near_bottom()
        max_width = max(320, int(self.chat_scroll.viewport().width() * 0.78)) if hasattr(self, "chat_scroll") else 900
        message = ChatMessageWidget(
            role,
            content,
            markdown_to_html,
            self._resend_chat_message,
            metrics=metrics,
            resend_prompt=resend_prompt,
            max_width=max_width,
        )
        self.chat_messages.insertWidget(max(self.chat_messages.count() - 1, 0), message)
        if should_follow:
            message.scroll_later(lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))
        if role == "assistant":
            self.current_assistant_message = message
            self.current_assistant_browser = message.browser
            self.current_assistant_meta = message.meta_label
        return message.browser

    def _is_chat_near_bottom(self) -> bool:
        """Return whether the chat scroll is close enough to follow streaming.

        Returns:
            True when the view should auto-scroll.
        """

        if not hasattr(self, "chat_scroll"):
            return True
        bar = self.chat_scroll.verticalScrollBar()
        return bar.maximum() - bar.value() < 48

    def _clear_chat_messages(self) -> None:
        """Remove all message bubbles."""

        while self.chat_messages.count() > 1:
            item = self.chat_messages.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.current_assistant_message = None
        self.current_assistant_browser = None
        self.current_assistant_meta = None

    def _resend_chat_message(self, prompt: str) -> None:
        """Resend text from a message bubble.

        Args:
            prompt: Prompt text to send.
        """

        self.chat_input.setPlainText(prompt)
        self.send_chat_message()

    def _set_chat_stats(self, elapsed_seconds: float, token_count: int, tokens_per_second: float) -> None:
        """Update live chat generation metrics.

        Args:
            elapsed_seconds: Elapsed generation time.
            token_count: Generated token count.
            tokens_per_second: Approximate token speed.
        """

        text = f"Time: {elapsed_seconds:.2f}s  |  Tokens: {token_count:,}  |  Speed: {tokens_per_second:.2f} tok/s"
        self.chat_stats.setText(text)
        if self.current_assistant_meta is not None:
            self.current_assistant_meta.setText(text)
            self.current_assistant_meta.setVisible(True)

    def _chat_backend_value(self) -> str:
        """Return the selected chat model backend.

        Returns:
            Stable chat backend identifier.
        """

        if not hasattr(self, "chat_model_backend"):
            return "gguf"
        return "microgpt" if self.chat_model_backend.currentText() == "MicroGPT checkpoint" else "gguf"

    def _update_chat_backend_controls(self) -> None:
        """Show controls relevant to the selected chat backend."""

        if not hasattr(self, "chat_model_backend"):
            return
        native = self._chat_backend_value() == "microgpt"
        self.gguf_path_row.setVisible(not native)
        self.microgpt_path_row.setVisible(native)
        self.llama_gpu_layers.setEnabled(not native)
        self.llama_threads.setEnabled(not native)
        self.llama_context.setEnabled(not native)
        if native:
            self._tip(self.load_llm_button, "Load the native MicroGPT checkpoint into memory once for repeated chat messages.")
        else:
            self._tip(self.load_llm_button, "Load the GGUF model into memory once for repeated chat messages.")

    def _lightning_icon(self) -> QIcon:
        """Create the window lightning icon.

        Returns:
            Lightning icon.
        """

        return self._static_lightning_icon()

    @staticmethod
    def _static_lightning_icon() -> QIcon:
        """Create the static lightning icon.

        Returns:
            Lightning icon.
        """

        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QBrush(QColor("#1f1f1f")))
            painter.setPen(QPen(QColor("#f5b041"), 3))
            painter.drawRoundedRect(4, 4, 56, 56, 12, 12)
            bolt = QPolygon([
                QPoint(36, 8),
                QPoint(17, 35),
                QPoint(31, 35),
                QPoint(25, 56),
                QPoint(48, 25),
                QPoint(33, 25),
            ])
            painter.setPen(QPen(QColor("#ffd27a"), 2))
            painter.setBrush(QBrush(QColor("#f5b041")))
            painter.drawPolygon(bolt)
        finally:
            painter.end()
        return QIcon(pixmap)

    @staticmethod
    def _windows_icon_path() -> Path:
        """Return the Windows icon file path.

        Returns:
            Path to the generated ``.ico`` file.
        """

        return Path(__file__).with_name("micro_llm_creator_lightning.ico")

    @staticmethod
    def _ensure_windows_icon_file() -> Optional[Path]:
        """Ensure the generated Windows ``.ico`` file exists.

        Returns:
            Icon path on Windows, otherwise ``None``.
        """

        if sys.platform != "win32":
            return None
        icon_path = MainWindow._windows_icon_path()
        if icon_path.exists():
            return icon_path
        icon = MainWindow._static_lightning_icon()
        pixmap = icon.pixmap(256, 256)
        if pixmap.isNull() or not pixmap.save(str(icon_path), "ICO"):
            return None
        return icon_path

    def apply_windows_taskbar_icon(self) -> None:
        """Apply the lightning icon to the native Windows window handle."""

        if sys.platform != "win32":
            return
        icon_path = self._ensure_windows_icon_file()
        if icon_path is None:
            return

        hwnd = int(self.winId())
        if not hwnd:
            return

        wm_seticon = 0x0080
        icon_small = 0
        icon_big = 1
        image_icon = 1
        lr_loadfromfile = 0x0010

        user32 = ctypes.windll.user32
        hicon_big = user32.LoadImageW(None, str(icon_path), image_icon, 256, 256, lr_loadfromfile)
        hicon_small = user32.LoadImageW(None, str(icon_path), image_icon, 32, 32, lr_loadfromfile)
        if hicon_big:
            user32.SendMessageW(hwnd, wm_seticon, icon_big, hicon_big)
            self._windows_icon_handles.append(hicon_big)
        if hicon_small:
            user32.SendMessageW(hwnd, wm_seticon, icon_small, hicon_small)
            self._windows_icon_handles.append(hicon_small)

    def _browse(self, field: QLineEdit, directory: bool, file_filter: str = "Checkpoints (*.pt)") -> None:
        """Open a file or folder picker for a path field.

        Args:
            field: Path input to update.
            directory: Whether to select a folder instead of a file.
            file_filter: File dialog filter used for files.
        """

        if directory:
            value = QFileDialog.getExistingDirectory(self, "Choose folder", field.text() or str(Path.cwd()))
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Choose file", field.text() or str(Path.cwd()), file_filter)
        if value:
            field.setText(value)

    def save_project(self) -> None:
        """Save the current project settings into a named project folder."""

        project_name = self.search_box.text().strip() or "MicroLLMProject"
        safe_name = self._safe_project_name(project_name)
        if self.current_project_file is None:
            base_dir = QFileDialog.getExistingDirectory(self, "Choose parent folder for project", str(Path.cwd()))
            if not base_dir:
                return
            project_dir = Path(base_dir) / safe_name
            project_file = project_dir / "project.json"
        else:
            project_file = self.current_project_file
            project_dir = project_file.parent
        project_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_project_workspace(project_dir)
        if self.current_project_file is None:
            self._apply_project_workspace_paths(project_dir)
        project_file.write_text(json.dumps(self._project_state_dict(project_name, project_dir), indent=2), encoding="utf-8")
        self.current_project_file = project_file
        self._apply_project_runtime_environment(project_dir)
        self.project_state.setText("Project saved")
        LOGGER.info("Project saved: %s", project_file)
        if self.current_project_file == project_file:
            self.dataset_log.append(f"Project saved: {project_file}")
            self.dataset_log.append(f"Project workspace: {project_dir}")

    def new_project(self) -> None:
        """Start a fresh project and clear the active project file binding."""

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please stop or wait for the current task before creating a new project.")
            return
        if self.current_project_file is not None or self.search_box.text().strip():
            choice = QMessageBox.question(
                self,
                "New project",
                "Start a new project? Unsaved changes in the current project will not be saved automatically.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if choice != QMessageBox.Yes:
                return

        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        base_dir = QFileDialog.getExistingDirectory(self, "Choose folder where the new project will be created", str(Path.cwd()))
        if not base_dir:
            return
        project_name = self.search_box.text().strip() or "MicroLLMProject"
        project_dir = Path(base_dir) / self._safe_project_name(project_name)
        project_file = project_dir / "project.json"
        project_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_project_workspace(project_dir)
        self.current_project_file = project_file
        self._apply_project_state(self._default_project_state())
        self.search_box.setText(project_name)
        self._apply_project_workspace_paths(project_dir)
        self._apply_project_runtime_environment(project_dir)
        self._reset_project_runtime_state()
        project_file.write_text(json.dumps(self._project_state_dict(project_name, project_dir), indent=2), encoding="utf-8")
        self.project_state.setText("New project")
        LOGGER.info("New project created: %s", project_file)
        self.dataset_log.append(f"Started a new project: {project_file}")
        self.dataset_log.append(f"Project workspace: {project_dir}")

    def open_project(self) -> None:
        """Open a saved project file and restore UI settings."""

        project_file, _ = QFileDialog.getOpenFileName(
            self,
            "Open Micro LLM project",
            str(Path.cwd()),
            "Micro LLM project (project.json *.json);;All files (*)",
        )
        if not project_file:
            return
        try:
            data = json.loads(Path(project_file).read_text(encoding="utf-8"))
            self._apply_project_state(data)
        except Exception as exc:
            QMessageBox.warning(self, "Open failed", f"Could not open project:\n{exc}")
            return
        self.current_project_file = Path(project_file)
        self._ensure_project_workspace(self.current_project_file.parent)
        self._apply_project_runtime_environment(self.current_project_file.parent)
        if self.model_dir.text().strip():
            self._load_existing_telemetry(Path(self.model_dir.text()))
        self.project_state.setText("Project opened")
        LOGGER.info("Project opened: %s", project_file)
        self.dataset_log.append(f"Opened project: {project_file}")
        self.refresh_model_estimate()

    def _ensure_project_workspace(self, project_dir: Path) -> None:
        """Create standard folders inside a project.

        Args:
            project_dir: Project root folder.
        """

        for name in ("datasets", "models", "exports", "training_data", "cache", "temp"):
            (project_dir / name).mkdir(parents=True, exist_ok=True)

    def _apply_project_workspace_paths(self, project_dir: Path) -> None:
        """Point project output fields at the standard project folders.

        Args:
            project_dir: Project root folder.
        """

        dataset_dir = project_dir / "datasets"
        model_dir = project_dir / "models"
        export_dir = project_dir / "exports"
        training_data_dir = project_dir / "training_data"
        self.dataset_dir.setText(str(dataset_dir))
        self.train_data_dir.setText(str(dataset_dir))
        self.model_dir.setText(str(model_dir))
        self.export_model_dir.setText(str(model_dir))
        self.export_dir.setText(str(export_dir))
        self.gguf_output_path.setText(str(export_dir / "model.gguf"))
        if not self.input_dir.text().strip():
            self.input_dir.setText(str(training_data_dir))

    def _apply_project_runtime_environment(self, project_dir: Path) -> None:
        """Prefer project-local cache/temp folders for runtime work.

        Args:
            project_dir: Project root folder.
        """

        cache_dir = project_dir / "cache"
        temp_dir = project_dir / "temp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for key in ("TMPDIR", "TEMP", "TMP"):
            os.environ[key] = str(temp_dir)
        for key in ("TORCH_HOME", "HF_HOME", "TRANSFORMERS_CACHE", "PYTORCH_KERNEL_CACHE"):
            os.environ[key] = str(cache_dir / key.lower())
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    def _default_project_state(self) -> dict[str, Any]:
        """Build the default state used for a newly created project.

        Returns:
            JSON-style project state with fresh paths and default settings.
        """

        runs_dir = Path.cwd() / "runs"
        dataset_dir = runs_dir / "dataset"
        model_dir = runs_dir / "model"
        export_dir = runs_dir / "export"
        return {
            "schema": "micro_llm_creator_project",
            "version": 1,
            "project_name": "",
            "project_dir": "",
            "paths": {
                "source_vault": "",
                "dataset_core": str(dataset_dir),
                "training_dataset": str(dataset_dir),
                "model_output": str(model_dir),
                "export_model_core": str(model_dir),
                "export_output": str(export_dir),
                "llama_cpp_dir": "",
                "gguf_output_path": str(export_dir / "model.gguf"),
                "gguf_model": "",
                "microgpt_chat_model": "",
                "tokenizer_import": "",
                "resume_checkpoint": "",
                "fine_tune_checkpoint": "",
            },
            "dataset": {
                "auto_vocab": True,
                "manual_vocab_size": 8000,
                "include_conversation_datasets": False,
                "dataset_stage": "base",
                "conversation_datasets": [],
                "conversation_sample_limit": 20000,
                "mixture_weights": {
                    "local_prose": 50.0,
                    "source_code": 30.0,
                    "online_base": 20.0,
                    "instruction": 0.0,
                    "conversation": 0.0,
                },
                "min_frequency": 2,
                "context_length": 128,
                "validation_split": 0.1,
                "lowercase": False,
                "max_workers": 4,
                "prepare_mode": "incremental",
                "tokenizer_strategy": "auto",
                "code_training_mode": True,
                "include_prose": True,
                "include_source_code": True,
                "extract_code_blocks": True,
                "preserve_indentation": True,
                "instruction_samples": True,
                "reasoning_sample_mode": "scaffold",
            },
            "training": {
                "preset": "Tiny",
                "architecture_style": "Classic GPT",
                "launch_target": "local",
                "training_mode": "pretrain",
                "training_stage": "base",
                "peft_method": "none",
                "lora_rank": 8,
                "lora_alpha": 16.0,
                "lora_dropout": 0.05,
                "lora_target_modules": "attention",
                "n_embd": 128,
                "n_head": 4,
                "n_layer": 4,
                "context_length": 128,
                "dropout": 0.1,
                "training_profile": "Stable LLM",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.0003,
                "weight_decay": 0.1,
                "gradient_accumulation": 1,
                "warmup_steps": 100,
                "eval_interval": 100,
                "max_eval_batches": 50,
                "save_interval": 500,
                "data_loader_workers": 0,
                "max_grad_norm": 1.0,
                "seed": 1337,
                "device": self.device.currentText(),
                "use_amp": self.use_amp_default,
                "resume": True,
                "require_compatible_resume": True,
                "benchmark_prompts": "\n\n".join(DEFAULT_BENCHMARK_PROMPTS),
                "benchmark_tokens": 128,
                "benchmark_temperature": 0.7,
                "benchmark_kv_cache": True,
            },
            "export": {
                "quantization": "FP16 checkpoint",
                "gguf_outtype": "f16",
            },
            "chat": {
                "model_backend": "gguf",
                "context": 2048,
                "cpu_threads": 4,
                "gpu_layers": -1,
                "thinking_enabled": True,
                "reasoning_effort": "Balanced",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "system_prompt": "",
            },
            "distributed": {
                "host": "0.0.0.0",
                "port": 8765,
                "artifact_root": str(Path.home() / ".micro_llm_creator" / "artifacts"),
                "public_url": "http://127.0.0.1:8765",
            },
            "artifacts": {},
        }

    def _reset_project_runtime_state(self) -> None:
        """Clear logs, progress, charts, and status labels for a new project."""

        self.dataset_log.clear()
        self.training_log.clear()
        self.fine_tune_log.clear()
        self.benchmark_log.clear()
        self.export_log.setPlainText(
            "Export options:\n"
            "- Bundle copies final_model.pt, tokenizer.json, and training_summary.json.\n"
            "- HF package writes model_core/hf_model for portable MicroGPT loading.\n"
            "- FP16 checkpoint quantization works now.\n"
            "- GGUF conversion uses llama.cpp when model_core/hf_model exists.\n"
            "- Native MicroGPT checkpoints are not written as fake GGUF files.\n"
        )
        for progress in (
            self.dataset_progress,
            self.training_progress,
            self.fine_tune_progress,
            self.benchmark_progress,
            self.export_progress,
            self.chat_progress,
        ):
            progress.setRange(0, 100)
            progress.setValue(0)
        self.dataset_status.setText("Dataset: not prepared")
        self.train_status.setText("Training: idle")
        self.export_status.setText("Export: waiting")
        self.chat_status.setText("Chat: no model loaded")
        self.prepare_button.setText("Prepare Dataset")
        self.train_button.setText("Start Training")
        self.fine_tune_button.setText("Start Fine-Tune")
        self.stop_dataset_button.setEnabled(False)
        self.stop_training_button.setEnabled(False)
        self.stop_fine_tune_button.setEnabled(False)
        self.stop_benchmark_button.setEnabled(False)
        self.stop_chat_button.setEnabled(False)
        self.load_llm_button.setText("Load Model")
        self._update_chat_backend_controls()
        self._reset_dataset_quality_report()
        self.training_epoch_metric.setText("Epoch: -")
        self.training_step_metric.setText("Step: -")
        self.training_loss_metric.setText("Train loss: -")
        self.training_val_metric.setText("Val loss: -")
        self.training_lr_metric.setText("LR: -")
        self.training_speed_metric.setText("Speed: -")
        self.training_grad_metric.setText("Grad: -")
        self.training_vram_metric.setText("VRAM: -")
        self.training_eta_metric.setText("ETA: -")
        self.history_metric.setText(f"Runs: {len(self._load_training_history())}")
        self.loss_chart.clear()
        self.optimization_chart.clear()
        self.stability_chart.clear()
        self.throughput_chart.clear()
        self.memory_chart.clear()
        self.live_prediction_chart.update_distribution(0, None)
        self.live_attention_chart.update_heatmap(0, None)
        self.live_activation_chart.update_histogram(0, None)
        self.live_gradient_chart.update_flow(self.n_layer.value(), None, 0)
        self.live_sample_text.setText("Training text: -")
        self.telemetry_db_path = None
        self.telemetry_run_id = ""
        self.telemetry_latest_id = 0
        self.telemetry_latest_index = 0
        self.live_scrub_active = False
        self.live_time_slider.blockSignals(True)
        self.live_time_slider.setRange(0, 0)
        self.live_time_slider.setValue(0)
        self.live_time_slider.blockSignals(False)
        self.live_timeline_label.setText("Timeline: no saved telemetry")
        self._set_meter(self.live_cpu_bar, "CPU", self._system_cpu_value())
        self._set_meter(self.live_gpu_bar, "GPU memory", None)
        self._set_meter(self.live_vram_bar, "VRAM reserved", None)
        self._set_meter(self.live_ram_bar, "System RAM", self._system_ram_value())
        self.live_worker_status.setText(f"CPU workers: {self.data_loader_workers.value()}")
        self._clear_chat_messages()
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self.chat_stats.setText("Idle")
        self._add_chat_message("assistant", "Load a GGUF or MicroGPT model to start testing.")

    def _project_state_dict(self, project_name: str, project_dir: Path) -> dict[str, Any]:
        """Collect all UI state that defines a Micro LLM project.

        Args:
            project_name: User-facing project name.
            project_dir: Folder where the project file will live.

        Returns:
            JSON-serializable project state.
        """

        dataset_dir = Path(self.dataset_dir.text()) if self.dataset_dir.text().strip() else None
        model_dir = Path(self.model_dir.text()) if self.model_dir.text().strip() else None
        export_dir = Path(self.export_dir.text()) if self.export_dir.text().strip() else None
        return {
            "schema": "micro_llm_creator_project",
            "version": 1,
            "project_name": project_name,
            "project_dir": str(project_dir),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "paths": {
                "source_vault": self.input_dir.text(),
                "dataset_core": self.dataset_dir.text(),
                "training_dataset": self.train_data_dir.text(),
                "model_output": self.model_dir.text(),
                "export_model_core": self.export_model_dir.text(),
                "export_output": self.export_dir.text(),
                "llama_cpp_dir": self.llama_cpp_dir.text(),
                "gguf_output_path": self.gguf_output_path.text(),
                "gguf_model": self.gguf_path.text(),
                "microgpt_chat_model": self.microgpt_chat_path.text(),
                "tokenizer_import": self.tokenizer_path.text(),
                "resume_checkpoint": self.resume_checkpoint.text(),
                "fine_tune_checkpoint": self.fine_tune_checkpoint.text(),
            },
            "dataset": {
                "auto_vocab": self.auto_vocab.isChecked(),
                "manual_vocab_size": self.manual_vocab_size.value(),
                "include_conversation_datasets": self.include_conversation_datasets.isChecked(),
                "dataset_stage": self._dataset_stage_value(),
                "conversation_datasets": self._selected_conversation_datasets(),
                "conversation_sample_limit": self.conversation_sample_limit.value(),
                "mixture_weights": self._mixture_weights_from_ui(),
                "min_frequency": self.min_frequency.value(),
                "context_length": self.context_length.value(),
                "validation_split": self.validation_split.value(),
                "lowercase": self.lowercase.isChecked(),
                "max_workers": self.max_workers.value(),
                "prepare_mode": self._prepare_mode_value(),
                "tokenizer_strategy": self._tokenizer_strategy_value(),
                "code_training_mode": self.code_training_mode.isChecked(),
                "include_prose": self.include_prose.isChecked(),
                "include_source_code": self.include_source_code.isChecked(),
                "extract_code_blocks": self.extract_code_blocks.isChecked(),
                "preserve_indentation": self.preserve_indentation.isChecked(),
                "instruction_samples": self.instruction_samples.isChecked(),
                "reasoning_sample_mode": self._reasoning_sample_mode_value(),
            },
            "training": {
                "preset": self.preset.currentText(),
                "architecture_style": self.architecture_style.currentText(),
                "launch_target": self._training_launch_target_value(),
                "training_stage": self._training_stage_value(),
                "n_embd": self.n_embd.value(),
                "n_head": self.n_head.value(),
                "attention_type": self._attention_type_value(),
                "kv_head_count": self.kv_head_count.value(),
                "attention_backend": self._attention_backend_value(),
                "attention_window": self.attention_window.value(),
                "training_mode": self._training_mode_value(),
                "peft_method": self._peft_method_value(),
                "lora_rank": self.lora_rank.value(),
                "lora_alpha": self.lora_alpha.value(),
                "lora_dropout": self.lora_dropout.value(),
                "lora_target_modules": self._lora_target_value(),
                "n_layer": self.n_layer.value(),
                "context_length": self.train_context_length.value(),
                "dropout": self.dropout.value(),
                "training_profile": self.training_profile.currentText(),
                "epochs": self.epochs.value(),
                "batch_size": self.batch_size.value(),
                "learning_rate": self.learning_rate.value(),
                "weight_decay": self.weight_decay.value(),
                "optimizer_name": self._optimizer_value(),
                "scheduler_name": self._scheduler_value(),
                "scheduler_min_lr_ratio": self.min_lr_ratio.value(),
                "polynomial_power": self.polynomial_power.value(),
                "gradient_accumulation": self.gradient_accumulation.value(),
                "warmup_steps": self.warmup_steps.value(),
                "eval_interval": self.eval_interval.value(),
                "max_eval_batches": self.max_eval_batches.value(),
                "save_interval": self.save_interval.value(),
                "data_loader_workers": self.data_loader_workers.value(),
                "max_grad_norm": self.max_grad_norm.value(),
                "seed": self.seed.value(),
                "device": self.device.currentText(),
                "use_amp": self.use_amp.isChecked(),
                "precision": self._precision_value(),
                "resume": self.resume_training.isChecked(),
                "require_compatible_resume": self.resume_safety.isChecked(),
                "benchmark_prompts": self.benchmark_prompts.toPlainText(),
                "benchmark_tokens": self.benchmark_tokens.value(),
                "benchmark_temperature": self.benchmark_temperature.value(),
                "benchmark_kv_cache": self.benchmark_kv_cache.isChecked(),
            },
            "export": {
                "quantization": self.quant_mode.currentText(),
                "gguf_outtype": self.gguf_outtype.currentText(),
            },
            "chat": {
                "model_backend": self._chat_backend_value(),
                "context": self.llama_context.value(),
                "cpu_threads": self.llama_threads.value(),
                "gpu_layers": self.llama_gpu_layers.value(),
                "thinking_enabled": self.thinking_enabled.isChecked(),
                "reasoning_effort": self.reasoning_effort.currentText(),
                "max_tokens": self.chat_max_tokens.value(),
                "temperature": self.chat_temperature.value(),
                "top_p": self.chat_top_p.value(),
                "repeat_penalty": self.chat_repeat_penalty.value(),
                "system_prompt": self.system_prompt.toPlainText(),
            },
            "distributed": {
                "host": self.coordinator_host.text(),
                "port": self.coordinator_port.value(),
                "artifact_root": self.coordinator_artifact_root.text(),
                "public_url": self.coordinator_public_url.text(),
            },
            "artifacts": {
                "dataset_summary": self._read_json_if_exists(dataset_dir / "dataset_summary.json") if dataset_dir else None,
                "training_summary": self._read_json_if_exists(model_dir / "training_summary.json") if model_dir else None,
                "export_summary": self._read_json_if_exists(export_dir / "export_summary.json") if export_dir else None,
            },
        }

    def _apply_project_state(self, data: dict[str, Any]) -> None:
        """Restore UI state from a saved project dictionary.

        Args:
            data: Project state loaded from JSON.
        """

        self.search_box.setText(str(data.get("project_name", "")))
        paths = data.get("paths", {})
        dataset = data.get("dataset", {})
        training = data.get("training", {})
        export = data.get("export", {})
        chat = data.get("chat", {})
        distributed = data.get("distributed", {})

        self.input_dir.setText(str(paths.get("source_vault", "")))
        self.dataset_dir.setText(str(paths.get("dataset_core", "")))
        self.train_data_dir.setText(str(paths.get("training_dataset", "")))
        self.model_dir.setText(str(paths.get("model_output", "")))
        self.export_model_dir.setText(str(paths.get("export_model_core", "")))
        self.export_dir.setText(str(paths.get("export_output", "")))
        self.llama_cpp_dir.setText(str(paths.get("llama_cpp_dir", "")))
        self.gguf_output_path.setText(str(paths.get("gguf_output_path", "")))
        self.gguf_path.setText(str(paths.get("gguf_model", "")))
        self.microgpt_chat_path.setText(str(paths.get("microgpt_chat_model", "")))
        self.tokenizer_path.setText(str(paths.get("tokenizer_import", "")))
        self.resume_checkpoint.setText(str(paths.get("resume_checkpoint", "")))
        self.fine_tune_checkpoint.setText(str(paths.get("fine_tune_checkpoint", "")))

        self.auto_vocab.setChecked(bool(dataset.get("auto_vocab", True)))
        self.manual_vocab_size.setValue(int(dataset.get("manual_vocab_size", self.manual_vocab_size.value())))
        include_conversation = bool(dataset.get("include_conversation_datasets", False))
        self._set_dataset_stage(str(dataset.get("dataset_stage", "base")))
        self.include_conversation_datasets.setChecked(include_conversation)
        self._set_selected_conversation_datasets(list(dataset.get("conversation_datasets", [])))
        self.conversation_sample_limit.setValue(int(dataset.get("conversation_sample_limit", self.conversation_sample_limit.value())))
        self._set_mixture_weights(dict(dataset.get("mixture_weights", {})))
        self.min_frequency.setValue(int(dataset.get("min_frequency", self.min_frequency.value())))
        self.context_length.setValue(int(dataset.get("context_length", self.context_length.value())))
        self.validation_split.setValue(float(dataset.get("validation_split", self.validation_split.value())))
        self.lowercase.setChecked(bool(dataset.get("lowercase", False)))
        self.max_workers.setValue(int(dataset.get("max_workers", self.max_workers.value())))
        self._set_combo_by_data(self.prepare_mode, str(dataset.get("prepare_mode", "incremental")), {
            "incremental": "Incremental update",
            "full_rebuild": "Full rebuild",
            "force_reprocess": "Force reprocess",
        })
        self._set_combo_by_data(self.tokenizer_strategy, str(dataset.get("tokenizer_strategy", "auto")), {
            "auto": "Auto",
            "train_new": "Train new tokenizer",
            "reuse_dataset": "Reuse dataset tokenizer",
            "import_tokenizer": "Import tokenizer.json",
        })
        self.code_training_mode.setChecked(bool(dataset.get("code_training_mode", True)))
        self.include_prose.setChecked(bool(dataset.get("include_prose", True)))
        self.include_source_code.setChecked(bool(dataset.get("include_source_code", True)))
        self.extract_code_blocks.setChecked(bool(dataset.get("extract_code_blocks", True)))
        self.preserve_indentation.setChecked(bool(dataset.get("preserve_indentation", True)))
        self.instruction_samples.setChecked(bool(dataset.get("instruction_samples", True)))
        self._set_combo_by_data(self.reasoning_sample_mode, str(dataset.get("reasoning_sample_mode", "scaffold")), {
            "scaffold": "Reasoning scaffold",
            "detailed": "Detailed code reasoning",
            "none": "No reasoning wrapper",
        })

        self._set_combo_text(self.preset, str(training.get("preset", self.preset.currentText())))
        self._set_combo_text(self.architecture_style, str(training.get("architecture_style", self.architecture_style.currentText())))
        self._set_combo_by_data(self.training_launch_target, str(training.get("launch_target", "local")), {
            "local": "Local machine",
            "remote": "Remote workers",
        })
        self.n_embd.setValue(int(training.get("n_embd", self.n_embd.value())))
        self.n_head.setValue(int(training.get("n_head", self.n_head.value())))
        self._set_combo_by_data(self.attention_type, str(training.get("attention_type", "mha")), {
            "mha": "Multi-head",
            "gqa": "Grouped-query",
            "mqa": "Multi-query",
        })
        self.kv_head_count.setValue(int(training.get("kv_head_count", self.kv_head_count.value())))
        self._set_combo_by_data(self.attention_backend, str(training.get("attention_backend", "sdpa")), {
            "sdpa": "SDPA / Flash when available",
            "manual": "Manual",
        })
        self.attention_window.setValue(int(training.get("attention_window", self.attention_window.value())))
        self._set_combo_by_data(self.training_mode, str(training.get("training_mode", "pretrain")), {
            "pretrain": "Pretrain from scratch",
            "fine_tune": "Fine-tune checkpoint",
            "instruction_fine_tune": "Instruction fine-tune",
            "conversation_fine_tune": "Conversation fine-tune",
        })
        training_stage = str(training.get("training_stage", ""))
        if training_stage == "instruction":
            self._set_combo_text(self.training_mode, "Instruction fine-tune")
        elif training_stage == "conversation":
            self._set_combo_text(self.training_mode, "Conversation fine-tune")
        self._set_combo_by_data(self.peft_method, str(training.get("peft_method", "none")), {
            "none": "Full fine-tune",
            "lora": "LoRA adapters",
        })
        self.lora_rank.setValue(int(training.get("lora_rank", self.lora_rank.value())))
        self.lora_alpha.setValue(float(training.get("lora_alpha", self.lora_alpha.value())))
        self.lora_dropout.setValue(float(training.get("lora_dropout", self.lora_dropout.value())))
        self._set_combo_by_data(self.lora_targets, str(training.get("lora_target_modules", "attention")), {
            "attention": "Attention projections",
            "mlp": "MLP projections",
            "attention,mlp": "Attention + MLP",
        })
        self.n_layer.setValue(int(training.get("n_layer", self.n_layer.value())))
        self.train_context_length.setValue(int(training.get("context_length", self.train_context_length.value())))
        self.dropout.setValue(float(training.get("dropout", self.dropout.value())))
        self._set_combo_text(self.training_profile, str(training.get("training_profile", self.training_profile.currentText())))
        self.epochs.setValue(int(training.get("epochs", self.epochs.value())))
        self.batch_size.setValue(int(training.get("batch_size", self.batch_size.value())))
        self.learning_rate.setValue(float(training.get("learning_rate", self.learning_rate.value())))
        self.weight_decay.setValue(float(training.get("weight_decay", self.weight_decay.value())))
        self._set_combo_by_data(self.optimizer_name, str(training.get("optimizer_name", "adamw")), {
            "adamw": "AdamW",
            "adam": "Adam",
            "lion": "Lion",
            "adafactor": "Adafactor",
        })
        self._set_combo_by_data(self.scheduler_name, str(training.get("scheduler_name", "warmup_linear")), {
            "warmup_linear": "Warmup linear",
            "cosine": "Cosine decay",
            "polynomial": "Polynomial decay",
            "one_cycle": "One-cycle",
            "constant": "Constant",
        })
        self.min_lr_ratio.setValue(float(training.get("scheduler_min_lr_ratio", self.min_lr_ratio.value())))
        self.polynomial_power.setValue(float(training.get("polynomial_power", self.polynomial_power.value())))
        self.gradient_accumulation.setValue(int(training.get("gradient_accumulation", self.gradient_accumulation.value())))
        self.warmup_steps.setValue(int(training.get("warmup_steps", self.warmup_steps.value())))
        self.eval_interval.setValue(int(training.get("eval_interval", self.eval_interval.value())))
        self.max_eval_batches.setValue(int(training.get("max_eval_batches", self.max_eval_batches.value())))
        self.save_interval.setValue(int(training.get("save_interval", self.save_interval.value())))
        self.data_loader_workers.setValue(int(training.get("data_loader_workers", self.data_loader_workers.value())))
        self.max_grad_norm.setValue(float(training.get("max_grad_norm", self.max_grad_norm.value())))
        self.seed.setValue(int(training.get("seed", self.seed.value())))
        self._set_combo_text(self.device, str(training.get("device", self.device.currentText())))
        self.use_amp.setChecked(bool(training.get("use_amp", self.use_amp.isChecked())))
        self._set_combo_by_data(self.precision, str(training.get("precision", "fp16")), {
            "fp16": "FP16",
            "bf16": "BF16",
            "fp32": "FP32",
        })
        self.resume_training.setChecked(bool(training.get("resume", self.resume_training.isChecked())))
        self.resume_safety.setChecked(bool(training.get("require_compatible_resume", True)))
        self.benchmark_prompts.setPlainText(str(training.get("benchmark_prompts", self.benchmark_prompts.toPlainText())))
        self.benchmark_tokens.setValue(int(training.get("benchmark_tokens", self.benchmark_tokens.value())))
        self.benchmark_temperature.setValue(float(training.get("benchmark_temperature", self.benchmark_temperature.value())))
        self.benchmark_kv_cache.setChecked(bool(training.get("benchmark_kv_cache", True)))

        self._set_combo_text(self.quant_mode, str(export.get("quantization", self.quant_mode.currentText())))
        self._set_combo_text(self.gguf_outtype, str(export.get("gguf_outtype", self.gguf_outtype.currentText())))
        self.llama_context.setValue(int(chat.get("context", self.llama_context.value())))
        self._set_combo_by_data(self.chat_model_backend, str(chat.get("model_backend", "gguf")), {
            "gguf": "GGUF / llama.cpp",
            "microgpt": "MicroGPT checkpoint",
        })
        self.llama_threads.setValue(int(chat.get("cpu_threads", self.llama_threads.value())))
        self.llama_gpu_layers.setValue(int(chat.get("gpu_layers", self.llama_gpu_layers.value())))
        self.thinking_enabled.setChecked(bool(chat.get("thinking_enabled", True)))
        self._set_combo_text(self.reasoning_effort, str(chat.get("reasoning_effort", self.reasoning_effort.currentText())))
        self.reasoning_effort.setEnabled(self.thinking_enabled.isChecked())
        self.chat_max_tokens.setValue(int(chat.get("max_tokens", self.chat_max_tokens.value())))
        self.chat_temperature.setValue(float(chat.get("temperature", self.chat_temperature.value())))
        self.chat_top_p.setValue(float(chat.get("top_p", self.chat_top_p.value())))
        self.chat_repeat_penalty.setValue(float(chat.get("repeat_penalty", self.chat_repeat_penalty.value())))
        self.system_prompt.setPlainText(str(chat.get("system_prompt", "")))
        if hasattr(self, "coordinator_host"):
            self.coordinator_host.setText(str(distributed.get("host", self.coordinator_host.text())))
            self.coordinator_port.setValue(int(distributed.get("port", self.coordinator_port.value())))
            self.coordinator_artifact_root.setText(str(distributed.get("artifact_root", self.coordinator_artifact_root.text())))
            self.coordinator_public_url.setText(str(distributed.get("public_url", self.coordinator_public_url.text())))
        self._update_tokenizer_strategy_controls()
        self._update_training_mode_controls()
        self._restore_artifact_status(data.get("artifacts", {}))
        self.refresh_fine_tune_workflow()

    def _restore_artifact_status(self, artifacts: dict[str, Any]) -> None:
        """Refresh top-bar and button state from saved or existing artifacts.

        Args:
            artifacts: Saved artifact summary dictionary.
        """

        dataset_dir = Path(self.dataset_dir.text()) if self.dataset_dir.text().strip() else None
        if dataset_dir and self._dataset_artifacts_exist(dataset_dir):
            summary = self._read_json_if_exists(dataset_dir / "dataset_summary.json") or artifacts.get("dataset_summary") or {}
            document_count = int(summary.get("document_count", 0) or 0)
            token_count = int(summary.get("token_count", 0) or 0)
            code_count = int(summary.get("code_sample_count", 0) or 0)
            prose_count = int(summary.get("prose_sample_count", 0) or 0)
            conversation_count = int(summary.get("conversation_sample_count", 0) or 0)
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            self._update_dataset_quality_report(summary)
            self.prepare_button.setText("DataSet Prepared")
            self.dataset_progress.setValue(100)
            if vocab_size:
                self.auto_vocab_label.setText(f"{vocab_size:,}")
            if code_count or prose_count or conversation_count:
                self.dataset_status.setText(
                    f"Dataset: {code_count:,} code, {prose_count:,} prose, {conversation_count:,} chat, {token_count:,} tokens"
                )
            elif document_count or token_count:
                self.dataset_status.setText(f"Dataset: {document_count:,} files, {token_count:,} tokens")
            else:
                self.dataset_status.setText("Dataset: prepared")
            version = summary.get("dataset_version", {})
            if isinstance(version, dict) and version.get("version_id"):
                self.dataset_log.append(f"Dataset version: {version['version_id']}")
            self.train_data_dir.setText(str(dataset_dir))
            self.dataset_log.append(f"Dataset already prepared: {dataset_dir}")
        else:
            self.prepare_button.setText("Prepare Dataset")
            self.dataset_progress.setValue(0)
            self.dataset_status.setText("Dataset: not prepared")
            self.auto_vocab_label.setText("Auto after reading files")
            self._reset_dataset_quality_report()

        model_dir = Path(self.model_dir.text()) if self.model_dir.text().strip() else None
        if model_dir and (model_dir / "final_model.pt").exists():
            summary = self._read_json_if_exists(model_dir / "training_summary.json") or artifacts.get("training_summary") or {}
            loss = summary.get("final_train_loss")
            self.train_status.setText(f"Training: loss {float(loss):.4f}" if loss is not None else "Training: model ready")
            self.export_model_dir.setText(str(model_dir))

        export_dir = Path(self.export_dir.text()) if self.export_dir.text().strip() else None
        if export_dir and export_dir.exists() and any(export_dir.iterdir()):
            self.export_status.setText("Export: artifacts found")

    @staticmethod
    def _dataset_artifacts_exist(dataset_dir: Path) -> bool:
        """Return whether a dataset folder has the required prepared files.

        Args:
            dataset_dir: Dataset folder.

        Returns:
            True if required dataset artifacts exist.
        """

        required = ("tokenizer.json", "train_tokens.json", "val_tokens.json")
        return dataset_dir.exists() and all((dataset_dir / name).exists() for name in required)

    @staticmethod
    def _safe_project_name(project_name: str) -> str:
        """Return a filesystem-safe project folder name.

        Args:
            project_name: Raw user project name.

        Returns:
            Safe folder name.
        """

        return re.sub(r"[^A-Za-z0-9_.-]+", "_", project_name).strip("._") or "MicroLLMProject"

    @staticmethod
    def _read_json_if_exists(path: Path) -> Optional[Any]:
        """Read a JSON file when it exists.

        Args:
            path: JSON file path.

        Returns:
            Parsed JSON or ``None``.
        """

        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        """Set combo text when the value exists.

        Args:
            combo: Combo box to update.
            text: Display text to select.
        """

        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_combo_by_data(self, combo: QComboBox, value: str, labels: dict[str, str]) -> None:
        """Set a combo by internal saved value.

        Args:
            combo: Combo box to update.
            value: Internal saved value.
            labels: Mapping from saved value to display label.
        """

        self._set_combo_text(combo, labels.get(value, value))

    def _run_task(
        self,
        fn,
        args,
        on_finished,
        log: QTextEdit,
        progress_bar: QProgressBar,
        with_progress: bool = False,
        button: Optional[QPushButton] = None,
        stop_button: Optional[QPushButton] = None,
        busy_text: str = "Working",
    ) -> None:
        """Run a long task on a background thread.

        Args:
            fn: Callable to execute.
            args: Positional arguments for the callable.
            on_finished: Slot called with the task result.
            log: Log widget receiving progress messages.
            progress_bar: Progress bar receiving percent updates.
            with_progress: Whether to pass a progress callback to the task.
            button: Optional button to disable while running.
            stop_button: Optional stop button to enable while running.
            busy_text: Button text shown while running.
        """

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return

        LOGGER.info("Starting background task: %s", getattr(fn, "__name__", str(fn)))
        if button:
            self._set_button_busy(button, busy_text)
        if stop_button:
            stop_button.setEnabled(True)
            self.active_stop_button = stop_button

        self.stop_event = Event()
        self.progress_queue = Queue()
        self.active_log = log
        self.active_progress_bar = progress_bar
        self.thread = QThread(self)
        self.worker = TaskWorker(
            fn,
            *args,
            progress_queue=self.progress_queue,
            with_progress=with_progress,
            stop_event=self.stop_event,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(on_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self._task_failed_from_worker)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.progress_timer.start(100)
        self.thread.start()

    @Slot(str)
    def _task_failed_from_worker(self, message: str) -> None:
        """Handle a worker failure on the UI thread.

        Args:
            message: Error message emitted by the worker.
        """

        if self.active_log is None or self.active_progress_bar is None:
            return
        LOGGER.error("Background task failed: %s", message)
        self._task_failed(message, self.active_log, self.active_progress_bar)

    def stop_active_task(self) -> None:
        """Request a graceful stop for the active background task."""

        if self.stop_event is None:
            return
        LOGGER.info("Stop requested for active background task")
        self.stop_event.set()
        if self.active_log is not None:
            self.active_log.append("Stop requested. Finishing the current safe point...")
        if self.active_stop_button is not None:
            self.active_stop_button.setEnabled(False)

    @Slot()
    def request_shutdown_from_signal(self) -> None:
        """Handle Ctrl+C from a terminal without leaving Qt threads wedged."""

        self.interrupt_count += 1
        if self.interrupt_count > 1:
            os._exit(130)
        if self.stop_event is not None:
            self.stop_event.set()
        if self.active_log is not None:
            self.active_log.append("Interrupt received. Requesting stop...")
        self.project_state.setText("Stopping")
        if self.thread is None:
            QApplication.quit()
            return
        QTimer.singleShot(3000, lambda: os._exit(130) if self.thread is not None else QApplication.quit())

    def closeEvent(self, event: Any) -> None:
        """Clean up background services before the window closes.

        Args:
            event: Qt close event.
        """

        if self.coordinator_server is not None:
            self.stop_coordinator_server()
        super().closeEvent(event)

    def _handle_progress(self, event: object, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Apply one progress event to UI widgets.

        Args:
            event: Progress dictionary or message.
            log: Log widget to append messages to.
            progress_bar: Progress bar to update.
        """

        if isinstance(event, dict):
            if event.get("type") == "chat_delta":
                self._apply_chat_delta(event)
                return
            message = event.get("message")
            percent = event.get("percent")
            if log is self.training_log:
                self._update_training_metrics(event)
            if message:
                log.append(str(message))
                if log is self.training_log and hasattr(self, "live_log"):
                    self.live_log.append(str(message))
            if percent is not None:
                progress_bar.setValue(max(0, min(100, int(percent))))
                if log is self.training_log and hasattr(self, "live_progress"):
                    self.live_progress.setValue(max(0, min(100, int(percent))))
        else:
            log.append(str(event))

    def _update_training_metrics(self, event: dict[str, Any]) -> None:
        """Update training metric chips from a progress event.

        Args:
            event: Progress event emitted by the training backend.
        """

        if "epoch" in event and "total_epochs" in event:
            self.training_epoch_metric.setText(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if "step" in event and "total_steps" in event:
            self.training_step_metric.setText(f"Step: {event['step']}/{event['total_steps']}")
        train_loss = event.get("train_loss")
        if train_loss is not None:
            self.training_loss_metric.setText(f"Train loss: {float(train_loss):.4f}")
        val_loss = event.get("val_loss")
        if val_loss is not None:
            self.training_val_metric.setText(f"Val loss: {float(val_loss):.4f}")
        step = event.get("step")
        if step is not None and (train_loss is not None or val_loss is not None):
            self.loss_chart.add_metrics(int(step), train_loss, val_loss)
        if step is None:
            return
        step_int = int(step)
        self._record_live_metric(event)
        learning_rate = event.get("learning_rate")
        grad_norm = event.get("grad_norm")
        weight_norm = event.get("weight_norm")
        update_ratio = event.get("update_ratio")
        tokens_per_second = event.get("tokens_per_second")
        samples_per_second = event.get("samples_per_second")
        vram_allocated = event.get("vram_allocated_gb")
        vram_reserved = event.get("vram_reserved_gb")
        gpu_memory = event.get("gpu_memory_percent")
        system_cpu = event.get("system_cpu_percent")
        system_ram = event.get("system_ram_percent")
        data_workers = event.get("data_loader_workers")
        eta_seconds = event.get("eta_seconds")
        if learning_rate is not None:
            self.training_lr_metric.setText(f"LR: {float(learning_rate):.2e}")
        if grad_norm is not None:
            self.training_grad_metric.setText(f"Grad: {float(grad_norm):.3f}")
        if tokens_per_second is not None:
            self.training_speed_metric.setText(f"Speed: {float(tokens_per_second):.0f} tok/s")
        if vram_allocated is not None:
            self.training_vram_metric.setText(f"VRAM: {float(vram_allocated):.2f} GB")
        if eta_seconds is not None:
            self.training_eta_metric.setText(f"ETA: {self._format_duration(float(eta_seconds))}")
        if learning_rate is not None or grad_norm is not None:
            self.optimization_chart.add_values(step_int, learning_rate, grad_norm)
        if weight_norm is not None or update_ratio is not None:
            self.stability_chart.add_values(step_int, weight_norm, update_ratio)
        if tokens_per_second is not None or samples_per_second is not None:
            self.throughput_chart.add_values(step_int, tokens_per_second, samples_per_second)
        if vram_allocated is not None or vram_reserved is not None:
            self.memory_chart.add_values(step_int, vram_allocated, vram_reserved)
        if hasattr(self, "live_epoch_metric"):
            self._update_live_training_metrics(
                step_int,
                event,
                train_loss,
                learning_rate,
                grad_norm,
                update_ratio,
                tokens_per_second,
                samples_per_second,
                vram_allocated,
                vram_reserved,
                gpu_memory,
                system_cpu,
                system_ram,
                data_workers,
            )

    def _update_live_training_metrics(
        self,
        step: int,
        event: dict[str, Any],
        train_loss: Optional[float],
        learning_rate: Optional[float],
        grad_norm: Optional[float],
        update_ratio: Optional[float],
        tokens_per_second: Optional[float],
        samples_per_second: Optional[float],
        vram_allocated: Optional[float],
        vram_reserved: Optional[float],
        gpu_memory: Optional[float],
        system_cpu: Optional[float],
        system_ram: Optional[float],
        data_workers: Optional[int],
    ) -> None:
        """Update live tracker widgets from one training progress event.

        Args:
            step: Current optimizer step.
            event: Progress event emitted by training.
            train_loss: Latest training loss.
            learning_rate: Current learning rate.
            grad_norm: Current gradient norm.
            update_ratio: Current parameter update ratio.
            tokens_per_second: Current token throughput.
            samples_per_second: Current sample throughput.
            vram_allocated: Current CUDA allocated memory in GB.
            vram_reserved: Current CUDA reserved memory in GB.
            gpu_memory: Current GPU memory pressure percentage.
            system_cpu: Current system CPU utilization percentage.
            system_ram: Current system RAM utilization percentage.
            data_workers: CPU data-loader worker count.
        """

        total_steps = event.get("total_steps")
        if "epoch" in event and "total_epochs" in event:
            self.live_epoch_metric.setText(f"Epoch: {event['epoch']}/{event['total_epochs']}")
        if total_steps:
            self.live_step_metric.setText(f"Step: {step:,}/{int(total_steps):,}")
            data_percent = min(100.0, max(0.0, (step / max(1, int(total_steps))) * 100.0))
            self.live_data_metric.setText(f"Data: {data_percent:.1f}%")
            self.live_progress.setValue(int(data_percent))
        else:
            self.live_step_metric.setText(f"Step: {step:,}")
        if tokens_per_second is not None:
            self.live_tokens_metric.setText(f"Tokens/sec: {float(tokens_per_second):,.0f}")
        if train_loss is not None:
            self.live_loss_metric.setText(f"Loss: {float(train_loss):.4f}")
        if learning_rate is not None:
            self.live_lr_metric.setText(f"LR: {float(learning_rate):.2e}")
        sample_text = str(event.get("sample_text") or "").strip()
        if sample_text:
            self.live_sample_text.setText(f"Training text: {self._compact_preview_text(sample_text, 220)}")
        self.live_layer_status.setText(f"▣ Layers: {self.n_layer.value()}")
        self.live_head_status.setText(f"◎ Heads: {self.n_head.value()}")
        self.live_hidden_status.setText(f"▤ Hidden size: {self.n_embd.value()}")
        self.live_batch_status.setText(f"▥ Batch size: {self.batch_size.value()}")
        self.live_context_status.setText(f"▢ Context: {self.train_context_length.value()}")
        self.live_device_status.setText(f"Device: {self.device.currentText()}")
        self.live_worker_status.setText(f"CPU workers: {data_workers if data_workers is not None else self.data_loader_workers.value()}")
        self._set_meter(self.live_cpu_bar, "CPU", system_cpu if system_cpu is not None else self._system_cpu_value())
        self._set_meter(self.live_gpu_bar, "GPU memory", gpu_memory)
        if vram_allocated is not None or vram_reserved is not None:
            allocated = float(vram_allocated or 0.0)
            reserved = float(vram_reserved or 0.0)
            reserved_percent = None
            if self.device.currentText().startswith("cuda") and torch.cuda.is_available():
                try:
                    _, total_vram = torch.cuda.mem_get_info()
                    reserved_percent = min(100.0, 100.0 * reserved * (1024 ** 3) / max(total_vram, 1))
                except Exception:
                    reserved_percent = None
            self._set_meter(self.live_vram_bar, "VRAM reserved", reserved_percent)
            self.live_vram_label.setText(f"VRAM reserved: {reserved:.2f} GB ({allocated:.2f} GB active)")
        self._set_meter(self.live_ram_bar, "System RAM", system_ram if system_ram is not None else self._system_ram_value())
        latest_loss = float(train_loss) if train_loss is not None else None
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), step, latest_loss)
        self.live_prediction_chart.update_distribution(step, latest_loss)
        self.live_attention_chart.update_heatmap(step, grad_norm)
        self.live_activation_chart.update_histogram(step, tokens_per_second)
        self.live_gradient_chart.update_flow(self.n_layer.value(), grad_norm, step)

    def _system_ram_value(self) -> Optional[float]:
        """Read system RAM utilization for live telemetry.

        Returns:
            System RAM percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.virtual_memory().percent)

    def _system_cpu_value(self) -> Optional[float]:
        """Read system CPU utilization for live telemetry.

        Returns:
            System CPU percentage, or None when unavailable.
        """

        if psutil is None:
            return None
        return float(psutil.cpu_percent(interval=None))

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format a duration for compact UI display.

        Args:
            seconds: Duration in seconds.

        Returns:
            Human-readable compact duration.
        """

        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def _apply_chat_delta(self, event: dict[str, Any]) -> None:
        """Apply one streamed chat chunk to the rendered conversation.

        Args:
            event: Chat stream progress event.
        """

        self.chat_stream_reply += str(event.get("content", ""))
        should_follow = self._is_chat_near_bottom()
        self._render_chat_markdown(self.chat_stream_reply)
        if should_follow:
            self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum())
        self._set_chat_stats(
            float(event.get("elapsed_seconds", 0.0)),
            int(event.get("token_count", 0)),
            float(event.get("tokens_per_second", 0.0)),
        )

    def _drain_progress_queue(self) -> None:
        """Drain queued worker progress events on the UI thread."""

        if self.progress_queue is None or self.active_log is None or self.active_progress_bar is None:
            return
        drained = 0
        last_percent = None
        while drained < 50:
            try:
                event = self.progress_queue.get_nowait()
            except Empty:
                break
            if isinstance(event, dict) and event.get("percent") is not None:
                last_percent = event.get("percent")
                event = {**event, "percent": None}
            self._handle_progress(event, self.active_log, self.active_progress_bar)
            drained += 1
        if last_percent is not None:
            self.active_progress_bar.setValue(max(0, min(100, int(last_percent))))

    def _thread_finished(self) -> None:
        """Clean up thread bookkeeping after a worker finishes."""

        LOGGER.info("Background task thread finished")
        self._drain_progress_queue()
        if self.progress_timer.isActive():
            self.progress_timer.stop()
        self.thread = None
        self.worker = None
        self.stop_event = None
        self.progress_queue = None
        self.active_log = None
        self.active_progress_bar = None
        if self.active_stop_button is not None:
            self.active_stop_button.setEnabled(False)
        self.active_stop_button = None
        if self.active_button is not None:
            self._clear_button_busy()

    def _task_failed(self, message: str, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Handle background task failure.

        Args:
            message: Error message.
            log: Log widget to append to.
            progress_bar: Progress bar to reset.
        """

        stopped_by_user = "stopped by user" in message.lower()
        if stopped_by_user:
            LOGGER.info("Background task stopped by user: %s", message)
        else:
            LOGGER.error("Background task error: %s", message)
        log.append(f"Stopped: {message}" if stopped_by_user else f"Error: {message}")
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        if stopped_by_user:
            self.project_state.setText("Stopped")
        self._clear_button_busy()

    def _set_button_busy(self, button: QPushButton, text: str) -> None:
        """Disable a button and start its spinner text.

        Args:
            button: Button to mark busy.
            text: Busy label.
        """

        self.active_button = button
        self.active_button_text = text
        self.active_button_restore_text = button.text()
        self.spinner_index = 0
        button.setEnabled(False)
        button.setText(f"| {text}")
        self.spinner_timer.start(150)

    def _clear_button_busy(self, final_text: Optional[str] = None) -> None:
        """Restore the active busy button.

        Args:
            final_text: Optional final button text.
        """

        if self.spinner_timer.isActive():
            self.spinner_timer.stop()
        if self.active_button:
            self.active_button.setEnabled(True)
            self.active_button.setText(final_text or self.active_button_restore_text)
        if self.active_stop_button:
            self.active_stop_button.setEnabled(False)
        self.active_button = None
        self.active_button_text = ""
        self.active_button_restore_text = ""

    def _tick_spinner(self) -> None:
        """Advance the active button spinner frame."""

        if not self.active_button:
            return
        frames = "|/-\\"
        self.spinner_index = (self.spinner_index + 1) % len(frames)
        self.active_button.setText(f"{frames[self.spinner_index]} {self.active_button_text}")

    def _dataset_config_from_ui(self) -> DatasetConfig:
        """Collect dataset options from the current UI controls.

        Returns:
            Dataset preparation configuration.
        """

        return DatasetConfig(
            input_dir=Path(self.input_dir.text()),
            output_dir=Path(self.dataset_dir.text()),
            vocab_size=None if self.auto_vocab.isChecked() else self.manual_vocab_size.value(),
            conversation_datasets=self._selected_conversation_datasets(),
            conversation_sample_limit=self.conversation_sample_limit.value(),
            mixture_weights=self._mixture_weights_from_ui(),
            min_frequency=self.min_frequency.value(),
            context_length=self.context_length.value(),
            validation_split=self.validation_split.value(),
            lowercase=self.lowercase.isChecked(),
            max_workers=self.max_workers.value(),
            code_training_mode=self.code_training_mode.isChecked(),
            include_prose=self.include_prose.isChecked(),
            include_source_code=self.include_source_code.isChecked(),
            extract_code_blocks=self.extract_code_blocks.isChecked(),
            preserve_indentation=self.preserve_indentation.isChecked(),
            generate_instruction_samples=self.instruction_samples.isChecked(),
            reasoning_sample_mode=self._reasoning_sample_mode_value(),
            prepare_mode=self._prepare_mode_value(),
            tokenizer_strategy=self._tokenizer_strategy_value(),
            tokenizer_path=Path(self.tokenizer_path.text()) if self.tokenizer_path.text().strip() else None,
            dataset_stage=self._dataset_stage_value(),
        )

    def check_project_health(self) -> None:
        """Run a project health check in the background."""

        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self.dataset_log.append("Checking project health...")
        self.project_state.setText("Checking health")
        self._run_task(
            check_project_health,
            (
                Path(self.input_dir.text()),
                Path(self.dataset_dir.text()),
                Path(self.model_dir.text()),
                Path(self.export_dir.text()),
                Path(self.gguf_path.text()) if self.gguf_path.text().strip() else None,
                Path(self.llama_cpp_dir.text()) if self.llama_cpp_dir.text().strip() else None,
                self.device.currentText(),
            ),
            self._health_check_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.health_check_button,
            stop_button=self.stop_dataset_button,
            busy_text="Checking Health",
        )

    @Slot(object)
    def _health_check_finished(self, result: Any) -> None:
        """Display project health check results.

        Args:
            result: Project health result.
        """

        self.dataset_progress.setValue(100)
        self.dataset_log.append("")
        self.dataset_log.append(f"Project health: {result.status.upper()} ({result.summary})")
        for check in result.checks:
            marker = {"ok": "OK", "warning": "WARN", "error": "ERROR"}.get(check.get("status"), "INFO")
            self.dataset_log.append(f"[{marker}] {check.get('name')}: {check.get('detail')}")
        self.project_state.setText("Health checked")
        self._clear_button_busy("Check Health")

    def preview_dataset(self) -> None:
        """Run a dataset preview and quality scan in the background."""

        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self.dataset_log.append("Previewing dataset...")
        self.project_state.setText("Previewing dataset")
        self._run_task(
            scan_dataset_preview,
            (self._dataset_config_from_ui(),),
            self._dataset_preview_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.preview_dataset_button,
            stop_button=self.stop_dataset_button,
            busy_text="Previewing Dataset",
        )

    @Slot(object)
    def _dataset_preview_finished(self, result: Any) -> None:
        """Display dataset preview and quality scan results.

        Args:
            result: Dataset preview result.
        """

        self.dataset_progress.setValue(100)
        suffix_text = ", ".join(f"{suffix}: {count}" for suffix, count in result.suffix_counts.items()) or "none"
        self.dataset_log.append("")
        self.dataset_log.append(f"Source files: {result.source_file_count:,}; size: {result.total_bytes / (1024 * 1024):.2f} MB")
        self.dataset_log.append(f"File types: {suffix_text}")
        self.dataset_log.append(f"Prepared dataset artifacts: {'found' if result.prepared else 'not complete'}")
        self.dataset_log.append(f"Duplicate scan: {result.duplicate_count:,} file entries in {len(result.duplicate_groups):,} likely group(s).")
        self.dataset_log.append(f"Bad extraction scan: {result.bad_extraction_count:,} suspicious file(s).")
        self.dataset_log.append(f"Code/prose balance: {result.balance_label} ({result.code_preview_count:,}/{result.prose_preview_count:,}).")
        self.dataset_log.append(f"Training readiness: {result.readiness_label} ({result.readiness_score}/100).")
        for reason in result.readiness_reasons[:8]:
            self.dataset_log.append(f"- {reason}")
        self.dataset_quality_duplicates.setText(f"Duplicates: {result.duplicate_count:,}")
        self.dataset_quality_extraction.setText(f"Extraction: {result.bad_extraction_count:,} flagged")
        self.dataset_quality_balance.setText(f"Balance: {result.balance_label}")
        self.dataset_quality_readiness.setText(f"Readiness: {result.readiness_label} {result.readiness_score}/100")
        if result.summary:
            self._update_dataset_quality_report(result.summary)
            self.dataset_quality_duplicates.setText(f"Duplicates: {result.duplicate_count:,}")
            self.dataset_quality_extraction.setText(f"Extraction: {result.bad_extraction_count:,} flagged")
            self.dataset_quality_balance.setText(f"Balance: {result.balance_label}")
            self.dataset_quality_readiness.setText(f"Readiness: {result.readiness_label} {result.readiness_score}/100")
            tokens = int(result.summary.get("token_count", 0) or 0)
            vocab = int(result.summary.get("tokenizer_vocab_size", 0) or 0)
            self.dataset_log.append(f"Prepared summary: {tokens:,} tokens, vocab {vocab:,}.")
        else:
            self.dataset_quality_samples.setText(f"Samples: {len(result.sample_previews):,} previewed")
            self.dataset_quality_tokens.setText("Tokens: not prepared")
            self.dataset_quality_vocab.setText("Vocab: not prepared")
            self.dataset_quality_code.setText(f"Code/prose: {result.code_preview_count:,}/{result.prose_preview_count:,}")
            self.dataset_quality_cache.setText(f"Files: {result.source_file_count:,} source")
        if result.duplicate_groups:
            self.dataset_log.append("")
            self.dataset_log.append("Likely duplicates:")
            for group in result.duplicate_groups[:8]:
                self.dataset_log.append(f"- {group.get('type')}: {group.get('count')} file(s)")
                for path in group.get("files", [])[:4]:
                    self.dataset_log.append(f"    {Path(path).name}")
        if result.bad_extraction_files:
            self.dataset_log.append("")
            self.dataset_log.append("Suspicious extraction files:")
            for item in result.bad_extraction_files[:12]:
                self.dataset_log.append(f"- {Path(item.get('path', '')).name}: {item.get('reasons')}")
        suggestions: list[str] = []
        if result.duplicate_groups:
            suggestions.append("Remove or move duplicate files before preparing the final dataset.")
        if result.bad_extraction_files:
            suggestions.append("Replace flagged PDFs with text/source versions, or remove files with bad extraction.")
        if result.balance_label == "Prose heavy" and self.code_training_mode.isChecked():
            suggestions.append("Add real source-code folders or enable source-file inclusion for a stronger coding model.")
        if result.balance_label == "Code heavy":
            suggestions.append("Add README/tutorial/prose explanations if you want the model to explain code well.")
        if result.readiness_label in {"Needs cleanup", "Not ready"}:
            suggestions.append("Run Preview Dataset again after cleanup and only train once readiness improves.")
        if hasattr(self, "dataset_advisor"):
            if suggestions:
                self.dataset_advisor.setPlainText("\n".join(f"- {suggestion}" for suggestion in suggestions))
            else:
                self.dataset_advisor.setPlainText("No immediate cleanup suggestions. Dataset looks acceptable for the current preview.")
        if suggestions:
            self.dataset_log.append("")
            self.dataset_log.append("Cleanup suggestions:")
            for suggestion in suggestions:
                self.dataset_log.append(f"- {suggestion}")
        if result.issues:
            self.dataset_quality_warning.setText(f"Warnings: {len(result.issues)}")
            self.dataset_log.append("")
            self.dataset_log.append("Quality notes:")
            for issue in result.issues[:12]:
                self.dataset_log.append(f"- {issue}")
        else:
            self.dataset_quality_warning.setText("Warnings: none")
        if result.sample_previews:
            self.dataset_log.append("")
            self.dataset_log.append("Preview samples:")
            for index, sample in enumerate(result.sample_previews, start=1):
                label = sample.get("language") or sample.get("kind") or "text"
                self.dataset_log.append(f"\n[{index}] {Path(sample.get('path', '')).name} ({label}, {sample.get('characters')} chars)")
                self.dataset_log.append(sample.get("preview", "").replace("\n", "\n    ")[:1400])
        self.project_state.setText("Dataset previewed")
        self._clear_button_busy("Preview Dataset")

    def prepare_dataset(self) -> None:
        """Collect dataset options and start dataset preparation."""

        config = self._dataset_config_from_ui()
        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self._reset_dataset_quality_report()
        self.dataset_log.append("Preparing dataset...")
        self.dataset_log.append(f"App log file: {self.log_file_path}")
        self.dataset_log.append(f"Dataset purpose: {dataset_stage_label(config.dataset_stage)}")
        if self.include_conversation_datasets.isChecked():
            selected_labels = [
                checkbox.text()
                for checkbox in getattr(self, "conversation_dataset_checks", {}).values()
                if checkbox.isChecked() and checkbox.isVisible()
            ]
            if selected_labels:
                hf_cache = config.output_dir / "cache" / "huggingface"
                self.dataset_log.append(f"Online training datasets: {', '.join(selected_labels)}")
                self.dataset_log.append(f"Downloading/loading online data at: {hf_cache}")
                LOGGER.info("Online training datasets: %s", ", ".join(selected_labels))
                LOGGER.info("Downloading/loading online data at: %s", hf_cache)
            else:
                self.dataset_log.append("Online training datasets are enabled, but no dataset is selected for this purpose.")
                LOGGER.warning("Online training datasets enabled, but no dataset is selected")
        else:
            self.dataset_log.append("Online training datasets: off. Local source files only.")
            LOGGER.info("Online training datasets: off. Local source files only.")
            checked_count = sum(
                1
                for checkbox in getattr(self, "conversation_dataset_checks", {}).values()
                if checkbox.isChecked()
            )
            if checked_count:
                self.dataset_log.append("Checked online dataset choices are ignored until the master checkbox is enabled.")
                LOGGER.info("Checked online dataset choices are ignored until the master checkbox is enabled")
        LOGGER.info(
            "Preparing dataset: input=%s output=%s stage=%s online_datasets=%s",
            config.input_dir,
            config.output_dir,
            config.dataset_stage,
            ",".join(config.conversation_datasets) or "off",
        )
        self.project_state.setText("Preparing dataset")
        self.dataset_status.setText("Dataset: preparing")
        self.auto_vocab_label.setText("Calculating...")
        self._run_task(
            build_dataset,
            (config,),
            self._dataset_finished,
            self.dataset_log,
            self.dataset_progress,
            with_progress=True,
            button=self.prepare_button,
            stop_button=self.stop_dataset_button,
            busy_text="Preparing Dataset",
        )

    @Slot(object)
    def _dataset_finished(self, result: Any) -> None:
        """Update UI after dataset preparation finishes.

        Args:
            result: Dataset build result.
        """

        self.dataset_progress.setValue(100)
        self.auto_vocab_label.setText(f"{result.vocab_size:,}")
        LOGGER.info(
            "Dataset prepared: documents=%s tokens=%s vocab=%s code=%s prose=%s conversation=%s output=%s",
            result.document_count,
            result.token_count,
            result.vocab_size,
            result.code_sample_count,
            result.prose_sample_count,
            getattr(result, "conversation_sample_count", 0),
            result.output_dir,
        )
        self.dataset_log.append(
            f"Prepared {result.document_count} documents, {result.character_count:,} characters, "
            f"{result.token_count:,} tokens, vocab {result.vocab_size:,}."
        )
        self.dataset_log.append(
            f"Cache summary: reused {result.cached_file_count:,} file(s), processed {result.processed_file_count:,} file(s)."
        )
        if getattr(result, "dataset_version_id", ""):
            self.dataset_log.append(f"Dataset version: {result.dataset_version_id}")
        mixture_report = getattr(result, "mixture_report", {}) or {}
        if mixture_report:
            self.dataset_log.append("Mixture report:")
            for family in ("local_prose", "source_code", "online_base", "instruction", "conversation"):
                row = mixture_report.get("families", {}).get(family, {})
                if not row:
                    continue
                requested = float(row.get("requested_weight", 0.0) or 0.0)
                selected = int(row.get("selected_documents", 0) or 0)
                available = int(row.get("available_documents", 0) or 0)
                dropped = int(row.get("dropped_documents", 0) or 0)
                actual = float(row.get("actual_percent", 0.0) or 0.0)
                if requested > 0.0 or selected > 0 or available > 0:
                    self.dataset_log.append(
                        f"- {row.get('label', family)}: requested {requested:.1f}%, "
                        f"selected {selected:,}/{available:,}, dropped {dropped:,}, actual {actual:.1f}%"
                    )
        if result.warning:
            self.dataset_log.append(f"Recommendation: {result.warning}")
        self._update_dataset_quality_report(
            {
                "document_count": result.document_count,
                "token_count": result.token_count,
                "character_count": result.character_count,
                "tokenizer_vocab_size": result.vocab_size,
                "code_sample_count": result.code_sample_count,
                "prose_sample_count": result.prose_sample_count,
                "conversation_sample_count": getattr(result, "conversation_sample_count", 0),
                "cached_file_count": result.cached_file_count,
                "processed_file_count": result.processed_file_count,
                "skipped_file_count": result.skipped_file_count,
                "failed_file_count": result.failed_file_count,
                "warning": result.warning,
                "mixture_report": mixture_report,
            }
        )
        self.train_data_dir.setText(str(result.output_dir))
        self.project_state.setText("Dataset ready")
        self.dataset_status.setText(f"Dataset: {result.document_count} files, {result.token_count:,} tokens")
        if result.code_sample_count:
            self.dataset_status.setText(
                f"Dataset: {result.code_sample_count:,} code, {result.prose_sample_count:,} prose, {result.token_count:,} tokens"
            )
        self.refresh_model_estimate()
        self.refresh_fine_tune_workflow()
        self._clear_button_busy("DataSet Prepared")

    def _prepare_mode_value(self) -> str:
        """Return the selected dataset preparation mode.

        Returns:
            Internal mode value.
        """

        label = self.prepare_mode.currentText()
        if label == "Full rebuild":
            return "full_rebuild"
        if label == "Force reprocess":
            return "force_reprocess"
        return "incremental"

    def _tokenizer_strategy_value(self) -> str:
        """Return the selected tokenizer strategy.

        Returns:
            Internal tokenizer strategy value.
        """

        label = self.tokenizer_strategy.currentText()
        if label == "Train new tokenizer":
            return "train_new"
        if label == "Reuse dataset tokenizer":
            return "reuse_dataset"
        if label == "Import tokenizer.json":
            return "import_tokenizer"
        return "auto"

    def _reasoning_sample_mode_value(self) -> str:
        """Return the selected reasoning sample mode.

        Returns:
            Internal reasoning sample mode.
        """

        label = self.reasoning_sample_mode.currentText()
        if label == "Detailed code reasoning":
            return "detailed"
        if label == "No reasoning wrapper":
            return "none"
        return "scaffold"

    def _dataset_stage_value(self) -> str:
        """Return the selected dataset preparation stage.

        Returns:
            Dataset stage identifier.
        """

        return {
            "Base pretraining": "base",
            "Instruction fine-tune": "instruction",
            "Conversation fine-tune": "conversation",
        }.get(self.dataset_stage.currentText(), "base")

    def _set_dataset_stage(self, stage: str) -> None:
        """Set the dataset stage combo from an internal stage value.

        Args:
            stage: Dataset stage identifier.
        """

        self._set_combo_by_data(
            self.dataset_stage,
            stage,
            {
                "base": "Base pretraining",
                "instruction": "Instruction fine-tune",
                "conversation": "Conversation fine-tune",
            },
        )
        self._update_online_dataset_stage_controls()

    def _update_online_dataset_stage_controls(self) -> None:
        """Show and enable online datasets for the selected training stage."""

        if not hasattr(self, "dataset_stage"):
            return
        stage = self._dataset_stage_value()
        allowed = set(dataset_ids_for_stage(stage))
        include_online = self.include_conversation_datasets.isChecked()
        for dataset_id, checkbox in getattr(self, "conversation_dataset_checks", {}).items():
            visible = dataset_id in allowed
            checkbox.setVisible(visible)
            checkbox.setEnabled(include_online and visible)
        self.conversation_sample_limit.setEnabled(include_online)
        stage_name = dataset_stage_label(stage)
        if include_online:
            self.conversation_datasets_status.setText(f"{stage_name}: choose dataset checkboxes below to download/read.")
        elif stage == "base":
            self.conversation_datasets_status.setText("Base pretraining: choose optional online corpus datasets, or use local files only.")
        elif stage == "instruction":
            self.conversation_datasets_status.setText("Instruction fine-tune: Alpaca/Dolly/SlimOrca are available here. TinyStories is hidden.")
        else:
            self.conversation_datasets_status.setText("Conversation fine-tune: chat datasets are available here. TinyStories is hidden.")

    def _selected_conversation_datasets(self) -> list[str]:
        """Return selected built-in conversation dataset IDs.

        Returns:
            Selected dataset identifiers.
        """

        allowed = set(dataset_ids_for_stage(self._dataset_stage_value()))
        return [
            dataset_id
            for dataset_id, checkbox in getattr(self, "conversation_dataset_checks", {}).items()
            if dataset_id in allowed and checkbox.isChecked() and self.include_conversation_datasets.isChecked()
        ]

    def _set_selected_conversation_datasets(self, dataset_ids: list[str]) -> None:
        """Restore selected conversation dataset checkboxes.

        Args:
            dataset_ids: Dataset IDs to select.
        """

        selected = set(dataset_ids)
        for dataset_id, checkbox in getattr(self, "conversation_dataset_checks", {}).items():
            checkbox.setChecked(dataset_id in selected)
            checkbox.setEnabled(self.include_conversation_datasets.isChecked() and dataset_id in dataset_ids_for_stage(self._dataset_stage_value()))
        if hasattr(self, "conversation_sample_limit"):
            self.conversation_sample_limit.setEnabled(self.include_conversation_datasets.isChecked())
        if hasattr(self, "conversation_datasets_status"):
            self._update_online_dataset_stage_controls()

    def _mixture_weights_from_ui(self) -> dict[str, float]:
        """Return dataset mixture weights from the Ingest tab.

        Returns:
            Mapping from mixture source family to percentage.
        """

        if not hasattr(self, "mixture_local_prose"):
            return {}
        return {
            "local_prose": float(self.mixture_local_prose.value()),
            "source_code": float(self.mixture_source_code.value()),
            "online_base": float(self.mixture_online_base.value()),
            "instruction": float(self.mixture_instruction.value()),
            "conversation": float(self.mixture_conversation.value()),
        }

    def _set_mixture_weights(self, weights: dict[str, Any]) -> None:
        """Restore dataset mixture weights.

        Args:
            weights: Saved mixture weights by source family.
        """

        if not hasattr(self, "mixture_local_prose"):
            return
        defaults = {
            "local_prose": 50.0,
            "source_code": 30.0,
            "online_base": 20.0,
            "instruction": 0.0,
            "conversation": 0.0,
        }
        values = {**defaults, **(weights or {})}
        widgets = {
            "local_prose": self.mixture_local_prose,
            "source_code": self.mixture_source_code,
            "online_base": self.mixture_online_base,
            "instruction": self.mixture_instruction,
            "conversation": self.mixture_conversation,
        }
        for key, widget in widgets.items():
            try:
                widget.setValue(float(values.get(key, defaults[key])))
            except (TypeError, ValueError):
                widget.setValue(defaults[key])
        self._update_mixture_total()

    def _update_mixture_total(self) -> None:
        """Refresh the visible dataset mixture total."""

        if not hasattr(self, "mixture_total_label"):
            return
        total = sum(self._mixture_weights_from_ui().values())
        self.mixture_total_label.setText(f"Total: {total:.1f}%")
        self.mixture_total_label.setProperty("state", "ok" if abs(total - 100.0) <= 0.1 else "warning")
        self.mixture_total_label.style().unpolish(self.mixture_total_label)
        self.mixture_total_label.style().polish(self.mixture_total_label)

    def _normalize_mixture_weights(self) -> None:
        """Scale dataset mixture values so they total 100 percent."""

        weights = self._mixture_weights_from_ui()
        total = sum(weights.values())
        if total <= 0:
            self._set_mixture_weights({"local_prose": 100.0})
            return
        self._set_mixture_weights({key: value * 100.0 / total for key, value in weights.items()})

    def _training_launch_target_value(self) -> str:
        """Return whether training should launch locally or remotely.

        Returns:
            ``local`` or ``remote``.
        """

        return "remote" if self.training_launch_target.currentText() == "Remote workers" else "local"

    def _architecture_style_config(self) -> dict[str, Any]:
        """Return ModelConfig keyword arguments for the selected block style.

        Returns:
            Architecture style settings.
        """

        if self.architecture_style.currentText() == "Llama-like":
            return {
                "norm_type": "rmsnorm",
                "position_encoding": "rope",
                "mlp_type": "swiglu",
                "rope_theta": 10000.0,
            }
        return {
            "norm_type": "layernorm",
            "position_encoding": "learned",
            "mlp_type": "gelu",
            "rope_theta": 10000.0,
        }

    def _optimizer_value(self) -> str:
        """Return the selected optimizer identifier.

        Returns:
            Stable optimizer name used by the trainer.
        """

        return {
            "AdamW": "adamw",
            "Adam": "adam",
            "Lion": "lion",
            "Adafactor": "adafactor",
        }.get(self.optimizer_name.currentText(), "adamw")

    def _scheduler_value(self) -> str:
        """Return the selected scheduler identifier.

        Returns:
            Stable scheduler name used by the trainer.
        """

        return {
            "Warmup linear": "warmup_linear",
            "Cosine decay": "cosine",
            "Polynomial decay": "polynomial",
            "One-cycle": "one_cycle",
            "Constant": "constant",
        }.get(self.scheduler_name.currentText(), "warmup_linear")

    def _precision_value(self) -> str:
        """Return the selected numeric precision identifier.

        Returns:
            Stable precision name used by the trainer.
        """

        return {
            "FP16": "fp16",
            "BF16": "bf16",
            "FP32": "fp32",
        }.get(self.precision.currentText(), "fp16")

    def _training_mode_value(self) -> str:
        """Return the selected training mode identifier.

        Returns:
            Stable training mode used by the trainer.
        """

        return {
            "Pretrain from scratch": "pretrain",
            "Fine-tune checkpoint": "fine_tune",
            "Instruction fine-tune": "fine_tune",
            "Conversation fine-tune": "fine_tune",
        }.get(self.training_mode.currentText(), "pretrain")

    def _training_stage_value(self) -> str:
        """Return the higher-level training stage selected in the UI.

        Returns:
            Training stage identifier.
        """

        return {
            "Pretrain from scratch": "base",
            "Fine-tune checkpoint": "domain",
            "Instruction fine-tune": "instruction",
            "Conversation fine-tune": "conversation",
        }.get(self.training_mode.currentText(), "base")

    def _peft_method_value(self) -> str:
        """Return the selected PEFT method identifier.

        Returns:
            Stable PEFT method used by the trainer.
        """

        return {
            "Full fine-tune": "none",
            "LoRA adapters": "lora",
        }.get(self.peft_method.currentText(), "none")

    def _lora_target_value(self) -> str:
        """Return selected LoRA target groups.

        Returns:
            Comma-separated target group string.
        """

        return {
            "Attention projections": "attention",
            "MLP projections": "mlp",
            "Attention + MLP": "attention,mlp",
        }.get(self.lora_targets.currentText(), "attention")

    def _update_training_mode_controls(self) -> None:
        """Enable fine-tune controls only when fine-tuning is selected."""

        enabled = self._training_mode_value() == "fine_tune"
        lora_enabled = enabled and self._peft_method_value() == "lora"
        self.fine_tune_checkpoint.setEnabled(enabled)
        self.peft_method.setEnabled(enabled)
        self.fine_tune_check_button.setEnabled(enabled)
        self.lora_rank.setEnabled(lora_enabled)
        self.lora_alpha.setEnabled(lora_enabled)
        self.lora_dropout.setEnabled(lora_enabled)
        self.lora_targets.setEnabled(lora_enabled)
        self.refresh_fine_tune_workflow()

    def _current_dataset_summary(self) -> dict[str, Any]:
        """Read the active prepared dataset summary.

        Returns:
            Dataset summary dictionary, or an empty dictionary.
        """

        summary_path = Path(self.train_data_dir.text()) / "dataset_summary.json"
        if not summary_path.exists():
            summary_path = Path(self.dataset_dir.text()) / "dataset_summary.json"
        if not summary_path.exists():
            return {}
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            LOGGER.warning("Could not read dataset summary %s: %s", summary_path, exc)
            return {}

    def _fine_tune_dataset_stage_status(self) -> tuple[bool, str]:
        """Check whether the prepared dataset matches the fine-tune type.

        Returns:
            Tuple containing whether the workflow may proceed and a user-facing message.
        """

        expected_stage = self._training_stage_value()
        summary = self._current_dataset_summary()
        if not summary:
            return False, "Dataset: not prepared. Prepare the fine-tune dataset first."
        dataset_stage = str(summary.get("dataset_stage") or self._dataset_stage_value())
        tokens = int(summary.get("token_count", 0) or 0)
        vocab = int(summary.get("tokenizer_vocab_size", 0) or 0)
        stage_name = dataset_stage_label(dataset_stage) if dataset_stage in {"base", "instruction", "conversation"} else dataset_stage
        details = f"{stage_name}, {tokens:,} tokens, vocab {vocab:,}"
        if expected_stage == "instruction" and dataset_stage != "instruction":
            return False, f"Dataset mismatch: selected Instruction fine-tune, but prepared dataset is {details}."
        if expected_stage == "conversation" and dataset_stage != "conversation":
            return False, f"Dataset mismatch: selected Conversation fine-tune, but prepared dataset is {details}."
        if expected_stage == "domain" and dataset_stage == "base":
            return True, f"Dataset warning: {details}. Base datasets usually belong to pretraining; continue only for domain adaptation."
        return True, f"Dataset ready: {details}."

    def refresh_fine_tune_workflow(self) -> None:
        """Refresh fine-tune workflow guidance in the Fine-Tuning tab."""

        if not hasattr(self, "fine_tune_dataset_status"):
            return
        ok, message = self._fine_tune_dataset_stage_status()
        self.fine_tune_dataset_status.setText(message)
        self.fine_tune_dataset_status.setProperty("state", "ok" if ok else "warning")
        self.fine_tune_dataset_status.style().unpolish(self.fine_tune_dataset_status)
        self.fine_tune_dataset_status.style().polish(self.fine_tune_dataset_status)

    def apply_recommended_fine_tune_settings(self) -> None:
        """Apply conservative fine-tuning defaults for the selected workflow."""

        stage = self._training_stage_value()
        self._set_combo_text(self.peft_method, "LoRA adapters")
        self.lora_dropout.setValue(0.05)
        self._set_combo_text(self.lora_targets, "Attention projections")
        self.max_grad_norm.setValue(0.5)
        self.weight_decay.setValue(0.05)
        self._set_combo_by_data(self.scheduler_name, "cosine", {
            "warmup_linear": "Warmup linear",
            "cosine": "Cosine decay",
            "polynomial": "Polynomial decay",
            "one_cycle": "One-cycle",
            "constant": "Constant",
        })
        if stage == "conversation":
            self.lora_rank.setValue(16)
            self.lora_alpha.setValue(32.0)
            self.learning_rate.setValue(0.00003)
            self.epochs.setValue(max(1, min(self.epochs.value(), 2)))
        elif stage == "instruction":
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.learning_rate.setValue(0.00005)
            self.epochs.setValue(max(1, min(self.epochs.value(), 3)))
        else:
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.learning_rate.setValue(0.00005)
        self._update_training_mode_controls()
        self.fine_tune_preview.setText(
            "Recommended LoRA settings applied.\n"
            "Use Check Fine-tune before starting so checkpoint and tokenizer compatibility are verified."
        )

    def _attention_type_value(self) -> str:
        """Return the selected attention layout identifier.

        Returns:
            Stable attention type used by the model.
        """

        return {
            "Multi-head": "mha",
            "Grouped-query": "gqa",
            "Multi-query": "mqa",
        }.get(self.attention_type.currentText(), "mha")

    def _attention_backend_value(self) -> str:
        """Return the selected attention backend identifier.

        Returns:
            Stable attention backend used by the model.
        """

        return {
            "SDPA / Flash when available": "sdpa",
            "Manual": "manual",
        }.get(self.attention_backend.currentText(), "sdpa")

    def apply_training_profile(self) -> None:
        """Apply the selected optimizer/scheduler profile."""

        profile = self.training_profile.currentText()
        if profile == "Low-memory":
            self._set_combo_text(self.optimizer_name, "Adafactor")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.0002)
            self.weight_decay.setValue(0.05)
            self.min_lr_ratio.setValue(0.05)
            self.max_grad_norm.setValue(1.0)
            self._set_combo_text(self.precision, "BF16" if torch.cuda.is_available() else "FP32")
            self._set_combo_text(self.attention_type, "Grouped-query")
            self.kv_head_count.setValue(max(1, self.n_head.value() // 2))
        elif profile == "Code fine-tune":
            self._set_combo_text(self.optimizer_name, "AdamW")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.00005)
            self.weight_decay.setValue(0.05)
            self.min_lr_ratio.setValue(0.1)
            self.max_grad_norm.setValue(0.5)
            self.dropout.setValue(0.05)
            self._set_combo_text(self.training_mode, "Fine-tune checkpoint")
            self._set_combo_text(self.peft_method, "LoRA adapters")
            self.lora_rank.setValue(8)
            self.lora_alpha.setValue(16.0)
            self.lora_dropout.setValue(0.05)
            self._set_combo_text(self.lora_targets, "Attention projections")
        elif profile == "Experimental Lion":
            self._set_combo_text(self.optimizer_name, "Lion")
            self._set_combo_text(self.scheduler_name, "One-cycle")
            self.learning_rate.setValue(0.0001)
            self.weight_decay.setValue(0.1)
            self.min_lr_ratio.setValue(0.01)
            self.max_grad_norm.setValue(1.0)
        else:
            self._set_combo_text(self.optimizer_name, "AdamW")
            self._set_combo_text(self.scheduler_name, "Cosine decay")
            self.learning_rate.setValue(0.0003)
            self.weight_decay.setValue(0.1)
            self.min_lr_ratio.setValue(0.1)
            self.max_grad_norm.setValue(1.0)
            self._set_combo_text(self.precision, "FP16")
        self._update_training_mode_controls()
        self.refresh_model_estimate()
        self.training_log.append(f"Applied training profile: {profile}")

    def _tokenizer_strategy_reuses(self) -> bool:
        """Return whether current tokenizer strategy ignores vocabulary controls.

        Returns:
            True when an existing tokenizer is selected directly.
        """

        return self.tokenizer_strategy.currentText() in {"Reuse dataset tokenizer", "Import tokenizer.json"}

    def _update_tokenizer_strategy_controls(self) -> None:
        """Enable only the tokenizer inputs relevant to the selected strategy."""

        imports_tokenizer = self.tokenizer_strategy.currentText() == "Import tokenizer.json"
        reuses_tokenizer = self._tokenizer_strategy_reuses()
        if hasattr(self, "tokenizer_path_row"):
            self.tokenizer_path_row.setEnabled(imports_tokenizer)
        self.tokenizer_path.setEnabled(imports_tokenizer)
        self.auto_vocab.setEnabled(not reuses_tokenizer)
        self.manual_vocab_size.setEnabled(not reuses_tokenizer and not self.auto_vocab.isChecked())
        self.min_frequency.setEnabled(not reuses_tokenizer)

    def _update_model_estimate_chips(self, estimate: dict[str, Any]) -> None:
        """Update model and VRAM estimate chips.

        Args:
            estimate: Estimate dictionary from the training planning service.
        """

        params = int(estimate.get("parameters", 0))
        checkpoint_bytes = float(estimate.get("checkpoint_bytes", 0))
        vram_bytes = float(estimate.get("vram_bytes", 0))
        self.model_size_metric.setText(f"Model: {params / 1_000_000:.2f}M, ckpt {format_bytes(checkpoint_bytes)}")
        self.vram_estimate_metric.setText(f"VRAM est: {format_bytes(vram_bytes)}")

    def _current_model_config(self, vocab_size: int = 1) -> ModelConfig:
        """Build a model config from the current AI tab settings.

        Args:
            vocab_size: Tokenizer vocabulary size to use.

        Returns:
            Current model configuration.
        """

        return ModelConfig(
            vocab_size=vocab_size,
            context_length=self.train_context_length.value(),
            embedding_size=self.n_embd.value(),
            head_count=self.n_head.value(),
            layer_count=self.n_layer.value(),
            dropout=self.dropout.value(),
            attention_type=self._attention_type_value(),
            kv_head_count=self.kv_head_count.value(),
            attention_backend=self._attention_backend_value(),
            attention_window=self.attention_window.value(),
            **self._architecture_style_config(),
        )

    def _current_training_config(
        self,
        resume_path: Optional[Path] = None,
        training_mode: Optional[str] = None,
    ) -> TrainingConfig:
        """Build a training config from the current AI tab settings.

        Args:
            resume_path: Optional specific checkpoint to resume from.
            training_mode: Optional explicit training mode override.

        Returns:
            Current training configuration.
        """

        return TrainingConfig(
            output_dir=Path(self.model_dir.text()),
            epochs=self.epochs.value(),
            batch_size=self.batch_size.value(),
            learning_rate=self.learning_rate.value(),
            weight_decay=self.weight_decay.value(),
            optimizer_name=self._optimizer_value(),
            scheduler_name=self._scheduler_value(),
            scheduler_min_lr_ratio=self.min_lr_ratio.value(),
            polynomial_power=self.polynomial_power.value(),
            gradient_accumulation=self.gradient_accumulation.value(),
            warmup_steps=self.warmup_steps.value(),
            eval_interval=self.eval_interval.value(),
            max_eval_batches=self.max_eval_batches.value(),
            save_interval=self.save_interval.value(),
            data_loader_workers=self.data_loader_workers.value(),
            max_grad_norm=self.max_grad_norm.value(),
            device=self.device.currentText(),
            use_amp=self.use_amp.isChecked(),
            precision=self._precision_value(),
            seed=self.seed.value(),
            training_mode=training_mode or self._training_mode_value(),
            fine_tune_from_checkpoint=(
                Path(self.fine_tune_checkpoint.text())
                if training_mode != "pretrain" and self.fine_tune_checkpoint.text().strip()
                else None
            ),
            peft_method="none" if training_mode == "pretrain" else self._peft_method_value(),
            lora_rank=self.lora_rank.value(),
            lora_alpha=self.lora_alpha.value(),
            lora_dropout=self.lora_dropout.value(),
            lora_target_modules=self._lora_target_value(),
            resume=self.resume_training.isChecked(),
            resume_from_checkpoint=resume_path if self.resume_training.isChecked() else None,
            require_compatible_resume=self.resume_safety.isChecked(),
        )

    def _current_training_vocab_size(self, data_dir: Path) -> int:
        """Return the tokenizer vocabulary size for the current training dataset.

        Args:
            data_dir: Prepared dataset folder.

        Returns:
            Vocabulary size, or zero if unavailable.
        """

        summary_path = data_dir / "dataset_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            if vocab_size > 0:
                return vocab_size
        tokenizer_path = data_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
            vocab = tokenizer_data.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict):
                return len(vocab)
        return 0

    def _selected_resume_path(self) -> Optional[Path]:
        """Return the selected or latest checkpoint path.

        Returns:
            Checkpoint path, or ``None`` when no checkpoint exists.
        """

        if self.resume_checkpoint.text().strip():
            return Path(self.resume_checkpoint.text())
        return latest_checkpoint(Path(self.model_dir.text()) / "checkpoints")

    def preview_resume_compatibility(self) -> None:
        """Preview whether the selected checkpoint can resume safely."""

        if not self.resume_training.isChecked():
            self.resume_training_preview.setText("[INFO] Resume latest is off. Enable resume to continue from a checkpoint.")
            return
        resume_path = self._selected_resume_path()
        if resume_path is None:
            self.resume_training_preview.setText("[INFO] No checkpoint found in the current model folder.")
            return
        if not resume_path.exists():
            self.resume_training_preview.setText(f"[BLOCK] Checkpoint does not exist:\n{resume_path}")
            return
        try:
            vocab_size = self._current_training_vocab_size(Path(self.train_data_dir.text()))
            if vocab_size <= 0:
                self.resume_training_preview.setText("[BLOCK] Could not determine current dataset tokenizer vocabulary size.")
                return
            model_config = self._current_model_config(vocab_size=vocab_size)
            training_config = self._current_training_config(resume_path)
            model_config.validate()
            training_config.validate()
            report = check_resume_compatibility(resume_path, model_config, training_config)
            errors = list(report.errors)
            if training_config.require_compatible_resume:
                if not report.can_load_optimizer_state:
                    errors.append("Safe resume requires matching optimizer state.")
                if not report.can_load_scheduler_state:
                    errors.append("Safe resume requires matching scheduler state.")
                if not report.can_load_scaler_state:
                    errors.append("Safe resume requires matching AMP scaler state.")
            lines: list[str] = []
            if errors:
                lines.append("[BLOCK] Resume is not safe with the current settings.")
            elif report.warnings:
                lines.append("[WARN] Resume is possible, but settings changed.")
            else:
                lines.append("[OK] Checkpoint can resume with the current settings.")
            lines.extend(f"[OK] {line}" for line in report.info)
            lines.extend(f"[WARN] {line}" for line in report.warnings)
            lines.extend(f"[BLOCK] {line}" for line in errors)
            if not training_config.require_compatible_resume and not errors:
                lines.append("[INFO] Safe resume is off. Compatible weights will load; incompatible optimizer state may be skipped.")
            self.resume_training_preview.setText("\n".join(lines))
        except Exception as exc:
            self.resume_training_preview.setText(f"[BLOCK] Could not check resume compatibility:\n{exc}")

    def preview_fine_tune_compatibility(self) -> None:
        """Preview whether the selected checkpoint can be used for fine-tuning."""

        stage_ok, stage_message = self._fine_tune_dataset_stage_status()
        if not stage_ok:
            self.fine_tune_preview.setText(f"[BLOCK] {stage_message}")
            return
        base_path = Path(self.fine_tune_checkpoint.text()) if self.fine_tune_checkpoint.text().strip() else None
        if base_path is None:
            self.fine_tune_preview.setText("[BLOCK] Choose a base checkpoint for fine-tuning.")
            return
        if not base_path.exists():
            self.fine_tune_preview.setText(f"[BLOCK] Fine-tune base checkpoint does not exist:\n{base_path}")
            return
        try:
            vocab_size = self._current_training_vocab_size(Path(self.train_data_dir.text()))
            if vocab_size <= 0:
                self.fine_tune_preview.setText("[BLOCK] Could not determine current dataset tokenizer vocabulary size.")
                return
            model_config = self._current_model_config(vocab_size=vocab_size)
            training_config = self._current_training_config()
            model_config.validate()
            report = check_resume_compatibility(base_path, model_config, training_config)
            lines: list[str] = []
            if report.errors:
                lines.append("[BLOCK] Base checkpoint cannot be fine-tuned with the current model/dataset settings.")
            else:
                lines.append("[OK] Base checkpoint weights can be used for fine-tuning.")
            lines.append(f"[OK] {stage_message}" if stage_ok else f"[BLOCK] {stage_message}")
            lines.extend(f"[OK] {line}" for line in report.info)
            behavior_warnings = [
                warning for warning in report.warnings
                if not warning.startswith("Optimizer changed:") and not warning.startswith("LR scheduler changed:")
            ]
            lines.extend(f"[WARN] {line}" for line in behavior_warnings)
            lines.extend(f"[BLOCK] {line}" for line in report.errors)
            if not report.errors:
                lines.append("[INFO] Fine-tuning starts fresh optimizer, scheduler, and scaler state.")
            self.fine_tune_preview.setText("\n".join(lines))
        except Exception as exc:
            self.fine_tune_preview.setText(f"[BLOCK] Could not check fine-tune compatibility:\n{exc}")

    def _training_history_path(self) -> Path:
        """Return the training history path for the selected model folder.

        Returns:
            Path to ``training_history.json``.
        """

        return Path(self.model_dir.text()) / "training_history.json"

    def _load_training_history(self) -> list[dict[str, Any]]:
        """Load training run history.

        Returns:
            List of training run entries.
        """

        path = self._training_history_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def refresh_model_estimate(self) -> None:
        """Refresh model size, rough VRAM, and run history widgets."""

        model_config = self._current_model_config()
        training_config = self._current_training_config()
        data_dir = Path(self.train_data_dir.text())
        train_tokens = max(model_config.context_length * training_config.batch_size, 1)
        try:
            summary_path = data_dir / "dataset_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
                train_tokens = int(summary.get("train_token_count", summary.get("token_count", train_tokens)) or train_tokens)
                if vocab_size > 0:
                    model_config.vocab_size = vocab_size
            elif (data_dir / "tokenizer.json").exists():
                tokenizer_data = json.loads((data_dir / "tokenizer.json").read_text(encoding="utf-8"))
                vocab = tokenizer_data.get("model", {}).get("vocab", {})
                if vocab:
                    model_config.vocab_size = len(vocab)
        except Exception as exc:
            self.training_log.append(f"[WARN] Could not refresh dataset-based estimate: {exc}")
        estimate = estimate_training_resources(model_config, training_config, train_tokens)
        self.last_training_estimate = estimate
        self._update_model_estimate_chips(estimate)
        self.history_metric.setText(f"Runs: {len(self._load_training_history())}")
        self.training_log.append(
            "Model estimate refreshed: "
            f"{int(estimate['parameters']):,} params, "
            f"checkpoint {format_bytes(float(estimate['checkpoint_bytes']))}, "
            f"VRAM {format_bytes(float(estimate['vram_bytes']))}."
        )

    def _append_training_history(self, result: Any) -> None:
        """Persist a training run entry to ``training_history.json``.

        Args:
            result: Training result object.
        """

        history_path = self._training_history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = self._load_training_history()
        summary = {}
        try:
            if Path(result.summary_path).exists():
                summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        estimate = getattr(self, "last_training_estimate", {}) or {}
        entry = {
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "checkpoint_path": str(result.checkpoint_path),
            "summary_path": str(result.summary_path),
            "stopped": bool(getattr(result, "stopped", False)),
            "final_train_loss": result.final_train_loss,
            "final_val_loss": result.final_val_loss,
            "dataset_dir": self.train_data_dir.text(),
            "dataset_version": (summary.get("model_lineage") or {}).get("dataset_version"),
            "training_run_id": summary.get("training_run_id"),
            "parameters": estimate.get("parameters") or summary.get("parameters"),
            "model_config": summary.get("model_config"),
            "training_config": summary.get("training_config"),
        }
        history.append(entry)
        history_path.write_text(json.dumps(history[-200:], indent=2), encoding="utf-8")
        self.history_metric.setText(f"Runs: {len(history[-200:])}")
        (self.active_training_log or self.training_log).append(f"Training history updated: {history_path}")

    def _run_training_preflight(self, model_config: ModelConfig, training_config: TrainingConfig) -> bool:
        """Run pre-training checklist and disk-space guard.

        Args:
            model_config: Selected model architecture.
            training_config: Selected training settings.

        Returns:
            True when training may continue.
        """

        log = self.active_training_log or self.training_log
        data_dir = Path(self.train_data_dir.text())
        output_dir = Path(self.model_dir.text())
        errors: list[str] = []
        warnings: list[str] = []
        info: list[str] = []
        required = ("tokenizer.json", "train_tokens.json", "val_tokens.json")
        missing = [name for name in required if not (data_dir / name).exists()]
        if not data_dir.exists():
            errors.append(f"Dataset folder does not exist: {data_dir}")
        elif missing:
            errors.append(f"Dataset is not prepared. Missing: {', '.join(missing)}")
        else:
            info.append("Dataset artifacts found.")

        vocab_size = 0
        train_tokens = 0
        val_tokens = 0
        summary = {}
        try:
            summary_path = data_dir / "dataset_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            if vocab_size <= 0:
                tokenizer_data = json.loads((data_dir / "tokenizer.json").read_text(encoding="utf-8"))
                vocab_size = len(tokenizer_data.get("model", {}).get("vocab", {}))
            train_tokens = int(summary.get("train_token_count", 0) or 0)
            val_tokens = int(summary.get("val_token_count", 0) or 0)
            if train_tokens <= 0 and (data_dir / "train_tokens.json").exists():
                train_tokens = len(json.loads((data_dir / "train_tokens.json").read_text(encoding="utf-8")))
            if val_tokens <= 0 and (data_dir / "val_tokens.json").exists():
                val_tokens = len(json.loads((data_dir / "val_tokens.json").read_text(encoding="utf-8")))
        except Exception as exc:
            warnings.append(f"Could not fully inspect dataset metadata: {exc}")

        if vocab_size > 0:
            model_config.vocab_size = vocab_size
            info.append(f"Tokenizer vocab: {vocab_size:,}.")
        elif not missing:
            errors.append("Could not determine tokenizer vocabulary size.")
        if train_tokens and train_tokens <= model_config.context_length:
            errors.append("Training token count must be larger than context length.")
        elif train_tokens:
            info.append(f"Training tokens: {train_tokens:,}; validation tokens: {val_tokens:,}.")
            if train_tokens < 50_000:
                warnings.append("Training token count is very small; expect smoke-test quality.")

        try:
            model_config.validate()
        except Exception as exc:
            errors.append(f"Model architecture is invalid: {exc}")
        try:
            training_config.validate()
        except Exception as exc:
            errors.append(f"Training options are invalid: {exc}")
        if model_config.attention_backend == "sdpa":
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                if training_config.device == "cuda" and torch.cuda.is_available():
                    flash_enabled = bool(getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: False)())
                    info.append("Attention backend: SDPA selected; Flash Attention may be used by PyTorch." if flash_enabled else "Attention backend: SDPA selected; CUDA flash kernel is not enabled.")
                else:
                    info.append("Attention backend: SDPA selected; CPU/backend fallback will be used if needed.")
            else:
                warnings.append("SDPA attention selected, but this PyTorch build does not expose scaled_dot_product_attention.")
        else:
            warnings.append("Manual attention backend selected. This is useful for debugging but can be slower.")
        if training_config.peft_method == "lora":
            info.append(
                "PEFT: LoRA adapters enabled. Intermediate checkpoints will save adapter weights; final_model.pt will be merged."
            )

        if training_config.device == "cuda" and not torch.cuda.is_available():
            errors.append("CUDA is selected, but PyTorch cannot use CUDA on this machine.")
        elif training_config.device == "cuda":
            info.append(f"CUDA ready: {torch.cuda.get_device_name(0)}.")
            if training_config.data_loader_workers > 0:
                info.append(f"CPU-assisted batch loading enabled with {training_config.data_loader_workers} worker(s).")
        else:
            warnings.append("CPU training is selected. This can be very slow.")
        if sys.platform.startswith("win") and training_config.data_loader_workers > 4:
            warnings.append("High CPU worker counts can duplicate dataset memory on Windows. Start with 2-4 workers and increase carefully.")

        active_resume_path: Optional[Path] = None
        resume_path = training_config.resume_from_checkpoint if training_config.resume else None
        if resume_path and not Path(resume_path).exists():
            errors.append(f"Selected resume checkpoint does not exist: {resume_path}")
        elif training_config.resume:
            if resume_path is None:
                resume_path = latest_checkpoint(output_dir / "checkpoints")
            if resume_path is None:
                info.append("Resume latest is enabled, but no checkpoint exists yet.")
            else:
                active_resume_path = Path(resume_path)
                try:
                    compatibility = check_resume_compatibility(active_resume_path, model_config, training_config)
                    info.extend(compatibility.info)
                    warnings.extend(compatibility.warnings)
                    errors.extend(compatibility.errors)
                    if training_config.require_compatible_resume:
                        if not compatibility.can_load_optimizer_state:
                            errors.append("Safe resume requires matching optimizer state.")
                        if not compatibility.can_load_scheduler_state:
                            errors.append("Safe resume requires matching scheduler state.")
                        if not compatibility.can_load_scaler_state:
                            errors.append("Safe resume requires matching AMP scaler state.")
                except Exception as exc:
                    errors.append(f"Could not inspect resume checkpoint: {exc}")
        if training_config.training_mode == "fine_tune" and active_resume_path is None:
            base_path = training_config.fine_tune_from_checkpoint
            if base_path is None:
                errors.append("Fine-tune mode requires a base checkpoint.")
            elif not Path(base_path).exists():
                errors.append(f"Fine-tune base checkpoint does not exist: {base_path}")
            else:
                try:
                    compatibility = check_resume_compatibility(Path(base_path), model_config, training_config)
                    info.append(f"Fine-tune base checkpoint: {Path(base_path).name}.")
                    warnings.extend(
                        warning for warning in compatibility.warnings
                        if not warning.startswith("Optimizer changed:") and not warning.startswith("LR scheduler changed:")
                    )
                    errors.extend(compatibility.errors)
                    if not compatibility.errors:
                        info.append("Fine-tune base weights are compatible. Optimizer state will start fresh.")
                except Exception as exc:
                    errors.append(f"Could not inspect fine-tune base checkpoint: {exc}")
        elif training_config.training_mode == "pretrain" and active_resume_path is None:
            info.append("A fresh pretraining run will start from random weights.")
        elif training_config.training_mode == "fine_tune" and active_resume_path is not None:
            info.append("Existing run checkpoint found; training will resume that run instead of reloading the fine-tune base.")

        output_dir.mkdir(parents=True, exist_ok=True)
        estimate = estimate_training_resources(model_config, training_config, train_tokens)
        self.last_training_estimate = estimate
        self._update_model_estimate_chips(estimate)
        params = int(estimate["parameters"])
        checkpoint_bytes = float(estimate["checkpoint_bytes"])
        checkpoint_count = int(estimate["checkpoint_count"])
        estimated_storage = float(estimate["estimated_storage"])
        estimated_vram = float(estimate["vram_bytes"])
        free_bytes = shutil.disk_usage(output_dir).free
        info.append(f"Estimated parameters: {params:,}.")
        info.append(f"Estimated checkpoint size: {format_bytes(checkpoint_bytes)}.")
        info.append(f"Estimated training VRAM: {format_bytes(estimated_vram)}.")
        info.append(f"Estimated training storage need: {format_bytes(estimated_storage)}.")
        info.append(f"Free space on model drive: {format_bytes(free_bytes)}.")
        if training_config.device == "cuda" and torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info()
            info.append(f"GPU free/total VRAM: {format_bytes(free_vram)} / {format_bytes(total_vram)}.")
            if estimated_vram > free_vram * 0.9:
                warnings.append("Estimated VRAM is close to or above currently free GPU memory.")
        if free_bytes < estimated_storage * 1.25:
            errors.append("Not enough free disk space for estimated checkpoints and final model.")
        elif free_bytes < estimated_storage * 2:
            warnings.append("Free disk space is close to the estimated training storage need.")
        if checkpoint_count > 50:
            warnings.append("Save interval may create many checkpoints. Increase Save every or clean old checkpoints.")

        log.clear()
        log.append("Training checklist")
        for line in info:
            log.append(f"[OK] {line}")
        for line in warnings:
            log.append(f"[WARN] {line}")
        for line in errors:
            log.append(f"[ERROR] {line}")

        if errors:
            self.project_state.setText("Training blocked")
            self.train_status.setText("Training: blocked")
            QMessageBox.warning(self, "Training blocked", "Fix the checklist errors before starting training.")
            return False
        if warnings:
            message = "Training checklist has warnings. Continue anyway?\n\n" + "\n".join(f"- {warning}" for warning in warnings[:8])
            choice = QMessageBox.question(self, "Training warnings", message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if choice != QMessageBox.Yes:
                self.project_state.setText("Training cancelled")
                self.train_status.setText("Training: idle")
                return False
        return True

    def start_training(self) -> None:
        """Collect training options and start model training."""

        if self._training_launch_target_value() == "remote":
            self.publish_remote_training_job()
            return
        self.active_training_log = self.training_log
        self.active_training_progress = self.training_progress
        self.active_training_final_button_text = "Start Training"
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        dataset_dir = Path(self.train_data_dir.text())
        vocab_size = self._current_training_vocab_size(dataset_dir)
        if vocab_size <= 0:
            QMessageBox.warning(self, "Training blocked", "Could not determine tokenizer vocabulary size. Prepare the dataset first.")
            return
        model_config = self._current_model_config(vocab_size=vocab_size)
        training_config = self._current_training_config(resume_path, training_mode="pretrain")
        if not self._run_training_preflight(model_config, training_config):
            return
        self._init_telemetry_store(Path(self.model_dir.text()))
        self.training_log.append("")
        self.training_progress.setValue(0)
        self.training_epoch_metric.setText("Epoch: -")
        self.training_step_metric.setText("Step: -")
        self.training_loss_metric.setText("Train loss: -")
        self.training_val_metric.setText("Val loss: -")
        self.training_lr_metric.setText("LR: -")
        self.training_speed_metric.setText("Speed: -")
        self.training_grad_metric.setText("Grad: -")
        self.training_vram_metric.setText("VRAM: -")
        self.training_eta_metric.setText("ETA: -")
        self.loss_chart.clear()
        self.optimization_chart.clear()
        self.stability_chart.clear()
        self.throughput_chart.clear()
        self.memory_chart.clear()
        self.live_prediction_chart.update_distribution(0, None)
        self.live_attention_chart.update_heatmap(0, None)
        self.live_activation_chart.update_histogram(0, None)
        self.live_gradient_chart.update_flow(self.n_layer.value(), None, 0)
        self.live_progress.setValue(0)
        self.live_epoch_metric.setText("Epoch: -")
        self.live_step_metric.setText("Step: -")
        self.live_tokens_metric.setText("Tokens/sec: -")
        self.live_loss_metric.setText("Loss: -")
        self.live_lr_metric.setText("LR: -")
        self.live_data_metric.setText("Data: -")
        self.live_sample_text.setText("Training text: -")
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), 0, None)
        self._set_meter(self.live_cpu_bar, "CPU", self._system_cpu_value())
        self._set_meter(self.live_gpu_bar, "GPU memory", None)
        self._set_meter(self.live_vram_bar, "VRAM reserved", None)
        self._set_meter(self.live_ram_bar, "System RAM", self._system_ram_value())
        self.live_worker_status.setText(f"CPU workers: {self.data_loader_workers.value()}")
        self.training_log.append("Training started...")
        self.project_state.setText("Training")
        self.train_status.setText("Training: running")
        self._run_task(
            run_training_job,
            (dataset_dir, model_config, training_config),
            self._training_finished,
            self.training_log,
            self.training_progress,
            with_progress=True,
            button=self.train_button,
            stop_button=self.stop_training_button,
            busy_text="Training",
        )

    def start_fine_tuning(self) -> None:
        """Collect fine-tuning options and start adaptation training."""

        self.active_training_log = self.fine_tune_log
        self.active_training_progress = self.fine_tune_progress
        self.active_training_final_button_text = "Start Fine-Tune"
        stage_ok, stage_message = self._fine_tune_dataset_stage_status()
        self.refresh_fine_tune_workflow()
        if not stage_ok:
            self.fine_tune_log.append(stage_message)
            QMessageBox.warning(self, "Fine-tune blocked", stage_message)
            return
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        dataset_dir = Path(self.train_data_dir.text())
        vocab_size = self._current_training_vocab_size(dataset_dir)
        if vocab_size <= 0:
            QMessageBox.warning(self, "Fine-tune blocked", "Could not determine tokenizer vocabulary size. Prepare the fine-tuning dataset first.")
            return
        model_config = self._current_model_config(vocab_size=vocab_size)
        training_config = self._current_training_config(resume_path, training_mode="fine_tune")
        if not self._run_training_preflight(model_config, training_config):
            return
        self._init_telemetry_store(Path(self.model_dir.text()))
        self.fine_tune_log.append("")
        self.fine_tune_progress.setValue(0)
        self.training_progress.setValue(0)
        self.loss_chart.clear()
        self.optimization_chart.clear()
        self.stability_chart.clear()
        self.throughput_chart.clear()
        self.memory_chart.clear()
        self.live_progress.setValue(0)
        self.live_sample_text.setText("Training text: -")
        self.live_flow.set_state(self.n_layer.value(), self.n_head.value(), 0, None)
        self.fine_tune_log.append("Fine-tuning started...")
        self.project_state.setText("Fine-tuning")
        self.train_status.setText("Training: fine-tuning")
        self._run_task(
            run_fine_tuning_job,
            (dataset_dir, model_config, training_config, self._training_stage_value()),
            self._training_finished,
            self.fine_tune_log,
            self.fine_tune_progress,
            with_progress=True,
            button=self.fine_tune_button,
            stop_button=self.stop_fine_tune_button,
            busy_text="Fine-tuning",
        )

    @Slot(object)
    def _training_finished(self, result: Any) -> None:
        """Update UI after training finishes.

        Args:
            result: Training result.
        """

        log = self.active_training_log or self.training_log
        progress = self.active_training_progress or self.training_progress
        progress.setValue(100)
        if progress is not self.training_progress:
            self.training_progress.setValue(100)
        if hasattr(self, "live_progress"):
            self.live_progress.setValue(100)
        log.append(f"Saved model: {result.checkpoint_path}")
        log.append(f"Final train loss: {result.final_train_loss:.4f}")
        if result.final_val_loss is not None:
            log.append(f"Final validation loss: {result.final_val_loss:.4f}")
        self.export_model_dir.setText(str(Path(self.model_dir.text())))
        if getattr(result, "stopped", False):
            self.project_state.setText("Training stopped")
            self.train_status.setText("Training: stopped, checkpoint saved")
            log.append("Training stopped safely. Resume from this checkpoint or the latest checkpoint.")
        else:
            self.project_state.setText("Training complete")
            self.train_status.setText(f"Training: loss {result.final_train_loss:.4f}")
        self._append_training_history(result)
        self._clear_button_busy(self.active_training_final_button_text)
        self.active_training_log = None
        self.active_training_progress = None

    def run_benchmark(self) -> None:
        """Run benchmark prompts against the current trained model."""

        prompts = normalize_prompts(self.benchmark_prompts.toPlainText())
        self.benchmark_log.append(f"Running benchmark with {len(prompts)} prompt(s)...")
        self.benchmark_progress.setValue(0)
        self.project_state.setText("Benchmarking")
        self._run_task(
            evaluate_checkpoint,
            (
                Path(self.model_dir.text()),
                prompts,
                None,
                self.benchmark_tokens.value(),
                self.benchmark_temperature.value(),
                50,
                self.device.currentText(),
                self.benchmark_kv_cache.isChecked(),
            ),
            self._benchmark_finished,
            self.benchmark_log,
            self.benchmark_progress,
            with_progress=True,
            button=self.run_benchmark_button,
            stop_button=self.stop_benchmark_button,
            busy_text="Benchmarking",
        )

    @Slot(object)
    def _benchmark_finished(self, result: Any) -> None:
        """Update UI after benchmark prompts finish.

        Args:
            result: Benchmark result object.
        """

        self.benchmark_progress.setRange(0, 100)
        self.benchmark_progress.setValue(100)
        self.benchmark_log.append(
            f"Benchmark complete: {result.prompt_count} prompt(s), {result.total_seconds:.2f}s, "
            f"{result.total_generated_tokens} generated token(s), {result.tokens_per_second:.2f} tok/s."
        )
        self.benchmark_log.append(f"Benchmark saved: {result.output_path}")
        self.project_state.setText("Benchmark complete")
        self._clear_button_busy("Run Benchmark")

    def toggle_llm_model(self) -> None:
        """Load or unload the selected chat model depending on current state."""

        if self.chat_session is not None:
            self.unload_llm_model()
            return
        self.load_llm_model()

    def load_llm_model(self) -> None:
        """Load a selected model backend for chat testing."""

        backend = self._chat_backend_value()
        path_text = self.microgpt_chat_path.text().strip() if backend == "microgpt" else self.gguf_path.text().strip()
        if not path_text:
            required = "MicroGPT model folder or checkpoint" if backend == "microgpt" else "GGUF model file"
            QMessageBox.information(self, "Model required", f"Choose a {required} first.")
            return
        model_path = Path(path_text)
        self.chat_progress.setValue(0)
        self._render_chat_markdown("**Loading model...**")
        self.chat_stats.setText("Loading model...")
        self.project_state.setText("Loading chat model")
        self.chat_status.setText("Chat: loading model")
        loader = load_microgpt_chat_session if backend == "microgpt" else load_llama_chat_session
        args = (
            (model_path, self.device.currentText())
            if backend == "microgpt"
            else (model_path, self.llama_context.value(), self.llama_threads.value(), self.llama_gpu_layers.value())
        )
        self._run_task(
            loader,
            args,
            self._llm_loaded,
            self.chat_event_log,
            self.chat_progress,
            button=self.load_llm_button,
            busy_text="Loading Model",
        )

    @Slot(object)
    def _llm_loaded(self, session: Any) -> None:
        """Store a loaded GGUF chat session.

        Args:
            session: Loaded ``LlamaChatSession``.
        """

        self.chat_session = session
        self._clear_chat_messages()
        self.chat_markdown = ""
        self._add_chat_message(
            "assistant",
            f"Loaded model: `{session.model_path.name}`\n\n{session.runtime_summary}\n\nSend a message to begin.",
        )
        self.chat_progress.setValue(100)
        self.chat_stats.setText(session.runtime_summary)
        self.project_state.setText("Chat model loaded")
        self.chat_status.setText(f"Chat: {session.runtime_summary}")
        self._clear_button_busy("Unload")
        self._tip(self.load_llm_button, "Unload the currently loaded model from memory.")

    def unload_llm_model(self) -> None:
        """Unload the active chat model and clear chat state."""

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return
        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        self._clear_chat_messages()
        self.chat_markdown = ""
        self._add_chat_message("assistant", "Model unloaded.\n\nLoad a model to start testing.")
        self.chat_progress.setRange(0, 100)
        self.chat_progress.setValue(0)
        self.chat_stats.setText("Idle")
        self.project_state.setText("Ready")
        self.chat_status.setText("Chat: no model loaded")
        self.load_llm_button.setText("Load Model")
        self._update_chat_backend_controls()

    def send_chat_message(self) -> None:
        """Send a prompt to the loaded chat model."""

        if self.chat_session is None:
            QMessageBox.information(self, "Load model", "Load a model before sending a message.")
            return
        prompt = self.chat_input.toPlainText().strip()
        if not prompt:
            return
        self.pending_user_message = prompt
        self.chat_input.clear()
        self._add_chat_message("user", prompt, resend_prompt=prompt)
        self.chat_stream_reply = ""
        self._add_chat_message("assistant", "_Thinking..._", resend_prompt=prompt)
        self.chat_progress.setRange(0, 0)
        self.chat_stats.setText("Thinking...")
        self.project_state.setText("Generating")
        self.chat_status.setText("Chat: generating reply")
        streamer = stream_microgpt_chat_reply if self._chat_backend_value() == "microgpt" else stream_chat_reply
        self._run_task(
            streamer,
            (
                self.chat_session,
                prompt,
                self.system_prompt.toPlainText(),
                self.chat_max_tokens.value(),
                self.chat_temperature.value(),
                self.chat_top_p.value(),
                self.chat_repeat_penalty.value(),
                self.reasoning_effort.currentText(),
                self.thinking_enabled.isChecked(),
            ),
            self._chat_reply_finished,
            self.chat_event_log,
            self.chat_progress,
            with_progress=True,
            button=self.send_chat_button,
            stop_button=self.stop_chat_button,
            busy_text="Thinking",
        )

    @Slot(object)
    def _chat_reply_finished(self, reply: Any) -> None:
        """Render the model reply.

        Args:
            reply: Assistant reply text and metrics.
        """

        result = reply if isinstance(reply, dict) else {"reply": str(reply)}
        text = str(result.get("reply", "")).strip()
        if text:
            self.chat_stream_reply = text
        else:
            self.chat_stream_reply = self.chat_stream_reply or "_No reply returned._"
        self._render_chat_markdown(self.chat_stream_reply)
        self.chat_progress.setRange(0, 100)
        self.chat_progress.setValue(100)
        self._set_chat_stats(
            float(result.get("elapsed_seconds", 0.0)),
            int(result.get("token_count", 0)),
            float(result.get("tokens_per_second", 0.0)),
        )
        self.project_state.setText("Ready")
        self.chat_status.setText("Chat: ready")
        self._clear_button_busy("Send")

    def reset_chat(self) -> None:
        """Clear the chat transcript and model conversation memory."""

        if self.chat_session is not None:
            self.chat_session.reset()
        self._clear_chat_messages()
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self._add_chat_message("assistant", "Chat reset.")
        self.chat_stats.setText("Idle")
        self.chat_status.setText("Chat: ready")

    def _append_chat_markdown(self, role: str, content: str) -> None:
        """Append one rendered chat message.

        Args:
            role: Display role heading.
            content: Markdown content.
        """

        block = f"### {role}\n{content.strip()}\n"
        self.chat_markdown = f"{self.chat_markdown.rstrip()}\n\n{block}" if self.chat_markdown else block
        self._add_chat_message("user" if role.lower() in {"you", "user"} else "assistant", content)

    def create_bundle(self) -> None:
        """Create a portable model export bundle."""

        self.export_log.append("Creating model bundle...")
        self.export_progress.setValue(15)
        try:
            output = export_project_bundle(Path(self.export_model_dir.text()), Path(self.export_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Bundle created: {output}")
        self.export_status.setText("Export: bundle created")

    def quantize_model(self) -> None:
        """Create a quantized FP16 checkpoint when selected."""

        mode = self.quant_mode.currentText()
        if not mode.startswith("FP16"):
            self.export_log.append("This GGUF quantization target is planned. FP16 checkpoint quantization is available now.")
            return
        checkpoint = Path(self.export_model_dir.text()) / "final_model.pt"
        output = Path(self.export_dir.text()) / "final_model_fp16.pt"
        self.export_log.append("Creating FP16 checkpoint...")
        self.export_progress.setValue(20)
        try:
            result = quantize_checkpoint(checkpoint, output, mode="fp16")
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"Quantized checkpoint created: {result}")
        self.export_status.setText("Export: FP16 checkpoint ready")

    def export_hf_package(self) -> None:
        """Create an HF-style MicroGPT package."""

        self.export_log.append("Creating HF-style MicroGPT package...")
        self.export_progress.setValue(20)
        try:
            result = export_hf_microgpt_package(Path(self.export_model_dir.text()))
        except Exception as exc:
            self.export_log.append(f"Error: {exc}")
            self.export_progress.setValue(0)
            return
        self.export_progress.setValue(100)
        self.export_log.append(f"HF package created: {result}")
        self.export_log.append("Note: this package is MicroGPT model_type, not a llama.cpp-supported Llama model.")
        self.export_status.setText("Export: HF package ready")

    def convert_hf_to_gguf(self) -> None:
        """Convert an HF-compatible model folder to GGUF through llama.cpp."""

        model_dir_text = self.export_model_dir.text().strip()
        llama_dir_text = self.llama_cpp_dir.text().strip()
        output_text = self.gguf_output_path.text().strip()
        if not model_dir_text:
            QMessageBox.warning(self, "GGUF blocked", "Choose the model core folder first.")
            return
        if not (Path(model_dir_text) / "hf_model").exists():
            QMessageBox.warning(
                self,
                "GGUF blocked",
                "GGUF conversion needs an HF model package first. Use Export HF Package, then convert a llama.cpp-supported model.",
            )
            return
        if not llama_dir_text:
            QMessageBox.warning(self, "GGUF blocked", "Choose your local llama.cpp folder containing convert_hf_to_gguf.py.")
            return
        if not output_text:
            QMessageBox.warning(self, "GGUF blocked", "Choose a GGUF output file path.")
            return
        self.export_log.append("Starting llama.cpp GGUF conversion...")
        self.export_progress.setValue(0)
        self._run_task(
            export_gguf_with_llama_cpp,
            (
                Path(model_dir_text),
                Path(llama_dir_text),
                Path(output_text),
                self.gguf_outtype.currentText(),
            ),
            self._gguf_conversion_finished,
            self.export_log,
            self.export_progress,
            button=self.gguf_convert_button,
            busy_text="Converting GGUF",
        )

    @Slot(object)
    def _gguf_conversion_finished(self, result: Any) -> None:
        """Update UI after GGUF conversion finishes.

        Args:
            result: GGUF output path.
        """

        self.export_progress.setValue(100)
        self.export_log.append(f"GGUF created: {result}")
        self.gguf_path.setText(str(result))
        self.export_status.setText("Export: GGUF ready")
        self._clear_button_busy("Convert HF to GGUF")

    def _apply_preset(self, preset: str) -> None:
        """Apply architecture values for a preset.

        Args:
            preset: Selected preset name.
        """

        if preset == "Tiny":
            self.n_embd.setValue(128)
            self.n_head.setValue(4)
            self.n_layer.setValue(4)
        elif preset == "Small":
            self.n_embd.setValue(512)
            self.n_head.setValue(8)
            self.n_layer.setValue(8)


def main() -> None:
    """Launch the PySide6 desktop application."""

    log_file = setup_logging()
    qInstallMessageHandler(qt_message_handler)
    LOGGER.info("Starting Micro LLM Creator. Log file: %s", log_file)
    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
        except Exception:
            LOGGER.exception("Could not set Windows app user model ID")
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    app.setWindowIcon(MainWindow._static_lightning_icon())
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.apply_windows_taskbar_icon)
    interrupt_timer = QTimer()
    interrupt_timer.timeout.connect(lambda: None)
    interrupt_timer.start(200)
    window.interrupt_timer = interrupt_timer
    signal.signal(signal.SIGINT, lambda *_: QTimer.singleShot(0, window.request_shutdown_from_signal))
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

