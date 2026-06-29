from __future__ import annotations

import ctypes
from datetime import datetime
import json
import os
from queue import Empty, Queue
import re
import signal
import sys
from pathlib import Path
from threading import Event
from typing import Any, Optional, Union
from urllib.parse import quote

import torch
from PySide6.QtCore import QObject, QPoint, QPointF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QSpinBox,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from llm_trainer.config import DatasetConfig, ModelConfig, TrainingConfig
from llm_trainer.evaluation import DEFAULT_BENCHMARK_PROMPTS, evaluate_checkpoint, normalize_prompts
from llm_trainer.export import export_gguf_with_llama_cpp, export_hf_microgpt_package, export_project_bundle, quantize_checkpoint
from llm_trainer.llama_chat import LlamaChatSession, load_llama_chat_session, stream_chat_reply
from llm_trainer.services import build_dataset, train_from_dataset
from llm_trainer.ui.chat_widgets import ChatInputEdit, ChatMessageWidget


WINDOWS_APP_ID = "MicroLLMCreator.Lightning"


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


class LossChartWidget(QWidget):
    """Compact live chart for one or two training metric series."""

    def __init__(
        self,
        primary_label: str = "Train",
        secondary_label: str = "Val",
        empty_text: str = "Loss chart will appear during training",
    ) -> None:
        """Create an empty chart widget.

        Args:
            primary_label: Label for the primary series.
            secondary_label: Label for the secondary series.
            empty_text: Text shown before samples arrive.
        """

        super().__init__()
        self.setObjectName("LossChart")
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.primary_label = primary_label
        self.secondary_label = secondary_label
        self.empty_text = empty_text
        self.train_points: list[tuple[int, float]] = []
        self.val_points: list[tuple[int, float]] = []

    def clear(self) -> None:
        """Remove all plotted loss values."""

        self.train_points.clear()
        self.val_points.clear()
        self.update()

    def add_metrics(self, step: int, train_loss: Optional[float], val_loss: Optional[float]) -> None:
        """Add a training metric sample.

        Args:
            step: Optimizer step for the sample.
            train_loss: Optional training loss value.
            val_loss: Optional validation loss value.
        """

        if train_loss is not None:
            self.train_points.append((step, float(train_loss)))
        if val_loss is not None:
            self.val_points.append((step, float(val_loss)))
        self.train_points = self.train_points[-400:]
        self.val_points = self.val_points[-400:]
        self.update()

    def add_values(self, step: int, primary_value: Optional[float], secondary_value: Optional[float] = None) -> None:
        """Add a generic metric sample.

        Args:
            step: Optimizer step for the sample.
            primary_value: Primary series value.
            secondary_value: Optional secondary series value.
        """

        self.add_metrics(step, primary_value, secondary_value)

    def paintEvent(self, event: Any) -> None:
        """Render the chart.

        Args:
            event: Qt paint event.
        """

        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 10, -12, -18)
        painter.fillRect(self.rect(), QColor("#141414"))
        painter.setPen(QPen(QColor("#3d3d3d"), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 8, 8)

        all_points = self.train_points + self.val_points
        if not all_points:
            painter.setPen(QColor("#8d8d8d"))
            painter.drawText(rect, Qt.AlignCenter, self.empty_text)
            painter.end()
            return

        min_step = min(step for step, _ in all_points)
        max_step = max(step for step, _ in all_points)
        min_loss = min(loss for _, loss in all_points)
        max_loss = max(loss for _, loss in all_points)
        if max_step == min_step:
            max_step += 1
        if max_loss == min_loss:
            max_loss += 1.0
        loss_padding = (max_loss - min_loss) * 0.08
        min_loss = max(0.0, min_loss - loss_padding)
        max_loss += loss_padding

        painter.setPen(QPen(QColor("#333333"), 1))
        for index in range(1, 4):
            y = rect.top() + int(rect.height() * index / 4)
            painter.drawLine(rect.left(), y, rect.right(), y)

        self._draw_series(painter, rect, self.train_points, min_step, max_step, min_loss, max_loss, QColor("#f5b041"))
        self._draw_series(painter, rect, self.val_points, min_step, max_step, min_loss, max_loss, QColor("#b6d77a"))
        painter.setPen(QColor("#f5b041"))
        painter.drawText(rect.left(), self.height() - 6, self.primary_label)
        painter.setPen(QColor("#b6d77a"))
        painter.drawText(rect.left() + 82, self.height() - 6, self.secondary_label)
        painter.setPen(QColor("#cfcfcf"))
        painter.drawText(rect.right() - 120, self.height() - 6, f"{min_loss:.3f} - {max_loss:.3f}")
        painter.end()

    def _draw_series(
        self,
        painter: QPainter,
        rect: Any,
        points: list[tuple[int, float]],
        min_step: int,
        max_step: int,
        min_loss: float,
        max_loss: float,
        color: QColor,
    ) -> None:
        """Draw one line series.

        Args:
            painter: Active painter.
            rect: Plot rectangle.
            points: Step/loss pairs.
            min_step: Minimum plotted step.
            max_step: Maximum plotted step.
            min_loss: Minimum plotted loss.
            max_loss: Maximum plotted loss.
            color: Series color.
        """

        if not points:
            return
        mapped = [
            QPointF(
                rect.left() + (step - min_step) / max(max_step - min_step, 1) * rect.width(),
                rect.bottom() - (loss - min_loss) / max(max_loss - min_loss, 1e-9) * rect.height(),
            )
            for step, loss in points
        ]
        painter.setPen(QPen(color, 2))
        for start, end in zip(mapped, mapped[1:]):
            painter.drawLine(start, end)
        painter.setBrush(QBrush(color))
        for point in mapped[-8:]:
            painter.drawEllipse(point, 2.5, 2.5)

class MainWindow(QMainWindow):
    """Main PySide6 window for Micro LLM Creator."""

    def __init__(self) -> None:
        """Create the main application window."""

        super().__init__()
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
        self.training_cards: list[QWidget] = []
        self.training_controls_grid: Optional[QGridLayout] = None
        self.training_controls_columns = 3
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

        self._apply_style()

        shell = self._build_shell()
        self.setCentralWidget(shell)

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
        self.chat_status = QLabel("Chat: no GGUF loaded")
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
        self.benchmark_nav = self._nav_button("◷\nBench")
        self.export_nav = self._nav_button("⇧\nX")
        self.chat_nav = self._nav_button("◌\nChat")
        self._tip(self.dataset_nav, "Open dataset preparation: load text/PDF files and build tokenizer data.")
        self._tip(self.training_nav, "Open model training: configure architecture and optimization settings.")
        self._tip(self.benchmark_nav, "Open benchmark prompts: test checkpoint quality with repeatable prompts.")
        self._tip(self.export_nav, "Open export tools: bundle or quantize the trained model artifacts.")
        self._tip(self.chat_nav, "Open Chat: load a GGUF model once and send prompts.")
        self.dataset_nav.setChecked(True)
        self.dataset_nav.clicked.connect(lambda: self._switch_page(0))
        self.training_nav.clicked.connect(lambda: self._switch_page(1))
        self.benchmark_nav.clicked.connect(lambda: self._switch_page(2))
        self.export_nav.clicked.connect(lambda: self._switch_page(3))
        self.chat_nav.clicked.connect(lambda: self._switch_page(4))
        rail_layout.addWidget(self.dataset_nav)
        rail_layout.addWidget(self.training_nav)
        rail_layout.addWidget(self.benchmark_nav)
        rail_layout.addWidget(self.export_nav)
        rail_layout.addWidget(self.chat_nav)
        rail_layout.addStretch(1)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_dataset_tab())
        self.pages.addWidget(self._build_training_tab())
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
        buttons = [self.dataset_nav, self.training_nav, self.benchmark_nav, self.export_nav, self.chat_nav]
        for button_index, button in enumerate(buttons):
            button.setChecked(button_index == index)
        self._refresh_training_layout()

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

        page = self._panel()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("PageScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("Panel")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 10)
        layout.setSpacing(10)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        title = self._page_title("Data Ingestion Matrix")
        layout.addWidget(title)

        ingestion_body = QHBoxLayout()
        ingestion_body.setSpacing(14)
        left_column = QVBoxLayout()
        left_column.setSpacing(10)
        right_column = QVBoxLayout()
        right_column.setSpacing(10)
        ingestion_body.addLayout(left_column, 1)
        ingestion_body.addLayout(right_column, 1)

        source_form = QFormLayout()
        self._configure_form(source_form)
        tokenizer_form = QFormLayout()
        self._configure_form(tokenizer_form)

        form = QFormLayout()
        self._configure_form(form)

        self.input_dir = QLineEdit()
        self._tip(self.input_dir, "Folder containing PDFs, text, Markdown, or JSONL files. More clean text usually improves the model.")
        self.dataset_dir = QLineEdit(str(Path.cwd() / "runs" / "dataset"))
        self._tip(self.dataset_dir, "Folder where prepared corpus, tokenizer, token files, and dataset summary are saved.")
        self.auto_vocab = QCheckBox("Choose automatically")
        self.auto_vocab.setChecked(True)
        self._tip(self.auto_vocab, "Automatically choose vocabulary size based on corpus size and word variety. Safer for most users.")
        self.manual_vocab_size = self._spin(256, 100000, 8000)
        self.manual_vocab_size.setEnabled(False)
        self._tip(self.manual_vocab_size, "Manual tokenizer vocabulary size. Larger vocab can preserve more words but increases model output size.")
        self.auto_vocab.toggled.connect(lambda checked: self.manual_vocab_size.setEnabled(not checked and not self._tokenizer_strategy_reuses()))
        self.auto_vocab_label = QLabel("Auto after reading files")
        self.auto_vocab_label.setObjectName("Metric")
        self._tip(self.auto_vocab_label, "The actual vocabulary size selected after reading the corpus.")
        self.min_frequency = self._spin(1, 1000, 2)
        self._tip(self.min_frequency, "Minimum token frequency for tokenizer training. Higher values remove rare fragments and can reduce noise.")
        self.context_length = self._spin(16, 4096, 128)
        self._tip(self.context_length, "Number of tokens per training sequence. Longer context lets the model learn longer dependencies but uses more memory.")
        self.validation_split = self._double_spin(0.0, 0.5, 0.1, 0.01, 3)
        self._tip(self.validation_split, "Fraction of tokens held out for validation. Validation helps detect overfitting during training.")
        self.lowercase = QCheckBox("Lowercase text")
        self._tip(self.lowercase, "Convert all text to lowercase. This shrinks vocabulary but removes capitalization patterns.")
        self.max_workers = self._spin(1, 64, 4)
        self._tip(self.max_workers, "Number of parallel file readers. More workers can speed PDF/text loading but uses more CPU and disk activity.")
        self.prepare_mode = QComboBox()
        self.prepare_mode.addItems(["Incremental update", "Full rebuild", "Force reprocess"])
        self.prepare_mode.setMaximumWidth(260)
        self._tip(
            self.prepare_mode,
            "Incremental update reuses cached extracted text and the existing tokenizer. Full rebuild rebuilds tokenizer/tokens. Force reprocess ignores cache.",
        )
        self.tokenizer_strategy = QComboBox()
        self.tokenizer_strategy.addItems(["Auto", "Train new tokenizer", "Reuse dataset tokenizer", "Import tokenizer.json"])
        self.tokenizer_strategy.setMaximumWidth(260)
        self._tip(
            self.tokenizer_strategy,
            "Controls tokenizer reuse. Auto reuses the dataset tokenizer during incremental updates; Import lets you use a compatible tokenizer.json.",
        )
        self.tokenizer_path = QLineEdit()
        self.tokenizer_path.setEnabled(False)
        self._tip(self.tokenizer_path, "Existing tokenizer.json to import. Use this when continuing a compatible tokenizer family.")
        self.tokenizer_strategy.currentTextChanged.connect(self._update_tokenizer_strategy_controls)
        self.code_training_mode = QCheckBox("Code Training Mode")
        self.code_training_mode.setChecked(True)
        self._tip(self.code_training_mode, "Prepare a programming dataset by preserving source code, tagging code/prose, and extracting code-like blocks.")
        self.include_prose = QCheckBox("Include explanations")
        self.include_prose.setChecked(True)
        self._tip(self.include_prose, "Keep prose from PDFs/books. This helps the model learn programming concepts and explanations.")
        self.include_source_code = QCheckBox("Include source files")
        self.include_source_code.setChecked(True)
        self._tip(self.include_source_code, "Include real code files such as .py, .js, .java, .cpp, .cs, .go, .rs, and similar.")
        self.extract_code_blocks = QCheckBox("Extract code blocks")
        self.extract_code_blocks.setChecked(True)
        self._tip(self.extract_code_blocks, "Try to detect code-like blocks inside PDFs/text and train them as code samples.")
        self.preserve_indentation = QCheckBox("Preserve indentation")
        self.preserve_indentation.setChecked(True)
        self._tip(self.preserve_indentation, "Keep line breaks and indentation for code. This is important for Python and readable generated code.")
        self.instruction_samples = QCheckBox("Instruction-style samples")
        self.instruction_samples.setChecked(True)
        self._tip(self.instruction_samples, "Wrap code samples with simple instruction tags so the model sees code as task-oriented examples.")
        self.reasoning_sample_mode = QComboBox()
        self.reasoning_sample_mode.addItems(["Reasoning scaffold", "Detailed code reasoning", "No reasoning wrapper"])
        self.reasoning_sample_mode.setMaximumWidth(260)
        self._tip(
            self.reasoning_sample_mode,
            "Shapes code samples as task/reasoning/answer examples. This teaches response structure, not guaranteed deep reasoning by itself.",
        )
        self.instruction_samples.toggled.connect(self.reasoning_sample_mode.setEnabled)

        source_form.addRow("Source vault", self._path_row(self.input_dir, directory=True))
        source_form.addRow("Dataset core", self._path_row(self.dataset_dir, directory=True))
        source_form.addRow("Parallel lanes", self.max_workers)
        source_form.addRow("Prepare mode", self.prepare_mode)
        source_form.addRow("", self.lowercase)
        source_form.addRow("", self.code_training_mode)
        source_form.addRow("", self.include_source_code)

        tokenizer_form.addRow("Auto vocabulary", self.auto_vocab)
        tokenizer_form.addRow("Manual vocabulary", self.manual_vocab_size)
        tokenizer_form.addRow("Selected vocab", self.auto_vocab_label)
        self.tokenizer_path_row = self._path_row(self.tokenizer_path, directory=False, file_filter="Tokenizer JSON (*.json);;All files (*)")
        self.tokenizer_path_row.setEnabled(False)
        tokenizer_form.addRow("Tokenizer policy", self.tokenizer_strategy)
        tokenizer_form.addRow("Import tokenizer", self.tokenizer_path_row)
        tokenizer_form.addRow("Min frequency", self.min_frequency)
        tokenizer_form.addRow("Context window", self.context_length)
        tokenizer_form.addRow("Validation split", self.validation_split)
        tokenizer_form.addRow("", self.include_prose)
        tokenizer_form.addRow("", self.extract_code_blocks)
        tokenizer_form.addRow("", self.preserve_indentation)
        tokenizer_form.addRow("", self.instruction_samples)
        tokenizer_form.addRow("Reasoning samples", self.reasoning_sample_mode)
        source_card = self._card("SOURCE ARRAY", source_form)
        source_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        tokenizer_card = self._card("TOKENIZER CORE", tokenizer_form)
        tokenizer_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        left_column.addWidget(source_card, 0)
        right_column.addWidget(tokenizer_card, 0)

        quality_grid = QGridLayout()
        quality_grid.setHorizontalSpacing(8)
        quality_grid.setVerticalSpacing(8)
        self.dataset_quality_samples = self._metric_chip("Samples: -", "Training samples after PDF/text/code expansion.")
        self.dataset_quality_tokens = self._metric_chip("Tokens: -", "Total encoded tokens available for training.")
        self.dataset_quality_vocab = self._metric_chip("Vocab: -", "Tokenizer vocabulary size used by the dataset.")
        self.dataset_quality_code = self._metric_chip("Code/prose: -", "Code and prose sample split.")
        self.dataset_quality_cache = self._metric_chip("Cache: -", "Files reused from cache versus processed this run.")
        self.dataset_quality_warning = self._metric_chip("Warnings: none", "Dataset quality warnings, if any.")
        quality_items = [
            self.dataset_quality_samples,
            self.dataset_quality_tokens,
            self.dataset_quality_vocab,
            self.dataset_quality_code,
            self.dataset_quality_cache,
            self.dataset_quality_warning,
        ]
        for index, item in enumerate(quality_items):
            quality_grid.addWidget(item, index // 3, index % 3)
        quality_card = self._card("DATASET QUALITY", quality_grid)
        quality_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        right_column.addWidget(quality_card, 0)

        self.prepare_button = QPushButton("Prepare Dataset")
        self._tip(self.prepare_button, "Read source files, clean text, train tokenizer, split tokens, and save the dataset project.")
        self.prepare_button.clicked.connect(self.prepare_dataset)
        self.prepare_button.setMaximumWidth(320)
        self.stop_dataset_button = QPushButton("Stop")
        self.stop_dataset_button.setEnabled(False)
        self.stop_dataset_button.setMaximumWidth(120)
        self.stop_dataset_button.clicked.connect(self.stop_active_task)
        self._tip(self.stop_dataset_button, "Request a graceful stop for dataset preparation.")

        self.dataset_log = QTextEdit()
        self.dataset_log.setReadOnly(True)
        self.dataset_log.setMinimumHeight(260)
        self.dataset_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.dataset_log, 1)
        left_column.addWidget(self._card("INGEST TELEMETRY", log_layout), 1)
        right_column.addStretch(1)
        layout.addLayout(ingestion_body, 1)
        action_row = QHBoxLayout()
        action_row.addWidget(self.prepare_button)
        action_row.addWidget(self.stop_dataset_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.dataset_progress = self._thin_progress()
        outer.addWidget(self.dataset_progress)
        return page

    def _build_training_tab(self) -> QWidget:
        """Build the training configuration page.

        Returns:
            Training page widget.
        """

        page = self._panel()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("PageScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("Panel")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 10)
        layout.setSpacing(10)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        layout.addWidget(self._page_title("Neural Forge"))
        training_body = QHBoxLayout()
        training_body.setSpacing(12)
        left_zone = QVBoxLayout()
        left_zone.setSpacing(10)
        right_zone = QVBoxLayout()
        right_zone.setSpacing(10)
        training_body.addLayout(left_zone, 2)
        training_body.addLayout(right_zone, 1)
        layout.addLayout(training_body, 1)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        left = QFormLayout()
        self._configure_form(left)
        self.train_data_dir = QLineEdit(str(Path.cwd() / "runs" / "dataset"))
        self._tip(self.train_data_dir, "Prepared dataset folder containing tokenizer.json and train/validation token files.")
        self.model_dir = QLineEdit(str(Path.cwd() / "runs" / "model"))
        self._tip(self.model_dir, "Folder where checkpoints, final model, tokenizer copy, and training summary are saved.")
        self.preset = QComboBox()
        self.preset.addItems(["Tiny", "Small", "Custom"])
        self.preset.setMaximumWidth(260)
        self._tip(self.preset, "Architecture preset. Tiny is faster; Small has more capacity but needs more memory and training data.")
        self.preset.currentTextChanged.connect(self._apply_preset)
        self.architecture_style = QComboBox()
        self.architecture_style.addItems(["Classic GPT", "Llama-like"])
        self.architecture_style.setMaximumWidth(260)
        self._tip(
            self.architecture_style,
            "Classic uses learned positions, LayerNorm, and GELU. Llama-like uses RoPE, RMSNorm, and SwiGLU.",
        )
        self.n_embd = self._spin(32, 4096, 128)
        self._tip(self.n_embd, "Embedding size, also called n_embd. Larger values increase model capacity and memory usage.")
        self.n_head = self._spin(1, 64, 4)
        self._tip(self.n_head, "Attention head count. More heads can model varied relationships, but n_embd must divide evenly by n_head.")
        self.n_layer = self._spin(1, 64, 4)
        self._tip(self.n_layer, "Transformer layer count. More layers improve capacity and reasoning patterns but slow training.")
        self.train_context_length = self._spin(16, 4096, 128)
        self._tip(self.train_context_length, "Training context length in tokens. Must fit your GPU/CPU memory.")
        self.dropout = self._double_spin(0.0, 0.9, 0.1, 0.01, 3)
        self._tip(self.dropout, "Dropout regularization. Higher values reduce overfitting but can slow learning.")
        left.addRow("Dataset", self._path_row(self.train_data_dir, directory=True))
        left.addRow("Model", self._path_row(self.model_dir, directory=True))
        left.addRow("Preset", self.preset)
        left.addRow("Block style", self.architecture_style)
        left.addRow("n_embd", self.n_embd)
        left.addRow("n_head", self.n_head)
        left.addRow("n_layer", self.n_layer)
        left.addRow("Context length", self.train_context_length)
        left.addRow("Dropout", self.dropout)

        right = QFormLayout()
        self._configure_form(right)
        self.epochs = self._spin(1, 10000, 5)
        self._tip(self.epochs, "Number of full passes over the training tokens. More epochs can improve learning or overfit small data.")
        self.batch_size = self._spin(1, 512, 16)
        self._tip(self.batch_size, "Sequences processed per step. Larger batches are smoother but require more memory.")
        self.learning_rate = self._double_spin(0.000001, 1.0, 0.0003, 0.0001, 6)
        self._tip(self.learning_rate, "Optimizer step size. Too high can destabilize training; too low trains slowly.")
        self.weight_decay = self._double_spin(0.0, 1.0, 0.1, 0.01, 4)
        self._tip(self.weight_decay, "Weight decay regularization. Helps control overfitting by discouraging large weights.")
        self.gradient_accumulation = self._spin(1, 256, 1)
        self._tip(self.gradient_accumulation, "Accumulate gradients across batches before updating. Simulates larger batches with less memory.")
        self.warmup_steps = self._spin(0, 1_000_000, 100)
        self._tip(self.warmup_steps, "Steps used to ramp up learning rate. Warmup helps avoid unstable early training.")
        self.eval_interval = self._spin(0, 1_000_000, 100)
        self._tip(self.eval_interval, "Training steps between validation checks. Set 0 to skip interval validation.")
        self.save_interval = self._spin(1, 1_000_000, 500)
        self._tip(self.save_interval, "Training steps between checkpoints. Lower values improve crash recovery but use more disk.")
        self.max_grad_norm = self._double_spin(0.1, 100.0, 1.0, 0.1, 3)
        self._tip(self.max_grad_norm, "Gradient clipping limit. Helps prevent exploding gradients during training.")
        self.seed = self._spin(1, 2_147_483_647, 1337)
        self._tip(self.seed, "Random seed for reproducible initialization and sampling order.")
        self.device = QComboBox()
        self.device.setMaximumWidth(260)
        self._tip(self.device, "Hardware target. CUDA uses NVIDIA GPU when available; CPU is slower but broadly compatible.")
        self.device_info = QLabel()
        self.device_info.setObjectName("Metric")
        self.device_info.setWordWrap(True)
        self.device_info.setMaximumWidth(260)
        self._configure_device_options()
        self.use_amp = QCheckBox("Mixed precision")
        self.use_amp.setChecked(self.use_amp_default)
        self._tip(self.use_amp, "Use mixed precision on CUDA. Usually faster and lighter on GPU memory.")
        self.resume_training = QCheckBox("Resume latest")
        self.resume_training.setChecked(True)
        self._tip(self.resume_training, "Continue from the latest checkpoint if training was interrupted.")
        self.resume_safety = QCheckBox("Safe resume")
        self.resume_safety.setChecked(True)
        self._tip(
            self.resume_safety,
            "Before resuming, verify that the dataset tokenizer and model architecture match the checkpoint.",
        )
        self.resume_checkpoint = QLineEdit()
        self._tip(self.resume_checkpoint, "Optional specific checkpoint file to resume from instead of the latest checkpoint.")
        right.addRow("Epochs", self.epochs)
        right.addRow("Batch", self.batch_size)
        right.addRow("LR", self.learning_rate)
        right.addRow("Decay", self.weight_decay)
        right.addRow("Grad accum", self.gradient_accumulation)
        right.addRow("Warmup", self.warmup_steps)
        right.addRow("Eval every", self.eval_interval)
        right.addRow("Save every", self.save_interval)
        right.addRow("Max grad", self.max_grad_norm)
        right.addRow("Seed", self.seed)
        runtime = QFormLayout()
        self._configure_form(runtime)
        runtime.addRow("Device", self.device)
        runtime.addRow("Hardware", self.device_info)
        runtime.addRow("", self.use_amp)
        runtime.addRow("", self.resume_training)
        runtime.addRow("", self.resume_safety)
        runtime.addRow("Checkpoint", self._path_row(self.resume_checkpoint, directory=False))

        self.training_cards = [
            self._card("MODEL ARCHITECTURE", left),
            self._card("OPTIMIZATION ENGINE", right),
        ]
        for card in self.training_cards:
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        for index, card in enumerate(self.training_cards):
            grid.addWidget(card, 0, index)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.training_controls_grid = grid
        self.training_controls_columns = 2
        controls = QWidget()
        controls.setLayout(grid)
        controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.training_controls = controls
        left_zone.addWidget(controls, 0)
        right_zone.addWidget(self._card("RUNTIME CONTROL", runtime), 0)

        self.train_button = QPushButton("Start Training")
        self._tip(self.train_button, "Start or resume training using the selected model and optimizer settings.")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setMaximumWidth(320)
        self.stop_training_button = QPushButton("Stop")
        self.stop_training_button.setEnabled(False)
        self.stop_training_button.setMaximumWidth(120)
        self.stop_training_button.clicked.connect(self.stop_active_task)
        self._tip(self.stop_training_button, "Request a graceful stop and save a resumable checkpoint.")

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(8)
        metrics_grid.setVerticalSpacing(8)
        self.training_epoch_metric = self._metric_chip("Epoch: -", "Current epoch and total epochs.")
        self.training_step_metric = self._metric_chip("Step: -", "Current optimizer step and total planned steps.")
        self.training_loss_metric = self._metric_chip("Train loss: -", "Latest training loss. Lower is usually better.")
        self.training_val_metric = self._metric_chip("Val loss: -", "Latest validation loss when validation is enabled.")
        self.training_lr_metric = self._metric_chip("LR: -", "Current learning rate from the scheduler.")
        self.training_speed_metric = self._metric_chip("Speed: -", "Current training throughput.")
        self.training_grad_metric = self._metric_chip("Grad: -", "Current gradient norm.")
        self.training_vram_metric = self._metric_chip("VRAM: -", "Current CUDA memory usage when training on GPU.")
        self.training_eta_metric = self._metric_chip("ETA: -", "Estimated time remaining based on recent optimizer steps.")
        for index, metric in enumerate((
            self.training_eta_metric,
            self.training_epoch_metric,
            self.training_step_metric,
            self.training_loss_metric,
            self.training_val_metric,
            self.training_lr_metric,
            self.training_speed_metric,
            self.training_grad_metric,
            self.training_vram_metric,
        )):
            metrics_grid.addWidget(metric, index, 0)
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(8)
        metrics_layout.addLayout(metrics_grid)
        self.loss_chart = LossChartWidget()
        self._tip(self.loss_chart, "Live training and validation loss. Falling values usually mean the model is learning.")
        self.optimization_chart = LossChartWidget("LR", "Grad", "Learning rate and gradient norm will appear during training")
        self._tip(self.optimization_chart, "Learning rate and gradient norm. Watch for unstable spikes or gradients collapsing toward zero.")
        self.stability_chart = LossChartWidget("Weight", "Update", "Weight norm and update ratio will appear during training")
        self._tip(self.stability_chart, "Weight norm and parameter update ratio. Large update ratios can destabilize training; tiny ratios can stall learning.")
        self.throughput_chart = LossChartWidget("Tok/s", "Samples/s", "Throughput will appear during training")
        self._tip(self.throughput_chart, "Training speed measured as tokens/sec and samples/sec.")
        self.memory_chart = LossChartWidget("VRAM alloc", "VRAM reserved", "VRAM usage will appear during CUDA training")
        self._tip(self.memory_chart, "CUDA memory usage in GB. Helps diagnose memory bottlenecks.")
        charts_grid = QGridLayout()
        charts_grid.setHorizontalSpacing(8)
        charts_grid.setVerticalSpacing(8)
        charts_grid.addWidget(self.loss_chart, 0, 0)
        charts_grid.addWidget(self.optimization_chart, 0, 1)
        charts_grid.addWidget(self.stability_chart, 1, 0)
        charts_grid.addWidget(self.throughput_chart, 1, 1)
        charts_grid.addWidget(self.memory_chart, 2, 0, 1, 2)
        charts_grid.setColumnStretch(0, 1)
        charts_grid.setColumnStretch(1, 1)
        left_zone.addWidget(self._card("TRAINING GRAPHS", charts_grid), 1)
        right_zone.addWidget(self._card("TRAINING METRICS", metrics_layout), 0)

        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMinimumHeight(320)
        self.training_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        telemetry_layout = QVBoxLayout()
        telemetry_layout.addWidget(self.training_log, 1)
        right_zone.addWidget(self._card("TRAINING TELEMETRY", telemetry_layout), 1)

        action_row = QHBoxLayout()
        action_row.addWidget(self.train_button)
        action_row.addWidget(self.stop_training_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.training_progress = self._thin_progress()
        outer.addWidget(self.training_progress)
        QTimer.singleShot(0, self._refresh_training_layout)
        return page

    def _build_export_tab(self) -> QWidget:
        """Build the export page.

        Returns:
            Export page widget.
        """

        page = self._panel()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("PageScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("Panel")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 10)
        layout.setSpacing(10)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        layout.addWidget(self._page_title("Export Bay"))

        form = QFormLayout()
        self._configure_form(form)
        self.export_model_dir = QLineEdit(str(Path.cwd() / "runs" / "model"))
        self._tip(self.export_model_dir, "Trained model folder containing final_model.pt and tokenizer.json.")
        self.export_dir = QLineEdit(str(Path.cwd() / "runs" / "export"))
        self._tip(self.export_dir, "Folder where export bundles or quantized checkpoints are written.")
        self.quant_mode = QComboBox()
        self.quant_mode.addItems(["FP16 checkpoint", "GGUF Q8_0 (planned)", "GGUF Q4_K_M (planned)", "GGUF Q5_K_M (planned)"])
        self.quant_mode.setMaximumWidth(260)
        self._tip(self.quant_mode, "Quantization target. FP16 reduces checkpoint size now; GGUF modes are planned for llama.cpp export.")
        self.llama_cpp_dir = QLineEdit()
        self._tip(self.llama_cpp_dir, "Local llama.cpp checkout containing convert_hf_to_gguf.py.")
        self.gguf_output_path = QLineEdit(str(Path.cwd() / "runs" / "export" / "model.gguf"))
        self._tip(self.gguf_output_path, "Destination GGUF file. Requires an HF-compatible hf_model folder in the model core.")
        self.gguf_outtype = QComboBox()
        self.gguf_outtype.addItems(["f16", "f32", "bf16", "q8_0"])
        self.gguf_outtype.setMaximumWidth(260)
        self._tip(self.gguf_outtype, "llama.cpp converter outtype. f16 is the usual starting point.")
        form.addRow("Model core", self._path_row(self.export_model_dir, directory=True))
        form.addRow("Output bay", self._path_row(self.export_dir, directory=True))
        form.addRow("Quantization", self.quant_mode)
        form.addRow("llama.cpp", self._path_row(self.llama_cpp_dir, directory=True))
        form.addRow("GGUF output", self._path_row(self.gguf_output_path, directory=False, file_filter="GGUF models (*.gguf);;All files (*)"))
        form.addRow("GGUF outtype", self.gguf_outtype)
        layout.addWidget(self._card("ARTIFACT CONFIGURATION", form))

        row = QHBoxLayout()
        row.setSpacing(10)
        bundle_button = QPushButton("Create Bundle")
        self._tip(bundle_button, "Copy final model, tokenizer, and summary into a portable export folder.")
        bundle_button.clicked.connect(self.create_bundle)
        quant_button = QPushButton("Quantize Model")
        self._tip(quant_button, "Create a smaller FP16 checkpoint for inference or later conversion workflows.")
        quant_button.clicked.connect(self.quantize_model)
        hf_button = QPushButton("Export HF Package")
        self._tip(hf_button, "Create model_core/hf_model with config, weights, tokenizer, lineage, and README.")
        hf_button.clicked.connect(self.export_hf_package)
        self.gguf_convert_button = QPushButton("Convert HF to GGUF")
        self._tip(self.gguf_convert_button, "Run llama.cpp convert_hf_to_gguf.py for model_core/hf_model when available.")
        self.gguf_convert_button.clicked.connect(self.convert_hf_to_gguf)
        bundle_button.setMaximumWidth(220)
        quant_button.setMaximumWidth(220)
        hf_button.setMaximumWidth(220)
        self.gguf_convert_button.setMaximumWidth(220)
        row.addWidget(bundle_button)
        row.addWidget(quant_button)
        row.addWidget(hf_button)
        row.addWidget(self.gguf_convert_button)
        row.addStretch(1)

        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setMinimumHeight(320)
        self.export_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.export_log.setPlainText(
            "Export options:\n"
            "- Bundle copies final_model.pt, tokenizer.json, and training_summary.json.\n"
            "- HF package writes model_core/hf_model for portable MicroGPT loading.\n"
            "- FP16 checkpoint quantization works now.\n"
            "- GGUF conversion uses llama.cpp when model_core/hf_model exists.\n"
            "- Native MicroGPT checkpoints are not written as fake GGUF files.\n"
        )
        export_log_layout = QVBoxLayout()
        export_log_layout.addWidget(self.export_log, 1)
        layout.addWidget(self._card("EXPORT TELEMETRY", export_log_layout), 1)
        layout.addLayout(row)

        self.export_progress = self._thin_progress()
        outer.addWidget(self.export_progress)
        return page

    def _build_benchmark_tab(self) -> QWidget:
        """Build the benchmark prompt page.

        Returns:
            Benchmark page widget.
        """

        page = self._panel()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea()
        scroll.setObjectName("PageScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("Panel")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 10)
        layout.setSpacing(10)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        layout.addWidget(self._page_title("Benchmark Console"))

        benchmark_grid = QGridLayout()
        benchmark_grid.setHorizontalSpacing(12)
        benchmark_grid.setVerticalSpacing(8)
        self.benchmark_prompts = QTextEdit()
        self.benchmark_prompts.setMinimumHeight(260)
        self.benchmark_prompts.setPlainText("\n\n".join(DEFAULT_BENCHMARK_PROMPTS))
        self._tip(self.benchmark_prompts, "Benchmark prompts separated by blank lines. Run the same prompts after each training run.")
        self.benchmark_tokens = self._spin(16, 1024, 128)
        self._tip(self.benchmark_tokens, "Maximum generated tokens per benchmark prompt.")
        self.benchmark_temperature = self._double_spin(0.0, 2.0, 0.7, 0.05, 2)
        self._tip(self.benchmark_temperature, "Sampling randomness for benchmark generation.")
        self.benchmark_kv_cache = QCheckBox("Use KV cache")
        self.benchmark_kv_cache.setChecked(True)
        self._tip(self.benchmark_kv_cache, "Reuse attention key/value tensors during MicroGPT benchmark generation for faster inference.")
        self.run_benchmark_button = QPushButton("Run Benchmark")
        self.run_benchmark_button.setMaximumWidth(180)
        self.run_benchmark_button.clicked.connect(self.run_benchmark)
        self._tip(self.run_benchmark_button, "Generate benchmark outputs from final_model.pt and save a benchmark JSON file.")
        self.stop_benchmark_button = QPushButton("Stop")
        self.stop_benchmark_button.setMaximumWidth(120)
        self.stop_benchmark_button.setEnabled(False)
        self.stop_benchmark_button.clicked.connect(self.stop_active_task)
        self._tip(self.stop_benchmark_button, "Request a graceful stop for benchmark generation.")
        benchmark_grid.addWidget(self.benchmark_prompts, 0, 0, 5, 1)
        benchmark_grid.addWidget(QLabel("Max tokens"), 0, 1)
        benchmark_grid.addWidget(self.benchmark_tokens, 0, 2)
        benchmark_grid.addWidget(QLabel("Temperature"), 1, 1)
        benchmark_grid.addWidget(self.benchmark_temperature, 1, 2)
        benchmark_grid.addWidget(self.benchmark_kv_cache, 2, 1, 1, 2)
        benchmark_grid.addWidget(self.run_benchmark_button, 3, 1, 1, 2)
        benchmark_grid.addWidget(self.stop_benchmark_button, 4, 1, 1, 2)
        benchmark_grid.setColumnStretch(0, 1)
        layout.addWidget(self._card("BENCHMARK PROMPTS", benchmark_grid), 0)

        self.benchmark_log = QTextEdit()
        self.benchmark_log.setReadOnly(True)
        self.benchmark_log.setMinimumHeight(260)
        self.benchmark_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        benchmark_log_layout = QVBoxLayout()
        benchmark_log_layout.addWidget(self.benchmark_log, 1)
        layout.addWidget(self._card("BENCHMARK TELEMETRY", benchmark_log_layout), 1)

        self.benchmark_progress = self._thin_progress()
        outer.addWidget(self.benchmark_progress)
        return page

    def _build_chat_tab(self) -> QWidget:
        """Build the GGUF model test chat page.

        Returns:
            Chat page widget.
        """

        page = self._panel()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 14)
        layout.setSpacing(12)

        main = QHBoxLayout()
        main.setSpacing(14)

        chat_column = QVBoxLayout()
        chat_column.setSpacing(10)

        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("ChatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setMinimumHeight(420)
        self._tip(self.chat_scroll, "Rendered Markdown conversation view.")
        self.chat_canvas = QWidget()
        self.chat_canvas.setObjectName("ChatCanvas")
        self.chat_messages = QVBoxLayout(self.chat_canvas)
        self.chat_messages.setContentsMargins(14, 14, 14, 14)
        self.chat_messages.setSpacing(12)
        self.chat_messages.addStretch(1)
        self.chat_scroll.setWidget(self.chat_canvas)
        self.chat_event_log = QTextEdit()
        self.chat_event_log.setVisible(False)
        self._add_chat_message("assistant", "Load a GGUF model to start testing.")
        self.chat_stats = QLabel("Idle")
        self.chat_stats.setObjectName("Metric")
        self.chat_stats.setVisible(False)
        self._tip(self.chat_stats, "Generation timing, produced tokens, and approximate token speed.")
        chat_column.addWidget(self.chat_scroll, 1)

        prompt_row = QHBoxLayout()
        prompt_row.setSpacing(10)
        self.chat_input = ChatInputEdit()
        self.chat_input.setObjectName("ChatInput")
        self.chat_input.setMaximumHeight(92)
        self.chat_input.setPlaceholderText("Send a message...")
        self._tip(self.chat_input, "Prompt to send to the loaded model.")
        self.chat_input.sendRequested.connect(self.send_chat_message)
        self.send_chat_button = QPushButton("Send")
        self.send_chat_button.setMaximumWidth(120)
        self.send_chat_button.clicked.connect(self.send_chat_message)
        self._tip(self.send_chat_button, "Send the message to the already loaded model.")
        self.stop_chat_button = QPushButton("Stop")
        self.stop_chat_button.setMaximumWidth(120)
        self.stop_chat_button.setEnabled(False)
        self.stop_chat_button.clicked.connect(self.stop_active_task)
        self._tip(self.stop_chat_button, "Stop the current streamed reply.")
        prompt_row.addWidget(self.chat_input, 1)
        prompt_row.addWidget(self.send_chat_button)
        prompt_row.addWidget(self.stop_chat_button)
        chat_column.addLayout(prompt_row)

        settings_column = QVBoxLayout()
        settings_column.setSpacing(12)
        settings_panel = QWidget()
        settings_panel.setMaximumWidth(390)
        settings_panel.setMinimumWidth(340)
        settings_panel.setLayout(settings_column)

        model_form = QFormLayout()
        self._configure_form(model_form)
        self.gguf_path = QLineEdit()
        self._tip(self.gguf_path, "Path to a GGUF model file produced by llama.cpp-compatible export tooling.")
        self.llama_context = self._spin(256, 131072, 2048)
        self._tip(self.llama_context, "llama.cpp context window. Larger values allow longer chats but use more memory.")
        self.llama_threads = self._spin(1, 128, 4)
        self._tip(self.llama_threads, "CPU threads used by llama.cpp inference.")
        self.llama_gpu_layers = self._spin(-1, 200, -1)
        self._tip(self.llama_gpu_layers, "Number of transformer layers to offload to GPU. Use -1 to offload all possible layers.")
        model_form.addRow("GGUF model", self._path_row(self.gguf_path, directory=False, file_filter="GGUF models (*.gguf);;All files (*)"))
        model_form.addRow("Context", self.llama_context)
        model_form.addRow("CPU threads", self.llama_threads)
        model_form.addRow("GPU layers", self.llama_gpu_layers)
        self.load_llm_button = QPushButton("Load Model")
        self.load_llm_button.setMaximumWidth(180)
        self.load_llm_button.clicked.connect(self.toggle_llm_model)
        self._tip(self.load_llm_button, "Load the GGUF model into memory once for repeated chat messages.")
        self.reset_chat_button = QPushButton("Reset Chat")
        self.reset_chat_button.setMaximumWidth(180)
        self.reset_chat_button.clicked.connect(self.reset_chat)
        self._tip(self.reset_chat_button, "Clear conversation memory while keeping the model loaded.")
        loader_buttons = QHBoxLayout()
        loader_buttons.addWidget(self.load_llm_button)
        loader_buttons.addWidget(self.reset_chat_button)
        loader_buttons.addStretch(1)
        model_form.addRow("", loader_buttons)

        sample_form = QFormLayout()
        self._configure_form(sample_form)
        self.thinking_enabled = QCheckBox("Thinking")
        self.thinking_enabled.setChecked(True)
        self._tip(self.thinking_enabled, "When enabled, the prompt asks the model to reason according to the selected effort level. Turn off for direct answers.")
        self.reasoning_effort = QComboBox()
        self.reasoning_effort.addItems(["Balanced", "Fast", "Deep"])
        self.reasoning_effort.setMaximumWidth(260)
        self._tip(self.reasoning_effort, "Controls the instruction style sent with each prompt. Deep asks for more careful reasoning.")
        self.thinking_enabled.toggled.connect(self.reasoning_effort.setEnabled)
        self.chat_max_tokens = self._spin(16, 8192, 512)
        self._tip(self.chat_max_tokens, "Maximum new tokens for each assistant reply.")
        self.chat_temperature = self._double_spin(0.0, 2.0, 0.7, 0.05, 2)
        self._tip(self.chat_temperature, "Sampling randomness. Lower is more focused; higher is more creative.")
        self.chat_top_p = self._double_spin(0.01, 1.0, 0.9, 0.01, 2)
        self._tip(self.chat_top_p, "Nucleus sampling. Lower values restrict the model to more likely tokens.")
        self.chat_repeat_penalty = self._double_spin(0.8, 2.0, 1.1, 0.01, 2)
        self._tip(self.chat_repeat_penalty, "Penalty for repeated text. Higher can reduce loops.")
        sample_form.addRow("", self.thinking_enabled)
        sample_form.addRow("Reasoning effort", self.reasoning_effort)
        sample_form.addRow("Max tokens", self.chat_max_tokens)
        sample_form.addRow("Temperature", self.chat_temperature)
        sample_form.addRow("Top-p", self.chat_top_p)
        sample_form.addRow("Repeat penalty", self.chat_repeat_penalty)

        self.system_prompt = QTextEdit()
        self.system_prompt.setObjectName("SystemPrompt")
        self.system_prompt.setMaximumHeight(120)
        self.system_prompt.setPlaceholderText("Optional system prompt")
        self._tip(self.system_prompt, "Optional behavior instruction sent to the model with each message.")
        system_layout = QVBoxLayout()
        system_layout.addWidget(self.system_prompt)

        settings_column.addWidget(self._card("MODEL LOADER", model_form))
        settings_column.addWidget(self._card("RESPONSE TUNING", sample_form))
        settings_column.addWidget(self._card("SYSTEM PROMPT", system_layout))
        settings_column.addStretch(1)

        main.addLayout(chat_column, 1)
        main.addWidget(settings_panel)
        layout.addLayout(main, 1)

        self.chat_progress = self._thin_progress()
        layout.addWidget(self.chat_progress)
        return page

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
        cached_count = int(summary.get("cached_file_count", 0) or 0)
        processed_count = int(summary.get("processed_file_count", 0) or 0)
        skipped_count = int(summary.get("skipped_file_count", 0) or 0)
        failed_count = int(summary.get("failed_file_count", 0) or 0)
        warning = str(summary.get("warning") or "none")
        self.dataset_quality_samples.setText(f"Samples: {document_count:,}")
        self.dataset_quality_tokens.setText(f"Tokens: {token_count:,}")
        self.dataset_quality_vocab.setText(f"Vocab: {vocab_size:,}" if vocab_size else "Vocab: -")
        self.dataset_quality_code.setText(f"Code/prose: {code_count:,}/{prose_count:,}")
        self.dataset_quality_cache.setText(f"Files: {processed_count:,} ok, {cached_count:,} cached, {skipped_count:,} skipped, {failed_count:,} failed")
        self.dataset_quality_warning.setText(f"Warnings: {warning}")
        self._tip(self.dataset_quality_samples, f"{character_count:,} source characters across prepared samples.")

    def _reset_dataset_quality_report(self) -> None:
        """Reset dataset quality chips to their empty state."""

        self.dataset_quality_samples.setText("Samples: -")
        self.dataset_quality_tokens.setText("Tokens: -")
        self.dataset_quality_vocab.setText("Vocab: -")
        self.dataset_quality_code.setText("Code/prose: -")
        self.dataset_quality_cache.setText("Cache: -")
        self.dataset_quality_warning.setText("Warnings: none")

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

    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert Markdown to themed HTML.

        Args:
            markdown_text: Markdown content.

        Returns:
            HTML suitable for a chat bubble.
        """

        try:
            import markdown as markdown_lib
            from pygments import highlight
            from pygments.formatters import HtmlFormatter
            from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer

            markdown_text = self._normalize_code_blocks(markdown_text)
            body_parts: list[str] = []
            pattern = re.compile(r"```(?P<lang>[\w+-]*)\n(?P<code>.*?)```", re.DOTALL)
            last = 0
            formatter = HtmlFormatter(style="monokai", noclasses=True, nowrap=True)
            for match in pattern.finditer(markdown_text):
                prose = markdown_text[last:match.start()]
                if prose.strip():
                    body_parts.append(markdown_lib.markdown(prose, extensions=["tables", "nl2br"]))
                code = match.group("code")
                lang = match.group("lang").strip()
                try:
                    lexer = get_lexer_by_name(lang) if lang else guess_lexer(code)
                except Exception:
                    lexer = TextLexer()
                highlighted = highlight(code, lexer, formatter)
                label = lang.title() if lang else lexer.name
                body_parts.append(self._code_block_html(label, highlighted, code))
                last = match.end()
            prose = markdown_text[last:]
            if prose.strip():
                body_parts.append(markdown_lib.markdown(prose, extensions=["tables", "nl2br"]))
            body = "\n".join(body_parts) if body_parts else ""
            return (
                f"""<!doctype html>
                <html>
                <head>
                <style>
                body {{
                    background: transparent;
                    color: #eeeeee;
                    font-family: Arial, "Segoe UI", sans-serif;
                    font-size: 14px;
                    line-height: 1.22;
                    margin: 0;
                }}
                h1 {{ color: #f2f2f2; font-size: 19px; margin: 6px 0 3px 0; }}
                h2 {{ color: #f2f2f2; font-size: 17px; margin: 6px 0 3px 0; }}
                h3 {{ color: #f2f2f2; font-size: 15px; margin: 5px 0 2px 0; }}
                p {{ margin: 2px 0; }}
                ol, ul {{ margin-top: 2px; margin-bottom: 2px; padding-left: 20px; }}
                li {{ margin: 1px 0; }}
                code {{
                    background: #1a1a1a;
                    color: #d4d4d4;
                    border-radius: 4px;
                    padding: 2px 4px;
                    font-family: Consolas, monospace;
                }}
                pre {{
                    background: transparent;
                    border: 0;
                    border-radius: 0;
                    padding: 0;
                    margin: 4px 0;
                    overflow: auto;
                    white-space: pre-wrap;
                }}
                pre code {{
                    background: transparent;
                    padding: 0;
                    color: #d4d4d4;
                    font-family: Consolas, monospace;
                    font-size: 13px;
                    line-height: 1.16;
                }}
                blockquote {{
                    border-left: 3px solid #f5b041;
                    margin-left: 0;
                    padding-left: 12px;
                    color: #cccccc;
                }}
                table {{ border-collapse: collapse; }}
                th, td {{ border: 1px solid #555555; padding: 6px 8px; }}
                .codeblock {{
                    background: #050505;
                    border: 1px solid #2b2b2b;
                    border-radius: 12px;
                    margin: 8px 0;
                }}
                .codebar {{
                    color: #f2f2f2;
                    background: #111111;
                    border-bottom: 1px solid #2b2b2b;
                    padding: 7px 10px;
                    font-size: 12px;
                    font-weight: bold;
                }}
                .copylink {{
                    color: #d7d7d7;
                    text-decoration: none;
                    float: right;
                    font-weight: normal;
                }}
                .codebody {{ padding: 12px 14px; }}
                </style>
                </head>
                <body>{body}</body>
                </html>
                """
            )
        except Exception:
            return self._basic_markdown_html(markdown_text)

    def _basic_markdown_html(self, markdown_text: str) -> str:
        """Render basic Markdown with simple code coloring.

        Args:
            markdown_text: Raw Markdown.

        Returns:
            Basic HTML.
        """

        text = self._normalize_code_blocks(markdown_text)
        parts: list[str] = []
        pattern = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
        last = 0
        for match in pattern.finditer(text):
            parts.append(self._render_basic_prose(text[last:match.start()]))
            code = match.group(1)
            parts.append(self._code_block_html("Code", self._colorize_code(code), code))
            last = match.end()
        parts.append(self._render_basic_prose(text[last:]))
        return (
            "<html><body style='background:transparent;color:#eeeeee;font-family:Arial;font-size:14px;line-height:1.22;'>"
            "<style>"
            "p{margin:2px 0;} h1{font-size:19px;margin:6px 0 3px;} h2{font-size:17px;margin:6px 0 3px;}"
            "h3{font-size:15px;margin:5px 0 2px;} ol,ul{margin-top:2px;margin-bottom:2px;padding-left:20px;} li{margin:1px 0;}"
            "code{background:#1a1a1a;color:#d4d4d4;border-radius:4px;padding:2px 4px;font-family:Consolas,monospace;}"
            "pre{background:transparent;border:0;border-radius:0;padding:0;margin:4px 0;"
            "font-family:Consolas,monospace;white-space:pre-wrap;line-height:1.16;font-size:13px;}"
            ".codeblock{background:#050505;border:1px solid #2b2b2b;border-radius:12px;margin:8px 0;}"
            ".codebar{color:#f2f2f2;background:#111;border-bottom:1px solid #2b2b2b;padding:7px 10px;font-size:12px;font-weight:bold;}"
            ".copylink{color:#d7d7d7;text-decoration:none;float:right;font-weight:normal;}.codebody{padding:12px 14px;}"
            "</style>"
            + "".join(parts)
            + "</body></html>"
        )

    def _code_block_html(self, label: str, highlighted_html: str, raw_code: str) -> str:
        """Build a code panel with a copy link.

        Args:
            label: Code language label.
            highlighted_html: Highlighted code HTML.
            raw_code: Raw code for clipboard copy.

        Returns:
            Code panel HTML.
        """

        return (
            "<div class='codeblock'>"
            f"<div class='codebar'>{self._escape_html(label or 'Code')}"
            f"<a class='copylink' href='copycode:{quote(raw_code)}'>⧉ Copy</a></div>"
            f"<div class='codebody'><pre><code>{highlighted_html}</code></pre></div>"
            "</div>"
        )

    def _render_basic_prose(self, text: str) -> str:
        """Render a small Markdown subset for fallback mode.

        Args:
            text: Markdown prose.

        Returns:
            HTML fragment.
        """

        html_lines: list[str] = []
        in_ordered = False
        in_unordered = False

        def close_lists() -> None:
            nonlocal in_ordered, in_unordered
            if in_ordered:
                html_lines.append("</ol>")
                in_ordered = False
            if in_unordered:
                html_lines.append("</ul>")
                in_unordered = False

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                close_lists()
                html_lines.append("<br>")
                continue
            if line.startswith("### "):
                close_lists()
                html_lines.append(f"<h3>{self._inline_basic_markdown(line[4:])}</h3>")
                continue
            if line.startswith("## "):
                close_lists()
                html_lines.append(f"<h2>{self._inline_basic_markdown(line[3:])}</h2>")
                continue
            if line.startswith("# "):
                close_lists()
                html_lines.append(f"<h1>{self._inline_basic_markdown(line[2:])}</h1>")
                continue
            ordered = re.match(r"^\d+\.\s+(.*)$", line)
            if ordered:
                if not in_ordered:
                    close_lists()
                    html_lines.append("<ol>")
                    in_ordered = True
                html_lines.append(f"<li>{self._inline_basic_markdown(ordered.group(1))}</li>")
                continue
            unordered = re.match(r"^[-*]\s+(.*)$", line)
            if unordered:
                if not in_unordered:
                    close_lists()
                    html_lines.append("<ul>")
                    in_unordered = True
                html_lines.append(f"<li>{self._inline_basic_markdown(unordered.group(1))}</li>")
                continue
            close_lists()
            html_lines.append(f"<p>{self._inline_basic_markdown(line)}</p>")
        close_lists()
        return "\n".join(html_lines)

    def _inline_basic_markdown(self, text: str) -> str:
        """Render inline Markdown for fallback mode.

        Args:
            text: Inline Markdown text.

        Returns:
            HTML fragment.
        """

        escaped = self._escape_html(text)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
        return escaped

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape text for HTML.

        Args:
            text: Raw text.

        Returns:
            Escaped text.
        """

        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _colorize_code(self, code: str) -> str:
        """Apply simple inline colors to Python-like code.

        Args:
            code: Source code.

        Returns:
            HTML code.
        """

        escaped = self._escape_html(code)
        keywords = {
            "def", "class", "import", "from", "for", "while", "if", "else", "elif",
            "try", "except", "return", "print", "with", "as", "in", "function", "const",
            "let", "var", "new", "typeof", "await", "async", "true", "false", "null",
            "True", "False", "None",
        }
        builtins = {"console", "Object", "process", "JSON", "Array", "String", "Number", "Boolean", "Math", "os", "sys"}
        token_pattern = re.compile(
            r"(?P<comment>//.*|#.*)"
            r"|(?P<string>`(?:\\.|[^`])*`|'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\")"
            r"|(?P<number>\b\d+(?:\.\d+)?\b)"
            r"|(?P<word>\b[A-Za-z_][A-Za-z0-9_]*\b)"
        )
        colored_lines: list[str] = []
        for line in escaped.splitlines():
            segments: list[str] = []
            last = 0
            for match in token_pattern.finditer(line):
                segments.append(line[last:match.start()])
                value = match.group(0)
                if match.lastgroup == "comment":
                    segments.append(f"<span style='color:#6a9955;'>{value}</span>")
                elif match.lastgroup == "string":
                    segments.append(f"<span style='color:#ce9178;'>{value}</span>")
                elif match.lastgroup == "number":
                    segments.append(f"<span style='color:#b5cea8;'>{value}</span>")
                elif match.lastgroup == "word":
                    next_chars = line[match.end(): match.end() + 2]
                    previous = line[max(0, match.start() - 1): match.start()]
                    if value in keywords:
                        segments.append(f"<span style='color:#569cd6;font-weight:bold;'>{value}</span>")
                    elif value in builtins:
                        segments.append(f"<span style='color:#4ec9b0;'>{value}</span>")
                    elif next_chars.startswith("(") and previous != ".":
                        segments.append(f"<span style='color:#dcdcaa;'>{value}</span>")
                    elif previous == ".":
                        segments.append(f"<span style='color:#9cdcfe;'>{value}</span>")
                    else:
                        segments.append(value)
                last = match.end()
            segments.append(line[last:])
            colored_lines.append("".join(segments))
        return "\n".join(colored_lines)

    def _normalize_code_blocks(self, markdown_text: str) -> str:
        """Fence obvious loose code blocks so syntax highlighting can run.

        Args:
            markdown_text: Raw model Markdown.

        Returns:
            Markdown with likely code blocks fenced.
        """

        if "```" in markdown_text:
            if markdown_text.count("```") % 2:
                return f"{markdown_text}\n```"
            return markdown_text
        lines = markdown_text.splitlines()
        normalized: list[str] = []
        code_block: list[str] = []

        def is_code_line(line: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return bool(code_block)
            if line.startswith(("    ", "\t")):
                return True
            if re.match(
                r"^(def|class|import|from|for|while|if|else:?|elif|try:?|except|return|print|with|"
                r"function|const|let|var|console\.|Object\.|process\.)\b",
                stripped,
            ):
                return True
            if stripped in {"{", "}", "};", "})", "});"}:
                return True
            if stripped.startswith(("#", "@")):
                return True
            return sum(stripped.count(symbol) for symbol in "()[]{}:=<>+-*/") >= 3

        def flush() -> None:
            nonlocal code_block
            if len([line for line in code_block if line.strip()]) >= 3:
                normalized.append(f"```{self._guess_code_language(code_block)}")
                normalized.extend(code_block)
                normalized.append("```")
            else:
                normalized.extend(code_block)
            code_block = []

        for line in lines:
            if is_code_line(line):
                code_block.append(line)
            else:
                flush()
                normalized.append(line)
        flush()
        return "\n".join(normalized)

    @staticmethod
    def _guess_code_language(lines: list[str]) -> str:
        """Guess a fence language for loose code.

        Args:
            lines: Code lines.

        Returns:
            Markdown fence language.
        """

        joined = "\n".join(lines).lower()
        if any(marker in joined for marker in ("console.", "const ", "let ", "function ", "process.env", "object.keys")):
            return "javascript"
        if any(marker in joined for marker in ("#include", "std::", "cout", "cin")):
            return "cpp"
        if any(marker in joined for marker in ("public class", "system.out", "private ", "protected ")):
            return "java"
        if any(marker in joined for marker in ("def ", "import ", "print(", "self.")):
            return "python"
        return "text"

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
            self._markdown_to_html,
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
        project_file.write_text(json.dumps(self._project_state_dict(project_name, project_dir), indent=2), encoding="utf-8")
        self.current_project_file = project_file
        self.project_state.setText("Project saved")
        if self.current_project_file == project_file:
            self.dataset_log.append(f"Project saved: {project_file}")

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
        self.current_project_file = None
        self._apply_project_state(self._default_project_state())
        self._reset_project_runtime_state()
        self.project_state.setText("New project")
        self.dataset_log.append("Started a new project.")

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
        self.project_state.setText("Project opened")
        self.dataset_log.append(f"Opened project: {project_file}")

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
                "tokenizer_import": "",
                "resume_checkpoint": "",
            },
            "dataset": {
                "auto_vocab": True,
                "manual_vocab_size": 8000,
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
                "n_embd": 128,
                "n_head": 4,
                "n_layer": 4,
                "context_length": 128,
                "dropout": 0.1,
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.0003,
                "weight_decay": 0.1,
                "gradient_accumulation": 1,
                "warmup_steps": 100,
                "eval_interval": 100,
                "save_interval": 500,
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
            "artifacts": {},
        }

    def _reset_project_runtime_state(self) -> None:
        """Clear logs, progress, charts, and status labels for a new project."""

        self.dataset_log.clear()
        self.training_log.clear()
        self.benchmark_log.clear()
        self.export_log.setPlainText(
            "Export options:\n"
            "- Bundle copies final_model.pt, tokenizer.json, and training_summary.json.\n"
            "- HF package writes model_core/hf_model for portable MicroGPT loading.\n"
            "- FP16 checkpoint quantization works now.\n"
            "- GGUF conversion uses llama.cpp when model_core/hf_model exists.\n"
            "- Native MicroGPT checkpoints are not written as fake GGUF files.\n"
        )
        for progress in (self.dataset_progress, self.training_progress, self.benchmark_progress, self.export_progress, self.chat_progress):
            progress.setRange(0, 100)
            progress.setValue(0)
        self.dataset_status.setText("Dataset: not prepared")
        self.train_status.setText("Training: idle")
        self.export_status.setText("Export: waiting")
        self.chat_status.setText("Chat: no GGUF loaded")
        self.prepare_button.setText("Prepare Dataset")
        self.train_button.setText("Start Training")
        self.stop_dataset_button.setEnabled(False)
        self.stop_training_button.setEnabled(False)
        self.stop_benchmark_button.setEnabled(False)
        self.stop_chat_button.setEnabled(False)
        self.load_llm_button.setText("Load Model")
        self._tip(self.load_llm_button, "Load the GGUF model into memory once for repeated chat messages.")
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
        self.loss_chart.clear()
        self.optimization_chart.clear()
        self.stability_chart.clear()
        self.throughput_chart.clear()
        self.memory_chart.clear()
        self._clear_chat_messages()
        self.chat_markdown = ""
        self.chat_stream_prefix = ""
        self.chat_stream_reply = ""
        self.chat_stats.setText("Idle")
        self._add_chat_message("assistant", "Load a GGUF model to start testing.")

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
                "tokenizer_import": self.tokenizer_path.text(),
                "resume_checkpoint": self.resume_checkpoint.text(),
            },
            "dataset": {
                "auto_vocab": self.auto_vocab.isChecked(),
                "manual_vocab_size": self.manual_vocab_size.value(),
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
                "n_embd": self.n_embd.value(),
                "n_head": self.n_head.value(),
                "n_layer": self.n_layer.value(),
                "context_length": self.train_context_length.value(),
                "dropout": self.dropout.value(),
                "epochs": self.epochs.value(),
                "batch_size": self.batch_size.value(),
                "learning_rate": self.learning_rate.value(),
                "weight_decay": self.weight_decay.value(),
                "gradient_accumulation": self.gradient_accumulation.value(),
                "warmup_steps": self.warmup_steps.value(),
                "eval_interval": self.eval_interval.value(),
                "save_interval": self.save_interval.value(),
                "max_grad_norm": self.max_grad_norm.value(),
                "seed": self.seed.value(),
                "device": self.device.currentText(),
                "use_amp": self.use_amp.isChecked(),
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

        self.input_dir.setText(str(paths.get("source_vault", "")))
        self.dataset_dir.setText(str(paths.get("dataset_core", "")))
        self.train_data_dir.setText(str(paths.get("training_dataset", "")))
        self.model_dir.setText(str(paths.get("model_output", "")))
        self.export_model_dir.setText(str(paths.get("export_model_core", "")))
        self.export_dir.setText(str(paths.get("export_output", "")))
        self.llama_cpp_dir.setText(str(paths.get("llama_cpp_dir", "")))
        self.gguf_output_path.setText(str(paths.get("gguf_output_path", "")))
        self.gguf_path.setText(str(paths.get("gguf_model", "")))
        self.tokenizer_path.setText(str(paths.get("tokenizer_import", "")))
        self.resume_checkpoint.setText(str(paths.get("resume_checkpoint", "")))

        self.auto_vocab.setChecked(bool(dataset.get("auto_vocab", True)))
        self.manual_vocab_size.setValue(int(dataset.get("manual_vocab_size", self.manual_vocab_size.value())))
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
        self.n_embd.setValue(int(training.get("n_embd", self.n_embd.value())))
        self.n_head.setValue(int(training.get("n_head", self.n_head.value())))
        self.n_layer.setValue(int(training.get("n_layer", self.n_layer.value())))
        self.train_context_length.setValue(int(training.get("context_length", self.train_context_length.value())))
        self.dropout.setValue(float(training.get("dropout", self.dropout.value())))
        self.epochs.setValue(int(training.get("epochs", self.epochs.value())))
        self.batch_size.setValue(int(training.get("batch_size", self.batch_size.value())))
        self.learning_rate.setValue(float(training.get("learning_rate", self.learning_rate.value())))
        self.weight_decay.setValue(float(training.get("weight_decay", self.weight_decay.value())))
        self.gradient_accumulation.setValue(int(training.get("gradient_accumulation", self.gradient_accumulation.value())))
        self.warmup_steps.setValue(int(training.get("warmup_steps", self.warmup_steps.value())))
        self.eval_interval.setValue(int(training.get("eval_interval", self.eval_interval.value())))
        self.save_interval.setValue(int(training.get("save_interval", self.save_interval.value())))
        self.max_grad_norm.setValue(float(training.get("max_grad_norm", self.max_grad_norm.value())))
        self.seed.setValue(int(training.get("seed", self.seed.value())))
        self._set_combo_text(self.device, str(training.get("device", self.device.currentText())))
        self.use_amp.setChecked(bool(training.get("use_amp", self.use_amp.isChecked())))
        self.resume_training.setChecked(bool(training.get("resume", self.resume_training.isChecked())))
        self.resume_safety.setChecked(bool(training.get("require_compatible_resume", True)))
        self.benchmark_prompts.setPlainText(str(training.get("benchmark_prompts", self.benchmark_prompts.toPlainText())))
        self.benchmark_tokens.setValue(int(training.get("benchmark_tokens", self.benchmark_tokens.value())))
        self.benchmark_temperature.setValue(float(training.get("benchmark_temperature", self.benchmark_temperature.value())))
        self.benchmark_kv_cache.setChecked(bool(training.get("benchmark_kv_cache", True)))

        self._set_combo_text(self.quant_mode, str(export.get("quantization", self.quant_mode.currentText())))
        self._set_combo_text(self.gguf_outtype, str(export.get("gguf_outtype", self.gguf_outtype.currentText())))
        self.llama_context.setValue(int(chat.get("context", self.llama_context.value())))
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
        self._update_tokenizer_strategy_controls()
        self._restore_artifact_status(data.get("artifacts", {}))

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
            vocab_size = int(summary.get("tokenizer_vocab_size", 0) or 0)
            self._update_dataset_quality_report(summary)
            self.prepare_button.setText("DataSet Prepared")
            self.dataset_progress.setValue(100)
            if vocab_size:
                self.auto_vocab_label.setText(f"{vocab_size:,}")
            if code_count or prose_count:
                self.dataset_status.setText(f"Dataset: {code_count:,} code, {prose_count:,} prose, {token_count:,} tokens")
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
        self._task_failed(message, self.active_log, self.active_progress_bar)

    def stop_active_task(self) -> None:
        """Request a graceful stop for the active background task."""

        if self.stop_event is None:
            return
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
            if percent is not None:
                progress_bar.setValue(max(0, min(100, int(percent))))
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
        learning_rate = event.get("learning_rate")
        grad_norm = event.get("grad_norm")
        weight_norm = event.get("weight_norm")
        update_ratio = event.get("update_ratio")
        tokens_per_second = event.get("tokens_per_second")
        samples_per_second = event.get("samples_per_second")
        vram_allocated = event.get("vram_allocated_gb")
        vram_reserved = event.get("vram_reserved_gb")
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

        self._drain_progress_queue()
        if self.progress_timer.isActive():
            self.progress_timer.stop()
        self.thread = None
        self.worker = None
        self.stop_event = None
        self.progress_queue = None
        self.active_log = None
        self.active_progress_bar = None
        self.active_stop_button = None

    def _task_failed(self, message: str, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Handle background task failure.

        Args:
            message: Error message.
            log: Log widget to append to.
            progress_bar: Progress bar to reset.
        """

        log.append(f"Error: {message}")
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        if "stopped by user" in message.lower():
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

    def prepare_dataset(self) -> None:
        """Collect dataset options and start dataset preparation."""

        config = DatasetConfig(
            input_dir=Path(self.input_dir.text()),
            output_dir=Path(self.dataset_dir.text()),
            vocab_size=None if self.auto_vocab.isChecked() else self.manual_vocab_size.value(),
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
        )
        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
        self._reset_dataset_quality_report()
        self.dataset_log.append("Preparing dataset...")
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
        self.dataset_log.append(
            f"Prepared {result.document_count} documents, {result.character_count:,} characters, "
            f"{result.token_count:,} tokens, vocab {result.vocab_size:,}."
        )
        self.dataset_log.append(
            f"Cache summary: reused {result.cached_file_count:,} file(s), processed {result.processed_file_count:,} file(s)."
        )
        if getattr(result, "dataset_version_id", ""):
            self.dataset_log.append(f"Dataset version: {result.dataset_version_id}")
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
                "cached_file_count": result.cached_file_count,
                "processed_file_count": result.processed_file_count,
                "skipped_file_count": result.skipped_file_count,
                "failed_file_count": result.failed_file_count,
                "warning": result.warning,
            }
        )
        self.train_data_dir.setText(str(result.output_dir))
        self.project_state.setText("Dataset ready")
        self.dataset_status.setText(f"Dataset: {result.document_count} files, {result.token_count:,} tokens")
        if result.code_sample_count:
            self.dataset_status.setText(
                f"Dataset: {result.code_sample_count:,} code, {result.prose_sample_count:,} prose, {result.token_count:,} tokens"
            )
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

    def start_training(self) -> None:
        """Collect training options and start model training."""

        model_config = ModelConfig(
            vocab_size=1,
            context_length=self.train_context_length.value(),
            embedding_size=self.n_embd.value(),
            head_count=self.n_head.value(),
            layer_count=self.n_layer.value(),
            dropout=self.dropout.value(),
            **self._architecture_style_config(),
        )
        resume_path = Path(self.resume_checkpoint.text()) if self.resume_checkpoint.text().strip() else None
        training_config = TrainingConfig(
            output_dir=Path(self.model_dir.text()),
            epochs=self.epochs.value(),
            batch_size=self.batch_size.value(),
            learning_rate=self.learning_rate.value(),
            weight_decay=self.weight_decay.value(),
            gradient_accumulation=self.gradient_accumulation.value(),
            warmup_steps=self.warmup_steps.value(),
            eval_interval=self.eval_interval.value(),
            save_interval=self.save_interval.value(),
            max_grad_norm=self.max_grad_norm.value(),
            device=self.device.currentText(),
            use_amp=self.use_amp.isChecked(),
            seed=self.seed.value(),
            resume=self.resume_training.isChecked(),
            resume_from_checkpoint=resume_path if self.resume_training.isChecked() else None,
            require_compatible_resume=self.resume_safety.isChecked(),
        )
        self.training_log.clear()
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
        self.training_log.append("Training started...")
        self.project_state.setText("Training")
        self.train_status.setText("Training: running")
        self._run_task(
            train_from_dataset,
            (Path(self.train_data_dir.text()), model_config, training_config),
            self._training_finished,
            self.training_log,
            self.training_progress,
            with_progress=True,
            button=self.train_button,
            stop_button=self.stop_training_button,
            busy_text="Training",
        )

    @Slot(object)
    def _training_finished(self, result: Any) -> None:
        """Update UI after training finishes.

        Args:
            result: Training result.
        """

        self.training_progress.setValue(100)
        self.training_log.append(f"Saved model: {result.checkpoint_path}")
        self.training_log.append(f"Final train loss: {result.final_train_loss:.4f}")
        if result.final_val_loss is not None:
            self.training_log.append(f"Final validation loss: {result.final_val_loss:.4f}")
        self.export_model_dir.setText(str(Path(self.model_dir.text())))
        if getattr(result, "stopped", False):
            self.project_state.setText("Training stopped")
            self.train_status.setText("Training: stopped, checkpoint saved")
            self.training_log.append("Training stopped safely. Resume from this checkpoint or the latest checkpoint.")
        else:
            self.project_state.setText("Training complete")
            self.train_status.setText(f"Training: loss {result.final_train_loss:.4f}")
        self._clear_button_busy("Start Training")

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
            f"Benchmark complete: {result.prompt_count} prompt(s), {result.total_seconds:.2f}s."
        )
        self.benchmark_log.append(f"Benchmark saved: {result.output_path}")
        self.project_state.setText("Benchmark complete")
        self._clear_button_busy("Run Benchmark")

    def toggle_llm_model(self) -> None:
        """Load or unload the GGUF model depending on current state."""

        if self.chat_session is not None:
            self.unload_llm_model()
            return
        self.load_llm_model()

    def load_llm_model(self) -> None:
        """Load a GGUF model for chat testing."""

        model_path = Path(self.gguf_path.text().strip())
        if not model_path:
            QMessageBox.information(self, "Model required", "Choose a GGUF model file first.")
            return
        self.chat_progress.setValue(0)
        self._render_chat_markdown("**Loading GGUF model...**")
        self.chat_stats.setText("Loading model...")
        self.project_state.setText("Loading GGUF")
        self.chat_status.setText("Chat: loading model")
        self._run_task(
            load_llama_chat_session,
            (model_path, self.llama_context.value(), self.llama_threads.value(), self.llama_gpu_layers.value()),
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
        self.project_state.setText("GGUF loaded")
        self.chat_status.setText(f"Chat: {session.runtime_summary}")
        self._clear_button_busy("Unload")
        self._tip(self.load_llm_button, "Unload the currently loaded GGUF model from memory.")

    def unload_llm_model(self) -> None:
        """Unload the active GGUF model and clear chat state."""

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return
        if self.chat_session is not None and hasattr(self.chat_session, "reset"):
            self.chat_session.reset()
        self.chat_session = None
        self._clear_chat_messages()
        self.chat_markdown = ""
        self._add_chat_message("assistant", "Model unloaded.\n\nLoad a GGUF model to start testing.")
        self.chat_progress.setRange(0, 100)
        self.chat_progress.setValue(0)
        self.chat_stats.setText("Idle")
        self.project_state.setText("Ready")
        self.chat_status.setText("Chat: no GGUF loaded")
        self.load_llm_button.setText("Load Model")
        self._tip(self.load_llm_button, "Load the GGUF model into memory once for repeated chat messages.")

    def send_chat_message(self) -> None:
        """Send a prompt to the loaded GGUF model."""

        if self.chat_session is None:
            QMessageBox.information(self, "Load model", "Load a GGUF model before sending a message.")
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
        self._run_task(
            stream_chat_reply,
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

        self.export_log.append("Starting llama.cpp GGUF conversion...")
        self.export_progress.setValue(0)
        self._run_task(
            export_gguf_with_llama_cpp,
            (
                Path(self.export_model_dir.text()),
                Path(self.llama_cpp_dir.text()),
                Path(self.gguf_output_path.text()),
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

    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
        except Exception:
            pass
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

