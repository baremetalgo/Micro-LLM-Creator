from __future__ import annotations

import ctypes
from queue import Empty, Queue
import sys
from pathlib import Path
from typing import Any

import torch
from PySide6.QtCore import QObject, QPoint, Qt, QThread, QTimer, Signal
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
    QSizePolicy,
    QStackedWidget,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from llm_trainer.config import DatasetConfig, ModelConfig, TrainingConfig
from llm_trainer.export import export_project_bundle, quantize_checkpoint
from llm_trainer.services import build_dataset, train_from_dataset


WINDOWS_APP_ID = "MicroLLMCreator.Lightning"


class TaskWorker(QObject):
    """Background worker used for long-running UI tasks."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Any, *args: Any, progress_queue: Queue | None = None, with_progress: bool = False) -> None:
        """Create a worker.

        Args:
            fn: Callable to execute in the worker thread.
            *args: Positional arguments passed to ``fn``.
            progress_queue: Optional queue for progress events.
            with_progress: Whether to pass a progress callback to ``fn``.
        """

        super().__init__()
        self.fn = fn
        self.args = args
        self.progress_queue = progress_queue
        self.with_progress = with_progress

    def run(self) -> None:
        """Execute the worker function and emit completion or failure."""

        try:
            if self.with_progress:
                self.finished.emit(self.fn(*self.args, progress=self._queue_progress))
            else:
                self.finished.emit(self.fn(*self.args))
        except Exception as exc:
            self.failed.emit(str(exc))

    def _queue_progress(self, event: Any) -> None:
        if self.progress_queue is not None:
            self.progress_queue.put(event)


class MainWindow(QMainWindow):
    """Main PySide6 window for Micro LLM Creator."""

    def __init__(self) -> None:
        """Create the main application window."""

        super().__init__()
        if QApplication.instance():
            QApplication.instance().setFont(QFont("Segoe UI", 10))
        self.setWindowTitle("Micro LLM Creator")
        self.setWindowIcon(self._lightning_icon())
        self._windows_icon_handles: list[int] = []
        self.resize(1240, 820)
        self.thread: QThread | None = None
        self.worker: TaskWorker | None = None
        self.progress_queue: Queue | None = None
        self.active_log: QTextEdit | None = None
        self.active_progress_bar: QProgressBar | None = None
        self.active_button: QPushButton | None = None
        self.active_button_text = ""
        self.active_button_restore_text = ""
        self.spinner_index = 0
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self._tick_spinner)
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._drain_progress_queue)

        self._apply_style()

        shell = self._build_shell()
        self.setCentralWidget(shell)

    def _apply_style(self) -> None:
        """Apply the sci-fi dashboard stylesheet."""

        self.setStyleSheet(
            """
            * { font-family: "Segoe UI", Arial, sans-serif; }
            QMainWindow, QWidget#AppShell {
                background: #050914;
                color: #d6e2dd;
            }
            QWidget#TopBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #07162b, stop:1 #101633);
                border-bottom: 1px solid #1d7ea3;
            }
            QWidget#SideRail {
                background: #050d1d;
                border-right: 1px solid #1d7ea3;
            }
            QWidget#RightPanel {
                background: #081426;
                border-left: 1px solid #1d7ea3;
            }
            QLabel#Logo {
                color: #73f7ff;
                font-size: 24px;
                font-weight: 900;
            }
            QLabel#AppTitle {
                color: #f5fff9;
                font-size: 28px;
                font-weight: 700;
            }
            QLabel#PageTitle {
                color: #f5fff9;
                font-size: 25px;
                font-weight: 700;
            }
            QLabel#SectionLabel {
                color: #73f7ff;
                font-size: 12px;
                font-weight: 800;
                text-transform: uppercase;
            }
            QLabel#SideTitle {
                color: #d8e7ff;
                font-size: 18px;
                font-weight: 800;
            }
            QWidget#Panel {
                background: #071225;
                color: #d6e2dd;
            }
            QWidget#Card {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0b1b34, stop:1 #101b3a);
                border: 1px solid #235e80;
                border-radius: 8px;
            }
            QPushButton#NavButton {
                background: #071225; color: #a8c7e8; border: 1px solid #235e80;
                border-radius: 8px; min-width: 42px; min-height: 42px; padding: 0;
            }
            QPushButton#NavButton:checked, QPushButton#NavButton:hover {
                background: #73f7ff; color: #071225; border-color: #b8fbff;
            }
            QLabel { color: #c8d8f0; font-size: 13px; }
            QLabel#Metric { color: #a4ff7a; font-size: 14px; font-weight: 700; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background: #050a12; color: #eff9ff; border: 1px solid #1d7ea3;
                border-radius: 7px; padding: 8px 10px; min-height: 22px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border-color: #73f7ff;
                background: #07101f;
            }
            QCheckBox { color: #c8d8f0; spacing: 8px; }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #73f7ff, stop:1 #49a7ff);
                color: #03101d; border: 1px solid #b8fbff;
                border-radius: 7px; padding: 9px 14px; font-weight: 800;
            }
            QPushButton:hover { background: #b8fbff; }
            QTextEdit {
                background: #03070d; color: #a8f1ff; border: 1px solid #1d7ea3;
                border-radius: 8px; padding: 10px; font-family: Consolas, monospace; font-size: 12px;
            }
            QProgressBar {
                background: #03070d; border: 0; border-radius: 2px;
                min-height: 4px; max-height: 4px;
            }
            QProgressBar::chunk { background: #73f7ff; border-radius: 2px; }
            """
        )

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
        top_layout.setContentsMargins(16, 10, 16, 10)
        logo = QLabel("ML")
        logo.setObjectName("Logo")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search project...")
        self.search_box.setMaximumWidth(320)
        self._tip(self.search_box, "Search across project paths and future saved presets. This does not affect training.")
        self.project_state = QLabel("Ready")
        self.project_state.setObjectName("Metric")
        top_layout.addWidget(logo)
        top_layout.addSpacing(18)
        top_layout.addWidget(self.search_box)
        top_layout.addStretch(1)
        top_layout.addWidget(self.project_state)
        root.addWidget(top)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        rail = QWidget()
        rail.setObjectName("SideRail")
        rail.setFixedWidth(72)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(12, 22, 12, 22)
        rail_layout.setSpacing(14)
        self.dataset_nav = self._nav_button("IN")
        self.training_nav = self._nav_button("AI")
        self.export_nav = self._nav_button("X")
        self._tip(self.dataset_nav, "Open dataset preparation: load text/PDF files and build tokenizer data.")
        self._tip(self.training_nav, "Open model training: configure architecture and optimization settings.")
        self._tip(self.export_nav, "Open export tools: bundle or quantize the trained model artifacts.")
        self.dataset_nav.setChecked(True)
        self.dataset_nav.clicked.connect(lambda: self._switch_page(0))
        self.training_nav.clicked.connect(lambda: self._switch_page(1))
        self.export_nav.clicked.connect(lambda: self._switch_page(2))
        rail_layout.addWidget(self.dataset_nav)
        rail_layout.addWidget(self.training_nav)
        rail_layout.addWidget(self.export_nav)
        rail_layout.addStretch(1)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_dataset_tab())
        self.pages.addWidget(self._build_training_tab())
        self.pages.addWidget(self._build_export_tab())

        body.addWidget(rail)
        body.addWidget(self.pages, 1)
        body.addWidget(self._build_status_panel())
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
        buttons = [self.dataset_nav, self.training_nav, self.export_nav]
        for button_index, button in enumerate(buttons):
            button.setChecked(button_index == index)

    def _build_status_panel(self) -> QWidget:
        """Build the right-side project status panel.

        Returns:
            Status panel widget.
        """

        panel = QWidget()
        panel.setObjectName("RightPanel")
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(26, 26, 26, 26)
        layout.setSpacing(18)
        title = QLabel("Project Status")
        title.setObjectName("SideTitle")
        self.dataset_status = QLabel("Dataset: not prepared")
        self.train_status = QLabel("Training: idle")
        self.export_status = QLabel("Export: waiting")
        for label in (self.dataset_status, self.train_status, self.export_status):
            label.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(self.dataset_status)
        layout.addWidget(self.train_status)
        layout.addWidget(self.export_status)
        layout.addStretch(1)
        return panel

    def _build_dataset_tab(self) -> QWidget:
        """Build the dataset preparation page.

        Returns:
            Dataset page widget.
        """

        page = self._panel()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 14)
        layout.setSpacing(12)
        title = self._page_title("Data Ingestion Matrix")
        layout.addWidget(title)

        module_grid = QGridLayout()
        module_grid.setHorizontalSpacing(14)
        module_grid.setVerticalSpacing(14)

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
        self.auto_vocab.toggled.connect(lambda checked: self.manual_vocab_size.setEnabled(not checked))
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

        source_form.addRow("Source vault", self._path_row(self.input_dir, directory=True))
        source_form.addRow("Dataset core", self._path_row(self.dataset_dir, directory=True))
        source_form.addRow("Parallel lanes", self.max_workers)
        source_form.addRow("", self.lowercase)
        source_form.addRow("", self.code_training_mode)
        source_form.addRow("", self.include_source_code)

        tokenizer_form.addRow("Auto vocabulary", self.auto_vocab)
        tokenizer_form.addRow("Manual vocabulary", self.manual_vocab_size)
        tokenizer_form.addRow("Selected vocab", self.auto_vocab_label)
        tokenizer_form.addRow("Min frequency", self.min_frequency)
        tokenizer_form.addRow("Context window", self.context_length)
        tokenizer_form.addRow("Validation split", self.validation_split)
        tokenizer_form.addRow("", self.include_prose)
        tokenizer_form.addRow("", self.extract_code_blocks)
        tokenizer_form.addRow("", self.preserve_indentation)
        tokenizer_form.addRow("", self.instruction_samples)
        module_grid.addWidget(self._card("SOURCE ARRAY", source_form), 0, 0)
        module_grid.addWidget(self._card("TOKENIZER CORE", tokenizer_form), 0, 1)
        module_grid.setColumnStretch(0, 1)
        module_grid.setColumnStretch(1, 1)
        layout.addLayout(module_grid)

        self.prepare_button = QPushButton("Prepare Dataset")
        self._tip(self.prepare_button, "Read source files, clean text, train tokenizer, split tokens, and save the dataset project.")
        self.prepare_button.clicked.connect(self.prepare_dataset)
        self.prepare_button.setMaximumWidth(320)

        self.dataset_log = QTextEdit()
        self.dataset_log.setReadOnly(True)
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.dataset_log)
        layout.addWidget(self._card("INGEST TELEMETRY", log_layout), 1)
        layout.addWidget(self.prepare_button)

        self.dataset_progress = self._thin_progress()
        layout.addWidget(self.dataset_progress)
        return page

    def _build_training_tab(self) -> QWidget:
        """Build the training configuration page.

        Returns:
            Training page widget.
        """

        page = self._panel()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 14)
        layout.setSpacing(12)
        layout.addWidget(self._page_title("Neural Forge"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

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
        left.addRow("Dataset project", self._path_row(self.train_data_dir, directory=True))
        left.addRow("Model output", self._path_row(self.model_dir, directory=True))
        left.addRow("Preset", self.preset)
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
        self.device.addItem("cuda" if torch.cuda.is_available() else "cpu")
        self.device.addItem("cpu")
        self.device.setMaximumWidth(260)
        self._tip(self.device, "Hardware target. CUDA uses NVIDIA GPU when available; CPU is slower but broadly compatible.")
        self.use_amp = QCheckBox("Use mixed precision on CUDA")
        self.use_amp.setChecked(torch.cuda.is_available())
        self._tip(self.use_amp, "Use mixed precision on CUDA. Usually faster and lighter on GPU memory.")
        self.resume_training = QCheckBox("Resume from latest checkpoint")
        self.resume_training.setChecked(True)
        self._tip(self.resume_training, "Continue from the latest checkpoint if training was interrupted.")
        self.resume_checkpoint = QLineEdit()
        self._tip(self.resume_checkpoint, "Optional specific checkpoint file to resume from instead of the latest checkpoint.")
        right.addRow("Epochs", self.epochs)
        right.addRow("Batch size", self.batch_size)
        right.addRow("Learning rate", self.learning_rate)
        right.addRow("Weight decay", self.weight_decay)
        right.addRow("Gradient accumulation", self.gradient_accumulation)
        right.addRow("Warmup steps", self.warmup_steps)
        right.addRow("Eval interval", self.eval_interval)
        right.addRow("Save interval", self.save_interval)
        right.addRow("Max grad norm", self.max_grad_norm)
        right.addRow("Seed", self.seed)
        runtime = QFormLayout()
        self._configure_form(runtime)
        runtime.addRow("Device", self.device)
        runtime.addRow("", self.use_amp)
        runtime.addRow("", self.resume_training)
        runtime.addRow("Resume checkpoint", self._path_row(self.resume_checkpoint, directory=False))

        grid.addWidget(self._card("MODEL ARCHITECTURE", left), 0, 0)
        grid.addWidget(self._card("OPTIMIZATION ENGINE", right), 0, 1)
        grid.addWidget(self._card("RUNTIME CONTROL", runtime), 0, 2)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 0)
        controls = QWidget()
        controls.setMaximumWidth(1380)
        controls.setLayout(grid)
        layout.addWidget(controls, 0, Qt.AlignLeft | Qt.AlignTop)

        self.train_button = QPushButton("Start Training")
        self._tip(self.train_button, "Start or resume training using the selected model and optimizer settings.")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setMaximumWidth(320)

        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        telemetry_layout = QVBoxLayout()
        telemetry_layout.addWidget(self.training_log)
        layout.addWidget(self._card("TRAINING TELEMETRY", telemetry_layout), 1)
        layout.addWidget(self.train_button)

        self.training_progress = self._thin_progress()
        layout.addWidget(self.training_progress)
        return page

    def _build_export_tab(self) -> QWidget:
        """Build the export page.

        Returns:
            Export page widget.
        """

        page = self._panel()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 14)
        layout.setSpacing(12)
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
        form.addRow("Model core", self._path_row(self.export_model_dir, directory=True))
        form.addRow("Output bay", self._path_row(self.export_dir, directory=True))
        form.addRow("Quantization", self.quant_mode)
        layout.addWidget(self._card("ARTIFACT CONFIGURATION", form))

        row = QHBoxLayout()
        row.setSpacing(10)
        bundle_button = QPushButton("Create Bundle")
        self._tip(bundle_button, "Copy final model, tokenizer, and summary into a portable export folder.")
        bundle_button.clicked.connect(self.create_bundle)
        quant_button = QPushButton("Quantize Model")
        self._tip(quant_button, "Create a smaller FP16 checkpoint for inference or later conversion workflows.")
        quant_button.clicked.connect(self.quantize_model)
        bundle_button.setMaximumWidth(220)
        quant_button.setMaximumWidth(220)
        row.addWidget(bundle_button)
        row.addWidget(quant_button)
        row.addStretch(1)

        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setPlainText(
            "Export options:\n"
            "- Bundle copies final_model.pt, tokenizer.json, and training_summary.json.\n"
            "- FP16 checkpoint quantization works now.\n"
            "- GGUF quantization options are shown as targets; the real converter path is the next milestone.\n"
        )
        export_log_layout = QVBoxLayout()
        export_log_layout.addWidget(self.export_log)
        layout.addWidget(self._card("EXPORT TELEMETRY", export_log_layout), 1)
        layout.addLayout(row)

        self.export_progress = self._thin_progress()
        layout.addWidget(self.export_progress)
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

    def _card(self, title: str, content_layout: QVBoxLayout | QFormLayout | QGridLayout | QHBoxLayout) -> QWidget:
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
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
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
        spin.setMaximumWidth(260)
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
        spin.setMaximumWidth(260)
        return spin

    def _path_row(self, field: QLineEdit, directory: bool = True) -> QWidget:
        """Create a path field with a browse button.

        Args:
            field: Path input widget.
            directory: Whether the browse dialog selects folders.

        Returns:
            Row widget containing the path input and button.
        """

        row = QWidget()
        row.setMaximumWidth(680)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        browse = QPushButton("Browse")
        browse.setFixedWidth(88)
        self._tip(browse, "Open a file/folder picker for this path.")
        field.setMinimumWidth(260)
        field.setMaximumWidth(560)
        browse.clicked.connect(lambda: self._browse(field, directory))
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
        form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(9)

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
            painter.setBrush(QBrush(QColor("#071225")))
            painter.setPen(QPen(QColor("#73f7ff"), 3))
            painter.drawRoundedRect(4, 4, 56, 56, 12, 12)
            bolt = QPolygon([
                QPoint(36, 8),
                QPoint(17, 35),
                QPoint(31, 35),
                QPoint(25, 56),
                QPoint(48, 25),
                QPoint(33, 25),
            ])
            painter.setPen(QPen(QColor("#f7ff7a"), 2))
            painter.setBrush(QBrush(QColor("#73f7ff")))
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
    def _ensure_windows_icon_file() -> Path | None:
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

    def _browse(self, field: QLineEdit, directory: bool) -> None:
        """Open a file or folder picker for a path field.

        Args:
            field: Path input to update.
            directory: Whether to select a folder instead of a file.
        """

        if directory:
            value = QFileDialog.getExistingDirectory(self, "Choose folder", field.text() or str(Path.cwd()))
        else:
            value, _ = QFileDialog.getOpenFileName(self, "Choose checkpoint", field.text() or str(Path.cwd()), "Checkpoints (*.pt)")
        if value:
            field.setText(value)

    def _run_task(
        self,
        fn,
        args,
        on_finished,
        log: QTextEdit,
        progress_bar: QProgressBar,
        with_progress: bool = False,
        button: QPushButton | None = None,
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
            busy_text: Button text shown while running.
        """

        if self.thread is not None:
            QMessageBox.information(self, "Task running", "Please wait for the current task to finish.")
            return

        if button:
            self._set_button_busy(button, busy_text)

        self.progress_queue = Queue()
        self.active_log = log
        self.active_progress_bar = progress_bar
        self.thread = QThread(self)
        self.worker = TaskWorker(fn, *args, progress_queue=self.progress_queue, with_progress=with_progress)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(on_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(lambda message: self._task_failed(message, log, progress_bar))
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.progress_timer.start(100)
        self.thread.start()

    def _handle_progress(self, event: object, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Apply one progress event to UI widgets.

        Args:
            event: Progress dictionary or message.
            log: Log widget to append messages to.
            progress_bar: Progress bar to update.
        """

        if isinstance(event, dict):
            message = event.get("message")
            percent = event.get("percent")
            if message:
                log.append(str(message))
            if percent is not None:
                progress_bar.setValue(max(0, min(100, int(percent))))
        else:
            log.append(str(event))

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
        self.progress_timer.stop()
        self.thread = None
        self.worker = None
        self.progress_queue = None
        self.active_log = None
        self.active_progress_bar = None

    def _task_failed(self, message: str, log: QTextEdit, progress_bar: QProgressBar) -> None:
        """Handle background task failure.

        Args:
            message: Error message.
            log: Log widget to append to.
            progress_bar: Progress bar to reset.
        """

        log.append(f"Error: {message}")
        progress_bar.setValue(0)
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

    def _clear_button_busy(self, final_text: str | None = None) -> None:
        """Restore the active busy button.

        Args:
            final_text: Optional final button text.
        """

        self.spinner_timer.stop()
        if self.active_button:
            self.active_button.setEnabled(True)
            self.active_button.setText(final_text or self.active_button_restore_text)
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
        )
        self.dataset_log.clear()
        self.dataset_progress.setValue(0)
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
            busy_text="Preparing Dataset",
        )

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
        if result.warning:
            self.dataset_log.append(f"Recommendation: {result.warning}")
        self.train_data_dir.setText(str(result.output_dir))
        self.project_state.setText("Dataset ready")
        self.dataset_status.setText(f"Dataset: {result.document_count} files, {result.token_count:,} tokens")
        if result.code_sample_count:
            self.dataset_status.setText(
                f"Dataset: {result.code_sample_count:,} code, {result.prose_sample_count:,} prose, {result.token_count:,} tokens"
            )
        self._clear_button_busy("DataSet Prepared")

    def start_training(self) -> None:
        """Collect training options and start model training."""

        model_config = ModelConfig(
            vocab_size=1,
            context_length=self.train_context_length.value(),
            embedding_size=self.n_embd.value(),
            head_count=self.n_head.value(),
            layer_count=self.n_layer.value(),
            dropout=self.dropout.value(),
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
        )
        self.training_log.clear()
        self.training_progress.setValue(0)
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
            busy_text="Training",
        )

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
        self.project_state.setText("Training complete")
        self.train_status.setText(f"Training: loss {result.final_train_loss:.4f}")
        self._clear_button_busy("Start Training")

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
    app.setFont(QFont("Segoe UI", 10))
    app.setWindowIcon(MainWindow._static_lightning_icon())
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, window.apply_windows_taskbar_icon)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
