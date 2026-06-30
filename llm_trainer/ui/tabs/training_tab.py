from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def build_training_tab(window) -> QWidget:
    """Build the training configuration page.

    Returns:
        Training page widget.
    """

    page = window._panel()
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
    layout.addWidget(window._page_title("Neural Forge"))
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
    window._configure_form(left)
    window.train_data_dir = QLineEdit(str(Path.cwd() / "runs" / "dataset"))
    window._tip(window.train_data_dir, "Prepared dataset folder containing tokenizer.json and train/validation token files.")
    window.model_dir = QLineEdit(str(Path.cwd() / "runs" / "model"))
    window._tip(window.model_dir, "Folder where checkpoints, final model, tokenizer copy, and training summary are saved.")
    window.preset = QComboBox()
    window.preset.addItems(["Tiny", "Small", "Custom"])
    window.preset.setMaximumWidth(260)
    window._tip(window.preset, "Architecture preset. Tiny is faster; Small has more capacity but needs more memory and training data.")
    window.preset.currentTextChanged.connect(window._apply_preset)
    window.architecture_style = QComboBox()
    window.architecture_style.addItems(["Classic GPT", "Llama-like"])
    window.architecture_style.setMaximumWidth(260)
    window._tip(
        window.architecture_style,
        "Classic uses learned positions, LayerNorm, and GELU. Llama-like uses RoPE, RMSNorm, and SwiGLU.",
    )
    window.n_embd = window._spin(32, 4096, 128)
    window._tip(window.n_embd, "Embedding size, also called n_embd. Larger values increase model capacity and memory usage.")
    window.n_head = window._spin(1, 64, 4)
    window._tip(window.n_head, "Attention head count. More heads can model varied relationships, but n_embd must divide evenly by n_head.")
    window.attention_type = QComboBox()
    window.attention_type.addItems(["Multi-head", "Grouped-query", "Multi-query"])
    window.attention_type.setMaximumWidth(260)
    window._tip(
        window.attention_type,
        "Attention layout. Grouped-query and multi-query share key/value heads to reduce memory and speed up generation.",
    )
    window.kv_head_count = window._spin(1, 64, 2)
    window._tip(window.kv_head_count, "Key/value heads for grouped-query attention. Must divide n_head. Ignored by multi-head and multi-query.")
    window.attention_backend = QComboBox()
    window.attention_backend.addItems(["SDPA / Flash when available", "Manual"])
    window.attention_backend.setMaximumWidth(260)
    window._tip(
        window.attention_backend,
        "Attention kernel. SDPA lets PyTorch use Flash Attention on supported GPUs and falls back safely otherwise.",
    )
    window.attention_window = window._spin(0, 4096, 0)
    window._tip(window.attention_window, "Sliding attention window. 0 uses full context; higher values restrict attention to recent tokens.")
    window.n_layer = window._spin(1, 64, 4)
    window._tip(window.n_layer, "Transformer layer count. More layers improve capacity and reasoning patterns but slow training.")
    window.train_context_length = window._spin(16, 4096, 128)
    window._tip(window.train_context_length, "Training context length in tokens. Must fit your GPU/CPU memory.")
    window.dropout = window._double_spin(0.0, 0.9, 0.1, 0.01, 3)
    window._tip(window.dropout, "Dropout regularization. Higher values reduce overfitting but can slow learning.")
    left.addRow("Dataset", window._path_row(window.train_data_dir, directory=True))
    left.addRow("Model", window._path_row(window.model_dir, directory=True))
    left.addRow("Preset", window.preset)
    left.addRow("Block style", window.architecture_style)
    left.addRow("n_embd", window.n_embd)
    left.addRow("n_head", window.n_head)
    left.addRow("Attention", window.attention_type)
    left.addRow("KV heads", window.kv_head_count)
    left.addRow("Backend", window.attention_backend)
    left.addRow("Window", window.attention_window)
    left.addRow("n_layer", window.n_layer)
    left.addRow("Context length", window.train_context_length)
    left.addRow("Dropout", window.dropout)

    right = QFormLayout()
    window._configure_form(right)
    window.epochs = window._spin(1, 10000, 5)
    window._tip(window.epochs, "Number of full passes over the training tokens. More epochs can improve learning or overfit small data.")
    window.batch_size = window._spin(1, 512, 16)
    window._tip(window.batch_size, "Sequences processed per step. Larger batches are smoother but require more memory.")
    window.learning_rate = window._double_spin(0.000001, 1.0, 0.0003, 0.0001, 6)
    window._tip(window.learning_rate, "Optimizer step size. Too high can destabilize training; too low trains slowly.")
    window.weight_decay = window._double_spin(0.0, 1.0, 0.1, 0.01, 4)
    window._tip(window.weight_decay, "Weight decay regularization. Helps control overfitting by discouraging large weights.")
    window.training_profile = QComboBox()
    window.training_profile.addItems(["Stable LLM", "Low-memory", "Code fine-tune", "Experimental Lion"])
    window.training_profile.setMaximumWidth(260)
    window._tip(window.training_profile, "Applies a practical optimizer, scheduler, precision, and regularization profile.")
    window.apply_training_profile_button = QPushButton("Apply Profile")
    window.apply_training_profile_button.setMaximumWidth(160)
    window.apply_training_profile_button.clicked.connect(window.apply_training_profile)
    window._tip(window.apply_training_profile_button, "Apply the selected training profile to the controls below.")
    window.optimizer_name = QComboBox()
    window.optimizer_name.addItems(["AdamW", "Adam", "Lion", "Adafactor"])
    window.optimizer_name.setMaximumWidth(260)
    window._tip(
        window.optimizer_name,
        "Optimizer algorithm. AdamW is the safest default; Lion can be efficient; Adafactor can reduce optimizer memory when supported.",
    )
    window.scheduler_name = QComboBox()
    window.scheduler_name.addItems(["Warmup linear", "Cosine decay", "Polynomial decay", "One-cycle", "Constant"])
    window.scheduler_name.setMaximumWidth(260)
    window._tip(
        window.scheduler_name,
        "Learning-rate schedule. Cosine and one-cycle are common for stable LLM training; constant is mostly for experiments.",
    )
    window.min_lr_ratio = window._double_spin(0.0, 1.0, 0.1, 0.01, 3)
    window._tip(window.min_lr_ratio, "Lowest learning-rate multiplier after decay. 0.1 means decay down to 10% of the base LR.")
    window.polynomial_power = window._double_spin(0.1, 10.0, 1.0, 0.1, 2)
    window._tip(window.polynomial_power, "Shape of polynomial decay. Higher values decay the learning rate more aggressively near the end.")
    window.gradient_accumulation = window._spin(1, 256, 1)
    window._tip(window.gradient_accumulation, "Accumulate gradients across batches before updating. Simulates larger batches with less memory.")
    window.warmup_steps = window._spin(0, 1_000_000, 100)
    window._tip(window.warmup_steps, "Steps used to ramp up learning rate. Warmup helps avoid unstable early training.")
    window.eval_interval = window._spin(0, 1_000_000, 100)
    window._tip(window.eval_interval, "Training steps between validation checks. Set 0 to skip interval validation.")
    window.max_eval_batches = window._spin(0, 1_000_000, 50)
    window._tip(window.max_eval_batches, "Maximum validation batches per validation check. Set 0 to evaluate the full validation split.")
    window.save_interval = window._spin(1, 1_000_000, 500)
    window._tip(window.save_interval, "Training steps between checkpoints. Lower values improve crash recovery but use more disk.")
    window.data_loader_workers = window._spin(0, 64, 0)
    window._tip(
        window.data_loader_workers,
        "CPU worker processes used to prepare training batches while the model trains. This is the safe GPU+CPU hybrid mode.",
    )
    window.max_grad_norm = window._double_spin(0.1, 100.0, 1.0, 0.1, 3)
    window._tip(window.max_grad_norm, "Gradient clipping limit. Helps prevent exploding gradients during training.")
    window.seed = window._spin(1, 2_147_483_647, 1337)
    window._tip(window.seed, "Random seed for reproducible initialization and sampling order.")
    window.device = QComboBox()
    window.device.setMaximumWidth(260)
    window._tip(window.device, "Hardware target. CUDA uses NVIDIA GPU when available; CPU is slower but broadly compatible.")
    window.device_info = QLabel()
    window.device_info.setObjectName("Metric")
    window.device_info.setWordWrap(True)
    window.device_info.setMaximumWidth(260)
    window._configure_device_options()
    window.use_amp = QCheckBox("Mixed precision")
    window.use_amp.setChecked(window.use_amp_default)
    window._tip(window.use_amp, "Use mixed precision on CUDA. Usually faster and lighter on GPU memory.")
    window.precision = QComboBox()
    window.precision.addItems(["FP16", "BF16", "FP32"])
    window.precision.setMaximumWidth(260)
    window._tip(
        window.precision,
        "Numeric precision. FP16 is fast on many NVIDIA GPUs; BF16 is more stable on supported GPUs; FP32 is safest but uses more memory.",
    )
    window.resume_training = QCheckBox("Resume latest")
    window.resume_training.setChecked(True)
    window._tip(window.resume_training, "Continue from the latest checkpoint if training was interrupted.")
    window.resume_safety = QCheckBox("Safe resume")
    window.resume_safety.setChecked(True)
    window._tip(
        window.resume_safety,
        "Before resuming, verify that the dataset tokenizer and model architecture match the checkpoint.",
    )
    window.resume_checkpoint = QLineEdit()
    window._tip(window.resume_checkpoint, "Optional specific checkpoint file to resume from instead of the latest checkpoint.")
    window.resume_check_button = QPushButton("Check Resume")
    window.resume_check_button.setMaximumWidth(180)
    window.resume_check_button.clicked.connect(window.preview_resume_compatibility)
    window._tip(window.resume_check_button, "Inspect checkpoint compatibility before starting training.")
    right.addRow("Epochs", window.epochs)
    right.addRow("Batch", window.batch_size)
    right.addRow("Profile", window.training_profile)
    right.addRow("", window.apply_training_profile_button)
    right.addRow("LR", window.learning_rate)
    right.addRow("Decay", window.weight_decay)
    right.addRow("Optimizer", window.optimizer_name)
    right.addRow("Schedule", window.scheduler_name)
    right.addRow("Min LR", window.min_lr_ratio)
    right.addRow("Poly power", window.polynomial_power)
    right.addRow("Grad accum", window.gradient_accumulation)
    right.addRow("Warmup", window.warmup_steps)
    right.addRow("Eval every", window.eval_interval)
    right.addRow("Eval batches", window.max_eval_batches)
    right.addRow("Save every", window.save_interval)
    right.addRow("CPU workers", window.data_loader_workers)
    right.addRow("Max grad", window.max_grad_norm)
    right.addRow("Seed", window.seed)
    runtime = QFormLayout()
    window._configure_form(runtime)
    window.training_launch_target = QComboBox()
    window.training_launch_target.addItems(["Local machine", "Remote workers"])
    window.training_launch_target.setMaximumWidth(260)
    window._tip(
        window.training_launch_target,
        "Local runs training on this computer. Remote publishes a job for available workers to claim from Job Manager.",
    )
    runtime.addRow("Launch", window.training_launch_target)
    runtime.addRow("Device", window.device)
    runtime.addRow("Hardware", window.device_info)
    runtime.addRow("", window.use_amp)
    runtime.addRow("Precision", window.precision)
    runtime.addRow("", window.resume_training)
    runtime.addRow("", window.resume_safety)
    runtime.addRow("Checkpoint", window._path_row(window.resume_checkpoint, directory=False))
    runtime.addRow("", window.resume_check_button)

    window.training_cards = [
        window._card("MODEL ARCHITECTURE", left),
        window._card("OPTIMIZATION ENGINE", right),
    ]
    for card in window.training_cards:
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    for index, card in enumerate(window.training_cards):
        grid.addWidget(card, 0, index)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    window.training_controls_grid = grid
    window.training_controls_columns = 2
    controls = QWidget()
    controls.setLayout(grid)
    controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    window.training_controls = controls
    left_zone.addWidget(controls, 0)
    right_zone.addWidget(window._card("RUNTIME CONTROL", runtime), 0)

    window.resume_training_preview = QTextEdit()
    window.resume_training_preview.setReadOnly(True)
    window.resume_training_preview.setMinimumHeight(110)
    window.resume_training_preview.setMaximumHeight(180)
    window.resume_training_preview.setText("No compatibility check has been run.")
    window.resume_preview = window.resume_training_preview
    window._tip(window.resume_training_preview, "Compatibility report for the selected or latest checkpoint.")
    resume_preview_layout = QVBoxLayout()
    resume_preview_layout.addWidget(window.resume_training_preview)
    right_zone.addWidget(window._card("RESUME COMPATIBILITY", resume_preview_layout), 0)

    window.train_button = QPushButton("Start Training")
    window._tip(window.train_button, "Start or resume training using the selected model and optimizer settings.")
    window.train_button.clicked.connect(window.start_training)
    window.train_button.setMaximumWidth(320)
    window.stop_training_button = QPushButton("Stop")
    window.stop_training_button.setEnabled(False)
    window.stop_training_button.setMaximumWidth(120)
    window.stop_training_button.clicked.connect(window.stop_active_task)
    window._tip(window.stop_training_button, "Request a graceful stop and save a resumable checkpoint.")

    action_row = QHBoxLayout()
    action_row.addWidget(window.train_button)
    action_row.addWidget(window.stop_training_button)
    action_row.addStretch(1)
    left_zone.addLayout(action_row)

    metrics_grid = QGridLayout()
    metrics_grid.setHorizontalSpacing(8)
    metrics_grid.setVerticalSpacing(8)
    window.training_epoch_metric = window._metric_chip("Epoch: -", "Current epoch and total epochs.")
    window.training_step_metric = window._metric_chip("Step: -", "Current optimizer step and total planned steps.")
    window.training_loss_metric = window._metric_chip("Train loss: -", "Latest training loss. Lower is usually better.")
    window.training_val_metric = window._metric_chip("Val loss: -", "Latest validation loss when validation is enabled.")
    window.training_lr_metric = window._metric_chip("LR: -", "Current learning rate from the scheduler.")
    window.training_speed_metric = window._metric_chip("Speed: -", "Current training throughput.")
    window.training_grad_metric = window._metric_chip("Grad: -", "Current gradient norm.")
    window.training_vram_metric = window._metric_chip("VRAM: -", "Current CUDA memory usage when training on GPU.")
    window.training_eta_metric = window._metric_chip("ETA: -", "Estimated time remaining based on recent optimizer steps.")
    window.model_size_metric = window._metric_chip("Model: -", "Estimated model parameter count and checkpoint size.")
    window.vram_estimate_metric = window._metric_chip("VRAM est: -", "Rough training VRAM estimate for selected architecture and batch.")
    window.history_metric = window._metric_chip("Runs: -", "Training run history count in the current model folder.")
    for index, metric in enumerate((
        window.training_eta_metric,
        window.training_epoch_metric,
        window.training_step_metric,
        window.training_loss_metric,
        window.training_val_metric,
        window.training_lr_metric,
        window.training_speed_metric,
        window.training_grad_metric,
        window.training_vram_metric,
    )):
        metrics_grid.addWidget(metric, index, 0)
    metrics_layout = QVBoxLayout()
    metrics_layout.setSpacing(8)
    metrics_layout.addLayout(metrics_grid)
    estimate_grid = QGridLayout()
    estimate_grid.setHorizontalSpacing(8)
    estimate_grid.setVerticalSpacing(8)
    estimate_grid.addWidget(window.model_size_metric, 0, 0)
    estimate_grid.addWidget(window.vram_estimate_metric, 1, 0)
    estimate_grid.addWidget(window.history_metric, 2, 0)
    window.refresh_estimate_button = QPushButton("Refresh Estimate")
    window.refresh_estimate_button.clicked.connect(window.refresh_model_estimate)
    window.refresh_estimate_button.setMaximumWidth(180)
    window._tip(window.refresh_estimate_button, "Refresh model size, rough VRAM, and training history estimates without starting training.")
    estimate_layout = QVBoxLayout()
    estimate_layout.setSpacing(8)
    estimate_layout.addLayout(estimate_grid)
    estimate_layout.addWidget(window.refresh_estimate_button)
    estimate_layout.addStretch(1)
    right_zone.addWidget(window._card("MODEL ESTIMATE", estimate_layout), 0)
    right_zone.addWidget(window._card("TRAINING METRICS", metrics_layout), 0)

    window.training_log = QTextEdit()
    window.training_log.setReadOnly(True)
    window.training_log.setMinimumHeight(320)
    window.training_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    telemetry_layout = QVBoxLayout()
    telemetry_layout.addWidget(window.training_log, 1)
    left_zone.addWidget(window._card("TRAINING TELEMETRY", telemetry_layout), 1)
    right_zone.addStretch(1)

    window.training_progress = window._thin_progress()
    outer.addWidget(window.training_progress)
    QTimer.singleShot(0, window._refresh_training_layout)
    return page
