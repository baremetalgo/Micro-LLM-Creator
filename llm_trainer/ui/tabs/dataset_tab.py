from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
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

from llm_trainer.conversation_datasets import CONVERSATION_DATASET_PRESETS


def build_dataset_tab(window) -> QWidget:
    """Build the dataset preparation page.

    Returns:
        Dataset page widget.
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
    title = window._page_title("Data Ingestion Matrix")
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
    window._configure_form(source_form)
    tokenizer_form = QFormLayout()
    window._configure_form(tokenizer_form)

    form = QFormLayout()
    window._configure_form(form)

    window.input_dir = QLineEdit()
    window._tip(window.input_dir, "Folder containing PDFs, text, Markdown, or JSONL files. More clean text usually improves the model.")
    window.dataset_dir = QLineEdit(str(Path.cwd() / "runs" / "dataset"))
    window._tip(window.dataset_dir, "Folder where prepared corpus, tokenizer, token files, and dataset summary are saved.")
    window.auto_vocab = QCheckBox("Choose automatically")
    window.auto_vocab.setChecked(True)
    window._tip(window.auto_vocab, "Automatically choose vocabulary size based on corpus size and word variety. Safer for most users.")
    window.manual_vocab_size = window._spin(256, 100000, 8000)
    window.manual_vocab_size.setEnabled(False)
    window._tip(window.manual_vocab_size, "Manual tokenizer vocabulary size. Larger vocab can preserve more words but increases model output size.")
    window.auto_vocab.toggled.connect(lambda checked: window.manual_vocab_size.setEnabled(not checked and not window._tokenizer_strategy_reuses()))
    window.auto_vocab_label = QLabel("Auto after reading files")
    window.auto_vocab_label.setObjectName("Metric")
    window._tip(window.auto_vocab_label, "The actual vocabulary size selected after reading the corpus.")
    window.min_frequency = window._spin(1, 1000, 2)
    window._tip(window.min_frequency, "Minimum token frequency for tokenizer training. Higher values remove rare fragments and can reduce noise.")
    window.context_length = window._spin(16, 4096, 128)
    window._tip(window.context_length, "Number of tokens per training sequence. Longer context lets the model learn longer dependencies but uses more memory.")
    window.validation_split = window._double_spin(0.0, 0.5, 0.1, 0.01, 3)
    window._tip(window.validation_split, "Fraction of tokens held out for validation. Validation helps detect overfitting during training.")
    window.lowercase = QCheckBox("Lowercase text")
    window._tip(window.lowercase, "Convert all text to lowercase. This shrinks vocabulary but removes capitalization patterns.")
    window.max_workers = window._spin(1, 64, 4)
    window._tip(window.max_workers, "Number of parallel file readers. More workers can speed PDF/text loading but uses more CPU and disk activity.")
    window.prepare_mode = QComboBox()
    window.prepare_mode.addItems(["Incremental update", "Full rebuild", "Force reprocess"])
    window.prepare_mode.setMaximumWidth(260)
    window._tip(
        window.prepare_mode,
        "Incremental update reuses cached extracted text and the existing tokenizer. Full rebuild rebuilds tokenizer/tokens. Force reprocess ignores cache.",
    )
    window.tokenizer_strategy = QComboBox()
    window.tokenizer_strategy.addItems(["Auto", "Train new tokenizer", "Reuse dataset tokenizer", "Import tokenizer.json"])
    window.tokenizer_strategy.setMaximumWidth(260)
    window._tip(
        window.tokenizer_strategy,
        "Controls tokenizer reuse. Auto reuses the dataset tokenizer during incremental updates; Import lets you use a compatible tokenizer.json.",
    )
    window.tokenizer_path = QLineEdit()
    window.tokenizer_path.setEnabled(False)
    window._tip(window.tokenizer_path, "Existing tokenizer.json to import. Use this when continuing a compatible tokenizer family.")
    window.tokenizer_strategy.currentTextChanged.connect(window._update_tokenizer_strategy_controls)
    window.code_training_mode = QCheckBox("Code Training Mode")
    window.code_training_mode.setChecked(True)
    window._tip(window.code_training_mode, "Prepare a programming dataset by preserving source code, tagging code/prose, and extracting code-like blocks.")
    window.include_prose = QCheckBox("Include explanations")
    window.include_prose.setChecked(True)
    window._tip(window.include_prose, "Keep prose from PDFs/books. This helps the model learn programming concepts and explanations.")
    window.include_source_code = QCheckBox("Include source files")
    window.include_source_code.setChecked(True)
    window._tip(window.include_source_code, "Include real code files such as .py, .js, .java, .cpp, .cs, .go, .rs, and similar.")
    window.extract_code_blocks = QCheckBox("Extract code blocks")
    window.extract_code_blocks.setChecked(True)
    window._tip(window.extract_code_blocks, "Try to detect code-like blocks inside PDFs/text and train them as code samples.")
    window.preserve_indentation = QCheckBox("Preserve indentation")
    window.preserve_indentation.setChecked(True)
    window._tip(window.preserve_indentation, "Keep line breaks and indentation for code. This is important for Python and readable generated code.")
    window.instruction_samples = QCheckBox("Instruction-style samples")
    window.instruction_samples.setChecked(True)
    window._tip(window.instruction_samples, "Wrap code samples with simple instruction tags so the model sees code as task-oriented examples.")
    window.reasoning_sample_mode = QComboBox()
    window.reasoning_sample_mode.addItems(["Reasoning scaffold", "Detailed code reasoning", "No reasoning wrapper"])
    window.reasoning_sample_mode.setMaximumWidth(260)
    window._tip(
        window.reasoning_sample_mode,
        "Shapes code samples as task/reasoning/answer examples. This teaches response structure, not guaranteed deep reasoning by itwindow.",
    )
    window.instruction_samples.toggled.connect(window.reasoning_sample_mode.setEnabled)

    conversation_form = QFormLayout()
    window._configure_form(conversation_form)
    window.dataset_stage = QComboBox()
    window.dataset_stage.addItems(["Base pretraining", "Instruction fine-tune", "Conversation fine-tune"])
    window.dataset_stage.setMaximumWidth(260)
    window._tip(
        window.dataset_stage,
        "Choose what this prepared dataset is for. Base datasets help language modeling; instruction/chat datasets belong to fine-tuning.",
    )
    window.include_conversation_datasets = QCheckBox("Include online training datasets")
    window.include_conversation_datasets.setChecked(False)
    window._tip(
        window.include_conversation_datasets,
        "Download/read selected Hugging Face datasets for the selected training purpose.",
    )
    conversation_form.addRow("Dataset purpose", window.dataset_stage)
    conversation_form.addRow("", window.include_conversation_datasets)
    window.conversation_datasets_status = QLabel("Base pretraining: choose optional online corpus datasets, or use local files only.")
    window.conversation_datasets_status.setObjectName("Muted")
    window._tip(
        window.conversation_datasets_status,
        "Turn on Include online conversation datasets to download/read the selected Hugging Face datasets.",
    )
    conversation_form.addRow("", window.conversation_datasets_status)
    window.conversation_dataset_checks = {}
    for dataset_id, preset in CONVERSATION_DATASET_PRESETS.items():
        checkbox = QCheckBox(preset.label)
        checkbox.setChecked(False)
        checkbox.setEnabled(False)
        window._tip(checkbox, preset.description)
        window.conversation_dataset_checks[dataset_id] = checkbox
        conversation_form.addRow("", checkbox)
    window.conversation_sample_limit = window._spin(0, 2_000_000, 20000)
    window._tip(
        window.conversation_sample_limit,
        "Maximum rows to read from each selected Hugging Face dataset. Use 0 only when you intentionally want the full dataset.",
    )
    window.conversation_sample_limit.setEnabled(False)
    window.include_conversation_datasets.toggled.connect(window.conversation_sample_limit.setEnabled)
    for checkbox in window.conversation_dataset_checks.values():
        window.include_conversation_datasets.toggled.connect(checkbox.setEnabled)
    window.include_conversation_datasets.toggled.connect(window._update_online_dataset_stage_controls)
    window.dataset_stage.currentTextChanged.connect(window._update_online_dataset_stage_controls)
    conversation_form.addRow("Rows per dataset", window.conversation_sample_limit)

    source_form.addRow("Source vault", window._path_row(window.input_dir, directory=True))
    source_form.addRow("Dataset core", window._path_row(window.dataset_dir, directory=True))
    source_form.addRow("Parallel lanes", window.max_workers)
    source_form.addRow("Prepare mode", window.prepare_mode)
    source_form.addRow("", window.lowercase)
    source_form.addRow("", window.code_training_mode)
    source_form.addRow("", window.include_source_code)

    tokenizer_form.addRow("Auto vocabulary", window.auto_vocab)
    tokenizer_form.addRow("Manual vocabulary", window.manual_vocab_size)
    tokenizer_form.addRow("Selected vocab", window.auto_vocab_label)
    window.tokenizer_path_row = window._path_row(window.tokenizer_path, directory=False, file_filter="Tokenizer JSON (*.json);;All files (*)")
    window.tokenizer_path_row.setEnabled(False)
    tokenizer_form.addRow("Tokenizer policy", window.tokenizer_strategy)
    tokenizer_form.addRow("Import tokenizer", window.tokenizer_path_row)
    tokenizer_form.addRow("Min frequency", window.min_frequency)
    tokenizer_form.addRow("Context window", window.context_length)
    tokenizer_form.addRow("Validation split", window.validation_split)
    tokenizer_form.addRow("", window.include_prose)
    tokenizer_form.addRow("", window.extract_code_blocks)
    tokenizer_form.addRow("", window.preserve_indentation)
    tokenizer_form.addRow("", window.instruction_samples)
    tokenizer_form.addRow("Reasoning samples", window.reasoning_sample_mode)
    source_card = window._card("SOURCE ARRAY", source_form)
    source_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    tokenizer_card = window._card("TOKENIZER CORE", tokenizer_form)
    tokenizer_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    conversation_card = window._card("CONVERSATION DATA", conversation_form)
    conversation_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    left_column.addWidget(source_card, 0)
    right_column.addWidget(tokenizer_card, 0)
    left_column.addWidget(conversation_card, 0)

    quality_grid = QGridLayout()
    quality_grid.setHorizontalSpacing(8)
    quality_grid.setVerticalSpacing(8)
    window.dataset_quality_samples = window._metric_chip("Samples: -", "Training samples after PDF/text/code expansion.")
    window.dataset_quality_tokens = window._metric_chip("Tokens: -", "Total encoded tokens available for training.")
    window.dataset_quality_vocab = window._metric_chip("Vocab: -", "Tokenizer vocabulary size used by the dataset.")
    window.dataset_quality_code = window._metric_chip("Code/prose: -", "Code and prose sample split.")
    window.dataset_quality_balance = window._metric_chip("Balance: -", "Code/prose balance detected during preview or preparation.")
    window.dataset_quality_readiness = window._metric_chip("Readiness: -", "Training readiness score based on size, duplicates, extraction quality, and dataset mix.")
    window.dataset_quality_cache = window._metric_chip("Cache: -", "Files reused from cache versus processed this run.")
    window.dataset_quality_duplicates = window._metric_chip("Duplicates: -", "Likely exact or extracted-text duplicate files.")
    window.dataset_quality_extraction = window._metric_chip("Extraction: -", "Files with suspicious text extraction quality.")
    window.dataset_quality_warning = window._metric_chip("Warnings: none", "Dataset quality warnings, if any.")
    quality_items = [
        window.dataset_quality_samples,
        window.dataset_quality_tokens,
        window.dataset_quality_vocab,
        window.dataset_quality_code,
        window.dataset_quality_balance,
        window.dataset_quality_readiness,
        window.dataset_quality_cache,
        window.dataset_quality_duplicates,
        window.dataset_quality_extraction,
        window.dataset_quality_warning,
    ]
    for index, item in enumerate(quality_items):
        quality_grid.addWidget(item, index // 2, index % 2)
    quality_card = window._card("DATASET QUALITY", quality_grid)
    quality_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    right_column.addWidget(quality_card, 0)

    window.dataset_advisor = QTextEdit()
    window.dataset_advisor.setReadOnly(True)
    window.dataset_advisor.setMinimumHeight(120)
    window.dataset_advisor.setMaximumHeight(190)
    window.dataset_advisor.setPlainText("Run Preview Dataset to get cleanup suggestions.")
    window._tip(window.dataset_advisor, "Actionable dataset cleanup advice from preview quality checks.")
    advisor_layout = QVBoxLayout()
    advisor_layout.addWidget(window.dataset_advisor)
    advisor_card = window._card("DATASET ADVISOR", advisor_layout)
    advisor_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    right_column.addWidget(advisor_card, 0)

    window.health_check_button = QPushButton("Check Health")
    window._tip(window.health_check_button, "Validate source, dataset, model, export, GGUF, and hardware readiness before long work.")
    window.health_check_button.clicked.connect(window.check_project_health)
    window.health_check_button.setMaximumWidth(160)
    window.preview_dataset_button = QPushButton("Preview Dataset")
    window._tip(window.preview_dataset_button, "Scan source files and show dataset quality plus sample text/code snippets without preparing tokens.")
    window.preview_dataset_button.clicked.connect(window.preview_dataset)
    window.preview_dataset_button.setMaximumWidth(180)
    window.prepare_button = QPushButton("Prepare Dataset")
    window._tip(window.prepare_button, "Read source files, clean text, train tokenizer, split tokens, and save the dataset project.")
    window.prepare_button.clicked.connect(window.prepare_dataset)
    window.prepare_button.setMaximumWidth(320)
    window.stop_dataset_button = QPushButton("Stop")
    window.stop_dataset_button.setEnabled(False)
    window.stop_dataset_button.setMaximumWidth(120)
    window.stop_dataset_button.clicked.connect(window.stop_active_task)
    window._tip(window.stop_dataset_button, "Request a graceful stop for dataset preparation.")

    window.dataset_log = QTextEdit()
    window.dataset_log.setReadOnly(True)
    window.dataset_log.setMinimumHeight(260)
    window.dataset_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    log_layout = QVBoxLayout()
    log_layout.addWidget(window.dataset_log, 1)
    left_column.addWidget(window._card("INGEST TELEMETRY", log_layout), 1)
    right_column.addStretch(1)
    layout.addLayout(ingestion_body, 1)
    action_row = QHBoxLayout()
    action_row.addWidget(window.health_check_button)
    action_row.addWidget(window.preview_dataset_button)
    action_row.addWidget(window.prepare_button)
    action_row.addWidget(window.stop_dataset_button)
    action_row.addStretch(1)
    layout.addLayout(action_row)

    window.dataset_progress = window._thin_progress()
    outer.addWidget(window.dataset_progress)
    window._update_online_dataset_stage_controls()
    return page
