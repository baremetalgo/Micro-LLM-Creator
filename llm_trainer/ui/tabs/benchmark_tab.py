from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from llm_trainer.evaluation import DEFAULT_BENCHMARK_PROMPTS


def build_benchmark_tab(window) -> QWidget:
    """Build the benchmark prompt page.

    Returns:
        Benchmark page widget.
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
    layout.addWidget(window._page_title("Benchmark Console"))

    benchmark_grid = QGridLayout()
    benchmark_grid.setHorizontalSpacing(12)
    benchmark_grid.setVerticalSpacing(8)
    window.benchmark_prompts = QTextEdit()
    window.benchmark_prompts.setMinimumHeight(260)
    window.benchmark_prompts.setPlainText("\n\n".join(DEFAULT_BENCHMARK_PROMPTS))
    window._tip(window.benchmark_prompts, "Benchmark prompts separated by blank lines. Run the same prompts after each training run.")
    window.benchmark_tokens = window._spin(16, 1024, 128)
    window._tip(window.benchmark_tokens, "Maximum generated tokens per benchmark prompt.")
    window.benchmark_temperature = window._double_spin(0.0, 2.0, 0.7, 0.05, 2)
    window._tip(window.benchmark_temperature, "Sampling randomness for benchmark generation.")
    window.benchmark_kv_cache = QCheckBox("Use KV cache")
    window.benchmark_kv_cache.setChecked(True)
    window._tip(window.benchmark_kv_cache, "Reuse attention key/value tensors during MicroGPT benchmark generation for faster inference.")
    window.run_benchmark_button = QPushButton("Run Benchmark")
    window.run_benchmark_button.setMaximumWidth(180)
    window.run_benchmark_button.clicked.connect(window.run_benchmark)
    window._tip(window.run_benchmark_button, "Generate benchmark outputs from final_model.pt and save a benchmark JSON file.")
    window.stop_benchmark_button = QPushButton("Stop")
    window.stop_benchmark_button.setMaximumWidth(120)
    window.stop_benchmark_button.setEnabled(False)
    window.stop_benchmark_button.clicked.connect(window.stop_active_task)
    window._tip(window.stop_benchmark_button, "Request a graceful stop for benchmark generation.")
    benchmark_grid.addWidget(window.benchmark_prompts, 0, 0, 5, 1)
    benchmark_grid.addWidget(QLabel("Max tokens"), 0, 1)
    benchmark_grid.addWidget(window.benchmark_tokens, 0, 2)
    benchmark_grid.addWidget(QLabel("Temperature"), 1, 1)
    benchmark_grid.addWidget(window.benchmark_temperature, 1, 2)
    benchmark_grid.addWidget(window.benchmark_kv_cache, 2, 1, 1, 2)
    benchmark_grid.addWidget(window.run_benchmark_button, 3, 1, 1, 2)
    benchmark_grid.addWidget(window.stop_benchmark_button, 4, 1, 1, 2)
    benchmark_grid.setColumnStretch(0, 1)
    layout.addWidget(window._card("BENCHMARK PROMPTS", benchmark_grid), 0)

    window.benchmark_log = QTextEdit()
    window.benchmark_log.setReadOnly(True)
    window.benchmark_log.setMinimumHeight(260)
    window.benchmark_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    benchmark_log_layout = QVBoxLayout()
    benchmark_log_layout.addWidget(window.benchmark_log, 1)
    layout.addWidget(window._card("BENCHMARK TELEMETRY", benchmark_log_layout), 1)

    window.benchmark_progress = window._thin_progress()
    outer.addWidget(window.benchmark_progress)
    return page
