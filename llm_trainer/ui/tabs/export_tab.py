from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def build_export_tab(window) -> QWidget:
    """Build the export page.

    Returns:
        Export page widget.
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
    layout.addWidget(window._page_title("Export Bay"))

    form = QFormLayout()
    window._configure_form(form)
    window.export_model_dir = QLineEdit(str(Path.cwd() / "runs" / "model"))
    window._tip(window.export_model_dir, "Trained model folder containing final_model.pt and tokenizer.json.")
    window.export_dir = QLineEdit(str(Path.cwd() / "runs" / "export"))
    window._tip(window.export_dir, "Folder where export bundles or quantized checkpoints are written.")
    window.quant_mode = QComboBox()
    window.quant_mode.addItems(["FP16 checkpoint", "GGUF Q8_0 (planned)", "GGUF Q4_K_M (planned)", "GGUF Q5_K_M (planned)"])
    window.quant_mode.setMaximumWidth(260)
    window._tip(window.quant_mode, "Quantization target. FP16 reduces checkpoint size now; GGUF modes are planned for llama.cpp export.")
    window.llama_cpp_dir = QLineEdit()
    window._tip(window.llama_cpp_dir, "Local llama.cpp checkout containing convert_hf_to_gguf.py.")
    window.gguf_output_path = QLineEdit(str(Path.cwd() / "runs" / "export" / "model.gguf"))
    window._tip(window.gguf_output_path, "Destination GGUF file. Requires an HF-compatible hf_model folder in the model core.")
    window.gguf_outtype = QComboBox()
    window.gguf_outtype.addItems(["f16", "f32", "bf16", "q8_0"])
    window.gguf_outtype.setMaximumWidth(260)
    window._tip(window.gguf_outtype, "llama.cpp converter outtype. f16 is the usual starting point.")
    form.addRow("Model core", window._path_row(window.export_model_dir, directory=True))
    form.addRow("Output bay", window._path_row(window.export_dir, directory=True))
    form.addRow("Quantization", window.quant_mode)
    form.addRow("llama.cpp", window._path_row(window.llama_cpp_dir, directory=True))
    form.addRow("GGUF output", window._path_row(window.gguf_output_path, directory=False, file_filter="GGUF models (*.gguf);;All files (*)"))
    form.addRow("GGUF outtype", window.gguf_outtype)
    layout.addWidget(window._card("ARTIFACT CONFIGURATION", form))

    row = QHBoxLayout()
    row.setSpacing(10)
    bundle_button = QPushButton("Create Bundle")
    window._tip(bundle_button, "Copy final model, tokenizer, and summary into a portable export folder.")
    bundle_button.clicked.connect(window.create_bundle)
    quant_button = QPushButton("Quantize Model")
    window._tip(quant_button, "Create a smaller FP16 checkpoint for inference or later conversion workflows.")
    quant_button.clicked.connect(window.quantize_model)
    hf_button = QPushButton("Export HF Package")
    window._tip(hf_button, "Create model_core/hf_model with config, weights, tokenizer, lineage, and README.")
    hf_button.clicked.connect(window.export_hf_package)
    window.gguf_convert_button = QPushButton("Convert HF to GGUF")
    window._tip(window.gguf_convert_button, "Run llama.cpp convert_hf_to_gguf.py for model_core/hf_model when available.")
    window.gguf_convert_button.clicked.connect(window.convert_hf_to_gguf)
    bundle_button.setMaximumWidth(220)
    quant_button.setMaximumWidth(220)
    hf_button.setMaximumWidth(220)
    window.gguf_convert_button.setMaximumWidth(220)
    row.addWidget(bundle_button)
    row.addWidget(quant_button)
    row.addWidget(hf_button)
    row.addWidget(window.gguf_convert_button)
    row.addStretch(1)

    window.export_log = QTextEdit()
    window.export_log.setReadOnly(True)
    window.export_log.setMinimumHeight(320)
    window.export_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    window.export_log.setPlainText(
        "Export options:\n"
        "- Bundle copies final_model.pt, tokenizer.json, and training_summary.json.\n"
        "- HF package writes model_core/hf_model for portable MicroGPT loading.\n"
        "- FP16 checkpoint quantization works now.\n"
        "- GGUF conversion uses llama.cpp when model_core/hf_model exists.\n"
        "- Native MicroGPT checkpoints are not written as fake GGUF files.\n"
    )
    export_log_layout = QVBoxLayout()
    export_log_layout.addWidget(window.export_log, 1)
    layout.addWidget(window._card("EXPORT TELEMETRY", export_log_layout), 1)
    layout.addLayout(row)

    window.export_progress = window._thin_progress()
    outer.addWidget(window.export_progress)
    return page
