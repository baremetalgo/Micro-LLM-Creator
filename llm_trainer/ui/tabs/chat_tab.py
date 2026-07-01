from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
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

from llm_trainer.ui.chat_widgets import ChatInputEdit


def build_chat_tab(window) -> QWidget:
    """Build the GGUF model test chat page.

    Returns:
        Chat page widget.
    """

    page = window._panel()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(24, 20, 24, 14)
    layout.setSpacing(12)

    main = QHBoxLayout()
    main.setSpacing(14)

    chat_column = QVBoxLayout()
    chat_column.setSpacing(10)

    window.chat_scroll = QScrollArea()
    window.chat_scroll.setObjectName("ChatScroll")
    window.chat_scroll.setWidgetResizable(True)
    window.chat_scroll.setMinimumHeight(420)
    window._tip(window.chat_scroll, "Rendered Markdown conversation view.")
    window.chat_canvas = QWidget()
    window.chat_canvas.setObjectName("ChatCanvas")
    window.chat_messages = QVBoxLayout(window.chat_canvas)
    window.chat_messages.setContentsMargins(14, 14, 14, 14)
    window.chat_messages.setSpacing(12)
    window.chat_messages.addStretch(1)
    window.chat_scroll.setWidget(window.chat_canvas)
    window.chat_event_log = QTextEdit()
    window.chat_event_log.setVisible(False)
    window._add_chat_message("assistant", "Load a GGUF model to start testing.")
    window.chat_stats = QLabel("Idle")
    window.chat_stats.setObjectName("Metric")
    window.chat_stats.setVisible(False)
    window._tip(window.chat_stats, "Generation timing, produced tokens, and approximate token speed.")
    chat_column.addWidget(window.chat_scroll, 1)

    prompt_row = QHBoxLayout()
    prompt_row.setSpacing(10)
    window.chat_input = ChatInputEdit()
    window.chat_input.setObjectName("ChatInput")
    window.chat_input.setMaximumHeight(92)
    window.chat_input.setPlaceholderText("Send a message...")
    window._tip(window.chat_input, "Prompt to send to the loaded model.")
    window.chat_input.sendRequested.connect(window.send_chat_message)
    window.send_chat_button = QPushButton("Send")
    window.send_chat_button.setMaximumWidth(120)
    window.send_chat_button.clicked.connect(window.send_chat_message)
    window._tip(window.send_chat_button, "Send the message to the already loaded model.")
    window.stop_chat_button = QPushButton("Stop")
    window.stop_chat_button.setMaximumWidth(120)
    window.stop_chat_button.setEnabled(False)
    window.stop_chat_button.clicked.connect(window.stop_active_task)
    window._tip(window.stop_chat_button, "Stop the current streamed reply.")
    prompt_row.addWidget(window.chat_input, 1)
    prompt_row.addWidget(window.send_chat_button)
    prompt_row.addWidget(window.stop_chat_button)
    chat_column.addLayout(prompt_row)

    settings_column = QVBoxLayout()
    settings_column.setSpacing(12)
    settings_panel = QWidget()
    settings_panel.setMaximumWidth(390)
    settings_panel.setMinimumWidth(340)
    settings_panel.setLayout(settings_column)

    model_form = QFormLayout()
    window._configure_form(model_form)
    window.chat_model_backend = QComboBox()
    window.chat_model_backend.addItems(["GGUF / llama.cpp", "MicroGPT checkpoint"])
    window.chat_model_backend.setMaximumWidth(260)
    window._tip(window.chat_model_backend, "Choose whether chat loads a GGUF model or a native MicroGPT final_model.pt checkpoint.")
    window.gguf_path = QLineEdit()
    window._tip(window.gguf_path, "Path to a GGUF model file produced by llama.cpp-compatible export tooling.")
    window.microgpt_chat_path = QLineEdit()
    window._tip(window.microgpt_chat_path, "MicroGPT final_model.pt checkpoint. A model folder containing final_model.pt and tokenizer.json also works if typed.")
    window.llama_context = window._spin(256, 131072, 2048)
    window._tip(window.llama_context, "llama.cpp context window. Larger values allow longer chats but use more memory.")
    window.llama_threads = window._spin(1, 128, 4)
    window._tip(window.llama_threads, "CPU threads used by llama.cpp inference.")
    window.llama_gpu_layers = window._spin(-1, 200, -1)
    window._tip(window.llama_gpu_layers, "Number of transformer layers to offload to GPU. Use -1 to offload all possible layers.")
    model_form.addRow("Model type", window.chat_model_backend)
    window.gguf_path_row = window._path_row(window.gguf_path, directory=False, file_filter="GGUF models (*.gguf);;All files (*)")
    window.microgpt_path_row = window._path_row(window.microgpt_chat_path, directory=False, file_filter="Checkpoints (*.pt);;All files (*)")
    model_form.addRow("GGUF model", window.gguf_path_row)
    model_form.addRow("MicroGPT checkpoint", window.microgpt_path_row)
    model_form.addRow("Context", window.llama_context)
    model_form.addRow("CPU threads", window.llama_threads)
    model_form.addRow("GPU layers", window.llama_gpu_layers)
    window.load_llm_button = QPushButton("Load Model")
    window.load_llm_button.setMaximumWidth(180)
    window.load_llm_button.clicked.connect(window.toggle_llm_model)
    window._tip(window.load_llm_button, "Load the selected model into memory once for repeated chat messages.")
    window.reset_chat_button = QPushButton("Reset Chat")
    window.reset_chat_button.setMaximumWidth(180)
    window.reset_chat_button.clicked.connect(window.reset_chat)
    window._tip(window.reset_chat_button, "Clear conversation memory while keeping the model loaded.")
    loader_buttons = QHBoxLayout()
    loader_buttons.addWidget(window.load_llm_button)
    loader_buttons.addWidget(window.reset_chat_button)
    loader_buttons.addStretch(1)
    model_form.addRow("", loader_buttons)
    window.chat_model_backend.currentTextChanged.connect(window._update_chat_backend_controls)

    sample_form = QFormLayout()
    window._configure_form(sample_form)
    window.thinking_enabled = QCheckBox("Thinking")
    window.thinking_enabled.setChecked(True)
    window._tip(window.thinking_enabled, "When enabled, the prompt asks the model to reason according to the selected effort level. Turn off for direct answers.")
    window.reasoning_effort = QComboBox()
    window.reasoning_effort.addItems(["Balanced", "Fast", "Deep"])
    window.reasoning_effort.setMaximumWidth(260)
    window._tip(window.reasoning_effort, "Controls the instruction style sent with each prompt. Deep asks for more careful reasoning.")
    window.thinking_enabled.toggled.connect(window.reasoning_effort.setEnabled)
    window.chat_max_tokens = window._spin(16, 8192, 512)
    window._tip(window.chat_max_tokens, "Maximum new tokens for each assistant reply.")
    window.chat_temperature = window._double_spin(0.0, 2.0, 0.7, 0.05, 2)
    window._tip(window.chat_temperature, "Sampling randomness. Lower is more focused; higher is more creative.")
    window.chat_top_p = window._double_spin(0.01, 1.0, 0.9, 0.01, 2)
    window._tip(window.chat_top_p, "Nucleus sampling. Lower values restrict the model to more likely tokens.")
    window.chat_repeat_penalty = window._double_spin(0.8, 2.0, 1.1, 0.01, 2)
    window._tip(window.chat_repeat_penalty, "Penalty for repeated text. Higher can reduce loops.")
    sample_form.addRow("", window.thinking_enabled)
    sample_form.addRow("Reasoning effort", window.reasoning_effort)
    sample_form.addRow("Max tokens", window.chat_max_tokens)
    sample_form.addRow("Temperature", window.chat_temperature)
    sample_form.addRow("Top-p", window.chat_top_p)
    sample_form.addRow("Repeat penalty", window.chat_repeat_penalty)

    window.system_prompt = QTextEdit()
    window.system_prompt.setObjectName("SystemPrompt")
    window.system_prompt.setMaximumHeight(120)
    window.system_prompt.setPlaceholderText("Optional system prompt")
    window._tip(window.system_prompt, "Optional behavior instruction sent to the model with each message.")
    system_layout = QVBoxLayout()
    system_layout.addWidget(window.system_prompt)

    settings_column.addWidget(window._card("MODEL LOADER", model_form))
    settings_column.addWidget(window._card("RESPONSE TUNING", sample_form))
    settings_column.addWidget(window._card("SYSTEM PROMPT", system_layout))
    settings_column.addStretch(1)

    main.addLayout(chat_column, 1)
    main.addWidget(settings_panel)
    layout.addLayout(main, 1)

    window.chat_progress = window._thin_progress()
    layout.addWidget(window.chat_progress)
    window._update_chat_backend_controls()
    return page
