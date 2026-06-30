from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from llm_trainer.ui.charts import LossChartWidget
from llm_trainer.ui.live_widgets import (
    LiveDistributionWidget,
    LiveGradientFlowWidget,
    LiveHeatmapWidget,
    LiveHistogramWidget,
    ModelFlowWidget,
)


def build_live_training_tab(window) -> QWidget:
    """Build the live training tracker page.

    Returns:
        Live training tracker page widget.
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

    header = QHBoxLayout()
    title = window._page_title("Model Training Live")
    live_badge = QLabel("● LIVE")
    live_badge.setObjectName("Metric")
    window.live_epoch_metric = window._metric_chip("Epoch: -", "Current epoch and total epochs.")
    window.live_step_metric = window._metric_chip("Step: -", "Current optimizer step and total planned steps.")
    window.live_tokens_metric = window._metric_chip("Tokens/sec: -", "Current token throughput.")
    window.live_loss_metric = window._metric_chip("Loss: -", "Latest training loss.")
    window.live_lr_metric = window._metric_chip("LR: -", "Current learning rate.")
    window.live_data_metric = window._metric_chip("Data: -", "Estimated percentage of planned training steps completed.")
    for chip in (
        window.live_epoch_metric,
        window.live_step_metric,
        window.live_tokens_metric,
        window.live_loss_metric,
        window.live_lr_metric,
        window.live_data_metric,
    ):
        chip.setMaximumWidth(180)
    header.addWidget(title)
    header.addWidget(live_badge)
    header.addSpacing(10)
    header.addWidget(window.live_epoch_metric)
    header.addWidget(window.live_step_metric)
    header.addWidget(window.live_tokens_metric)
    header.addWidget(window.live_loss_metric)
    header.addWidget(window.live_lr_metric)
    header.addWidget(window.live_data_metric)
    header.addStretch(1)
    layout.addLayout(header)

    body = QGridLayout()
    body.setSpacing(10)
    layout.addLayout(body, 1)

    left_column = QVBoxLayout()
    left_column.setSpacing(10)

    status_layout = QVBoxLayout()
    window.live_model_status = QLabel("◇ Model: Transformer decoder")
    window.live_layer_status = QLabel("▣ Layers: -")
    window.live_head_status = QLabel("◎ Heads: -")
    window.live_hidden_status = QLabel("▤ Hidden size: -")
    window.live_batch_status = QLabel("▥ Batch size: -")
    window.live_context_status = QLabel("▢ Context: -")
    for label in (
        window.live_model_status,
        window.live_layer_status,
        window.live_head_status,
        window.live_hidden_status,
        window.live_batch_status,
        window.live_context_status,
    ):
        status_layout.addWidget(label)
    left_column.addWidget(window._card("TRAINING STATUS", status_layout), 0)

    hardware_layout = QVBoxLayout()
    hardware_layout.setSpacing(7)
    window.live_device_status = QLabel("Device: -")
    window.live_cpu_label = QLabel("CPU: -")
    window.live_cpu_bar = window._hardware_meter("CPU")
    window.live_gpu_label = QLabel("GPU memory: -")
    window.live_gpu_bar = window._hardware_meter("GPU memory")
    window.live_vram_label = QLabel("VRAM reserved: -")
    window.live_vram_bar = window._hardware_meter("VRAM reserved")
    window.live_ram_label = QLabel("System RAM: -")
    window.live_ram_bar = window._hardware_meter("System RAM")
    window.live_worker_status = QLabel("CPU workers: -")
    window.hardware_meter_labels[id(window.live_cpu_bar)] = window.live_cpu_label
    window.hardware_meter_labels[id(window.live_gpu_bar)] = window.live_gpu_label
    window.hardware_meter_labels[id(window.live_vram_bar)] = window.live_vram_label
    window.hardware_meter_labels[id(window.live_ram_bar)] = window.live_ram_label
    hardware_layout.addWidget(window.live_device_status)
    for label, meter in (
        (window.live_cpu_label, window.live_cpu_bar),
        (window.live_gpu_label, window.live_gpu_bar),
        (window.live_vram_label, window.live_vram_bar),
        (window.live_ram_label, window.live_ram_bar),
    ):
        hardware_layout.addWidget(label)
        hardware_layout.addWidget(meter)
    hardware_layout.addWidget(window.live_worker_status)
    hardware_layout.addStretch(1)
    left_column.addWidget(window._card("HARDWARE", hardware_layout), 1)

    center_column = QVBoxLayout()
    center_column.setSpacing(10)
    flow_column = QVBoxLayout()
    flow_column.setContentsMargins(0, 0, 0, 0)
    flow_column.setSpacing(4)
    window.live_flow = ModelFlowWidget()
    window._tip(window.live_flow, "Visual summary of forward and backward flow through the configured transformer layers.")
    window.live_sample_text = QLabel("Training text: -")
    window.live_sample_text.setObjectName("TrainingSampleLine")
    window.live_sample_text.setWordWrap(False)
    window.live_sample_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
    window._tip(window.live_sample_text, "A compact preview of the current token window being used for training.")
    flow_column.addWidget(window.live_flow, 1)
    flow_column.addWidget(window.live_sample_text, 0)

    window.loss_chart = LossChartWidget("Train loss", "Validation loss", "Loss chart will appear during training", "Loss curve", "Cross entropy loss")
    window._tip(window.loss_chart, "Live training and validation loss. Falling values usually mean the model is learning.")
    window.optimization_chart = LossChartWidget("Learning rate", "Gradient norm", "Learning rate and gradient norm will appear during training", "Learning rate / gradient", "Value")
    window._tip(window.optimization_chart, "Learning rate and gradient norm. Watch for unstable spikes or gradients collapsing toward zero.")
    window.stability_chart = LossChartWidget("Weight norm", "Update ratio", "Weight norm and update ratio will appear during training", "Parameter stability", "Value")
    window._tip(window.stability_chart, "Weight norm and parameter update ratio. Large update ratios can destabilize training; tiny ratios can stall learning.")
    window.throughput_chart = LossChartWidget("Tokens/sec", "Samples/sec", "Throughput will appear during training", "Throughput", "Rate")
    window._tip(window.throughput_chart, "Training speed measured as tokens/sec and samples/sec.")
    window.memory_chart = LossChartWidget("VRAM allocated", "VRAM reserved", "VRAM usage will appear during CUDA training", "GPU memory", "GB")
    window._tip(window.memory_chart, "CUDA memory usage in GB. Helps diagnose memory bottlenecks.")
    charts_grid = QGridLayout()
    charts_grid.setHorizontalSpacing(8)
    charts_grid.setVerticalSpacing(8)
    charts_grid.addWidget(window.loss_chart, 0, 0)
    charts_grid.addWidget(window.optimization_chart, 0, 1)
    charts_grid.addWidget(window.stability_chart, 1, 0)
    charts_grid.addWidget(window.throughput_chart, 1, 1)
    charts_grid.addWidget(window.memory_chart, 2, 0, 1, 2)
    charts_grid.setColumnStretch(0, 1)
    charts_grid.setColumnStretch(1, 1)
    center_column.addWidget(window._card("TRAINING GRAPHS", charts_grid), 3)

    timeline_layout = QHBoxLayout()
    timeline_layout.setSpacing(8)
    window.live_time_slider = QSlider(Qt.Horizontal)
    window.live_time_slider.setRange(0, 0)
    window.live_time_slider.setValue(0)
    window.live_time_slider.sliderPressed.connect(window._begin_live_scrub)
    window.live_time_slider.sliderReleased.connect(window._end_live_scrub)
    window.live_time_slider.valueChanged.connect(window._scrub_live_timeline)
    window._tip(window.live_time_slider, "Drag to replay training graphs from recorded SQLite telemetry.")
    window.live_timeline_label = QLabel("Timeline: live")
    window.live_timeline_label.setObjectName("Metric")
    live_button = QPushButton("Live")
    live_button.setMaximumWidth(80)
    live_button.clicked.connect(window._jump_live_timeline_to_latest)
    window._tip(live_button, "Return the tracker to the latest live metrics.")
    timeline_layout.addWidget(QLabel("Time"))
    timeline_layout.addWidget(window.live_time_slider, 1)
    timeline_layout.addWidget(window.live_timeline_label)
    timeline_layout.addWidget(live_button)
    center_column.addWidget(window._card("TIMELINE", timeline_layout), 0)

    right_column = QVBoxLayout()
    right_column.setSpacing(10)
    window.live_prediction_chart = LiveDistributionWidget()
    window.live_attention_chart = LiveHeatmapWidget()
    window.live_activation_chart = LiveHistogramWidget()
    window.live_gradient_chart = LiveGradientFlowWidget()
    right_column.addWidget(window._card("PREDICTION DISTRIBUTION", single_widget_layout(window.live_prediction_chart)), 1)
    right_column.addWidget(window._card("ATTENTION", single_widget_layout(window.live_attention_chart)), 1)
    right_column.addWidget(window._card("ACTIVATION", single_widget_layout(window.live_activation_chart)), 1)
    right_column.addWidget(window._card("GRADIENT FLOW", single_widget_layout(window.live_gradient_chart)), 1)

    body.addLayout(flow_column, 0, 0, 1, 2)
    body.addLayout(left_column, 1, 0)
    body.addLayout(center_column, 1, 1)
    body.addLayout(right_column, 0, 2, 2, 1)
    body.setColumnStretch(0, 1)
    body.setColumnStretch(1, 4)
    body.setColumnStretch(2, 1)
    body.setRowStretch(0, 2)
    body.setRowStretch(1, 3)

    window.live_progress = window._thin_progress()
    outer.addWidget(window.live_progress)
    return page

def single_widget_layout(widget: QWidget) -> QVBoxLayout:
    """Wrap a single widget in a vertical layout.

    Args:
        widget: Widget to place in the layout.

    Returns:
        Layout containing the widget.
    """

    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget, 1)
    return layout
