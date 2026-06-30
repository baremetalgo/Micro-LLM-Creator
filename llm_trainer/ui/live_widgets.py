from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QVBoxLayout, QSizePolicy, QWidget


class ModelFlowWidget(QWidget):
    """Transformer flow preview for live training telemetry."""

    def __init__(self) -> None:
        """Create the live model-flow visualization."""

        super().__init__()
        self.setObjectName("ModelFlow")
        self.setMinimumHeight(360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layer_count = 8
        self.head_count = 8
        self.step = 0
        self.loss_value: Optional[float] = None

    def set_state(self, layer_count: int, head_count: int, step: int, loss_value: Optional[float]) -> None:
        """Update flow state from UI-thread training metrics.

        Args:
            layer_count: Number of configured transformer layers.
            head_count: Number of configured attention heads.
            step: Current optimizer step.
            loss_value: Latest training loss, when available.
        """

        self.layer_count = max(1, int(layer_count))
        self.head_count = max(1, int(head_count))
        self.step = max(0, int(step))
        self.loss_value = float(loss_value) if loss_value is not None else None
        self.update()

    def paintEvent(self, event: Any) -> None:
        """Paint the transformer flow diagram.

        Args:
            event: Qt paint event.
        """

        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            rect = self.rect().adjusted(18, 18, -18, -18)
            painter.fillRect(self.rect(), QColor("#111111"))
            self._draw_grid(painter, rect)
            layer_total = min(max(self.layer_count, 4), 10)
            node_total = min(max(self.head_count, 5), 10)
            x_gap = rect.width() / max(layer_total + 1, 2)
            columns: list[list[QPointF]] = []
            labels = ["EMBED"] + [f"L{i}" for i in range(1, layer_total - 1)] + ["OUT"]
            for column in range(layer_total):
                x_value = rect.left() + x_gap * (column + 0.75)
                top = rect.top() + 64
                bottom = rect.bottom() - 64
                y_gap = (bottom - top) / max(node_total - 1, 1)
                points = [QPointF(x_value, top + y_gap * index) for index in range(node_total)]
                columns.append(points)
                self._draw_layer_box(painter, x_value, top, bottom, labels[column])
                for index, point in enumerate(points):
                    self._draw_node(painter, point, column, (index + column + self.step) % 4 == 0)
            self._draw_connections(painter, columns)
            self._draw_flow_arrows(painter, rect)
            self._draw_output_panel(painter, rect, columns[-1] if columns else [])
        finally:
            painter.end()

    def _draw_grid(self, painter: QPainter, rect: Any) -> None:
        """Draw a faint dashboard grid."""

        painter.setPen(QPen(QColor("#242424"), 1))
        for offset in range(0, max(1, rect.width()), 42):
            painter.drawLine(rect.left() + offset, rect.top(), rect.left() + offset, rect.bottom())
        for offset in range(0, max(1, rect.height()), 42):
            painter.drawLine(rect.left(), rect.top() + offset, rect.right(), rect.top() + offset)

    def _draw_layer_box(self, painter: QPainter, x_value: float, top: float, bottom: float, label: str) -> None:
        """Draw one transformer layer column."""

        box = QRectF(x_value - 20, top - 26, 40, bottom - top + 52)
        painter.setPen(QPen(QColor("#4a90e2"), 1))
        painter.setBrush(QBrush(QColor(35, 80, 120, 55)))
        painter.drawRoundedRect(box, 6, 6)
        painter.setPen(QPen(QColor("#e6e6e6")))
        painter.drawText(QRectF(x_value - 42, top - 54, 84, 22), Qt.AlignCenter, label)

    def _draw_node(self, painter: QPainter, point: QPointF, column: int, active: bool) -> None:
        """Draw one activation node."""

        palette = ["#4ab3ff", "#26d6c9", "#a978ff", "#f06aa3", "#ffb13b"]
        color = QColor(palette[column % len(palette)])
        if active:
            painter.setPen(QPen(color.lighter(135), 2))
            painter.setBrush(QBrush(color))
            radius = 7
        else:
            painter.setPen(QPen(QColor("#8a8a8a"), 1))
            painter.setBrush(QBrush(QColor(92, 92, 92, 150)))
            radius = 5
        painter.drawEllipse(point, radius, radius)

    def _draw_connections(self, painter: QPainter, columns: list[list[QPointF]]) -> None:
        """Draw sampled forward and backward pass connections."""

        for column_index in range(len(columns) - 1):
            source_points = columns[column_index]
            target_points = columns[column_index + 1]
            inactive = QColor(95, 95, 95, 90)
            forward = QColor("#3ed7ff")
            backward = QColor("#ff4ca8")
            for index, source in enumerate(source_points):
                direct_target = target_points[index % len(target_points)]
                shifted_target = target_points[(index + column_index + 1) % len(target_points)]
                self._draw_arrow_line(painter, source, direct_target, inactive, 1, False)
                active_forward = (index + column_index + self.step) % 4 == 0
                active_target = shifted_target if active_forward else direct_target
                if active_forward:
                    self._draw_arrow_line(painter, source, active_target, forward, 2, True)
                if index % 4 == (self.step + column_index) % 4:
                    back_source = target_points[(index + 1) % len(target_points)]
                    self._draw_arrow_line(painter, back_source, source, backward, 2, True)

    def _draw_arrow_line(
        self,
        painter: QPainter,
        start: QPointF,
        end: QPointF,
        color: QColor,
        width: int,
        active: bool,
    ) -> None:
        """Draw a straight directional connection between two nodes.

        Args:
            painter: Active painter.
            start: Connection start point.
            end: Connection end point.
            color: Line and arrow color.
            width: Line width in pixels.
            active: Whether this is a lit active connection.
        """

        painter.setPen(QPen(color, width, Qt.SolidLine))
        painter.setBrush(QBrush(color if active else QColor(color.red(), color.green(), color.blue(), 60)))
        painter.drawLine(start, end)
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        arrow_size = 7 if active else 5
        wing_a = QPointF(
            end.x() - arrow_size * math.cos(angle - math.pi / 6),
            end.y() - arrow_size * math.sin(angle - math.pi / 6),
        )
        wing_b = QPointF(
            end.x() - arrow_size * math.cos(angle + math.pi / 6),
            end.y() - arrow_size * math.sin(angle + math.pi / 6),
        )
        painter.drawPolygon(QPolygonF([end, wing_a, wing_b]))

    def _draw_flow_arrows(self, painter: QPainter, rect: Any) -> None:
        """Draw forward and backward pass indicators."""

        forward_y = rect.bottom() - 32
        backward_y = rect.bottom() - 12
        painter.setPen(QPen(QColor("#3ed7ff"), 2))
        painter.drawLine(rect.left() + 34, forward_y, rect.right() - 34, forward_y)
        painter.drawLine(rect.right() - 42, forward_y - 8, rect.right() - 34, forward_y)
        painter.drawLine(rect.right() - 42, forward_y + 8, rect.right() - 34, forward_y)
        painter.setPen(QPen(QColor("#ff4ca8"), 2))
        painter.drawLine(rect.right() - 34, backward_y, rect.left() + 34, backward_y)
        painter.drawLine(rect.left() + 42, backward_y - 8, rect.left() + 34, backward_y)
        painter.drawLine(rect.left() + 42, backward_y + 8, rect.left() + 34, backward_y)
        painter.setPen(QPen(QColor("#e6e6e6")))
        painter.drawText(rect.left() + 44, rect.top() + 20, "MODEL FLOW (LIVE)")
        painter.setPen(QPen(QColor("#3ed7ff")))
        painter.drawText(rect.left() + 220, rect.top() + 20, "Forward pass")
        painter.setPen(QPen(QColor("#ff4ca8")))
        painter.drawText(rect.left() + 330, rect.top() + 20, "Backward pass")

    def _draw_output_panel(self, painter: QPainter, rect: Any, points: list[QPointF]) -> None:
        """Draw a compact next-token preview panel."""

        panel = QRectF(rect.right() - 136, rect.top() + 112, 118, 122)
        painter.setPen(QPen(QColor("#4f7b48"), 1))
        painter.setBrush(QBrush(QColor(25, 55, 30, 160)))
        painter.drawRoundedRect(panel, 5, 5)
        painter.setPen(QPen(QColor("#d7d7d7")))
        painter.drawText(QRectF(panel.left(), panel.top() - 44, panel.width(), 32), Qt.AlignCenter, "OUTPUT\n(NEXT TOKEN)")
        samples = [("token", 0.64), ("code", 0.14), ("text", 0.09), ("data", 0.06), ("...", 0.03)]
        for index, (token, score) in enumerate(samples):
            y_value = panel.top() + 18 + index * 19
            painter.setPen(QPen(QColor("#b6d77a") if index == 0 else QColor("#d7d7d7")))
            painter.drawText(QRectF(panel.left() + 12, y_value - 9, 54, 18), Qt.AlignLeft | Qt.AlignVCenter, token)
            painter.drawText(QRectF(panel.right() - 42, y_value - 9, 34, 18), Qt.AlignRight | Qt.AlignVCenter, f"{score:.2f}")
        painter.setPen(QPen(QColor("#ffb13b"), 1))
        sample_stride = max(1, len(points) // 5) if points else 1
        for point in points[::sample_stride]:
            painter.drawLine(point, QPointF(panel.left(), panel.center().y()))


class LiveDistributionWidget(QWidget):
    """Bar-style next-token distribution preview."""

    def __init__(self) -> None:
        """Create a prediction distribution widget."""

        super().__init__()
        self.setMinimumHeight(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#141414")
        self.plot.setTitle("Prediction distribution", color="#eeeeee", size="10pt")
        self.plot.setLabel("bottom", "Vocabulary index", color="#d7d7d7")
        self.plot.setLabel("left", "Probability", color="#d7d7d7")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.bars = pg.BarGraphItem(x=list(range(32)), height=[0.0] * 32, width=0.7, brush="#35d36f")
        self.plot.addItem(self.bars)
        self.plot.setYRange(0, 1.0)
        layout.addWidget(self.plot)

    def update_distribution(self, step: int, loss_value: Optional[float]) -> None:
        """Update synthetic distribution from live training state.

        Args:
            step: Current optimizer step.
            loss_value: Latest loss used to shape confidence.
        """

        center = 12 + (step % 8)
        confidence = 0.9 if loss_value is None else max(0.25, min(0.95, 1.0 / max(float(loss_value), 1.0)))
        heights = [
            max(0.01, confidence * math.exp(-((index - center) ** 2) / 22.0) + 0.03 * ((index + step) % 5))
            for index in range(32)
        ]
        peak = max(heights) or 1.0
        self.bars.setOpts(x=list(range(32)), height=[value / peak for value in heights], width=0.7, brush="#35d36f")


class LiveHeatmapWidget(QWidget):
    """Attention-style heatmap preview."""

    def __init__(self) -> None:
        """Create an attention heatmap widget."""

        super().__init__()
        self.setMinimumHeight(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#141414")
        self.plot.setTitle("Attention", color="#eeeeee", size="10pt")
        self.plot.setLabel("bottom", "Key token", color="#d7d7d7")
        self.plot.setLabel("left", "Query token", color="#d7d7d7")
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.plot.setXRange(0, 8)
        self.plot.setYRange(0, 8)
        layout.addWidget(self.plot)
        self.update_heatmap(0, None)

    def update_heatmap(self, step: int, grad_norm: Optional[float]) -> None:
        """Update attention proxy heatmap.

        Args:
            step: Current optimizer step.
            grad_norm: Latest gradient norm.
        """

        scale = 1.0 if grad_norm is None else max(0.2, min(2.0, float(grad_norm)))
        data = np.zeros((8, 8), dtype=float)
        for row in range(8):
            for column in range(8):
                diagonal = math.exp(-abs(row - column) / 2.2)
                wave = 0.25 * (1.0 + math.sin((row + 1) * (column + 1) + step / 7.0))
                data[row, column] = min(1.0, (diagonal + wave) * scale / 1.8)
        self.image.setImage(data, autoLevels=True)


class LiveHistogramWidget(QWidget):
    """Activation distribution histogram preview."""

    def __init__(self) -> None:
        """Create an activation histogram widget."""

        super().__init__()
        self.setMinimumHeight(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#141414")
        self.plot.setTitle("Activation distribution", color="#eeeeee", size="10pt")
        self.plot.setLabel("bottom", "Activation", color="#d7d7d7")
        self.plot.setLabel("left", "Density", color="#d7d7d7")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.x_values = [index / 10.0 - 1.5 for index in range(31)]
        self.bars = pg.BarGraphItem(x=self.x_values, height=[0.0] * 31, width=0.08, brush="#13c5d8")
        self.plot.addItem(self.bars)
        self.plot.setYRange(0, 1.0)
        layout.addWidget(self.plot)

    def update_histogram(self, step: int, tokens_per_second: Optional[float]) -> None:
        """Update activation histogram proxy.

        Args:
            step: Current optimizer step.
            tokens_per_second: Latest training throughput.
        """

        spread = 0.45 + min(max(float(tokens_per_second or 0.0) / 50000.0, 0.0), 0.4)
        center = math.sin(step / 30.0) * 0.12
        heights = [math.exp(-((x_value - center) ** 2) / (2 * spread * spread)) for x_value in self.x_values]
        peak = max(heights) or 1.0
        self.bars.setOpts(x=self.x_values, height=[value / peak for value in heights], width=0.08, brush="#13c5d8")


class LiveGradientFlowWidget(QWidget):
    """Horizontal gradient-flow bar preview."""

    def __init__(self) -> None:
        """Create a gradient-flow widget."""

        super().__init__()
        self.setMinimumHeight(150)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#141414")
        self.plot.setTitle("Gradient flow", color="#eeeeee", size="10pt")
        self.plot.setLabel("bottom", "Gradient norm", color="#d7d7d7")
        self.plot.setLabel("left", "Layer", color="#d7d7d7")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.layers = list(range(1, 17))
        self.bars = pg.BarGraphItem(x0=[0] * 16, x1=[0.0] * 16, y=self.layers, height=0.55, brush="#ff4ca8")
        self.plot.addItem(self.bars)
        self.plot.setYRange(0, 17)
        self.plot.setXRange(0, 1)
        layout.addWidget(self.plot)

    def update_flow(self, layer_count: int, grad_norm: Optional[float], step: int) -> None:
        """Update gradient flow proxy bars.

        Args:
            layer_count: Configured model layer count.
            grad_norm: Latest gradient norm.
            step: Current optimizer step.
        """

        count = min(max(layer_count, 1), 32)
        self.layers = list(range(1, count + 1))
        base = max(0.05, min(float(grad_norm or 0.4), 4.0))
        values = [
            base * (0.25 + 0.75 * (index + 1) / count) * (0.78 + 0.22 * math.sin(step / 12.0 + index))
            for index in range(count)
        ]
        max_value = max(values) or 1.0
        self.bars.setOpts(x0=[0] * count, x1=values, y=self.layers, height=0.55, brush="#ff4ca8")
        self.plot.setYRange(0, count + 1)
        self.plot.setXRange(0, max_value * 1.15)
