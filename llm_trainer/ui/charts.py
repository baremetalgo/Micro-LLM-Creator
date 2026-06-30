from __future__ import annotations

import math
from typing import Optional

import pyqtgraph as pg
from PySide6.QtCore import QPointF
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QSizePolicy, QToolTip, QVBoxLayout, QWidget


class LossChartWidget(QWidget):
    """Interactive training metric chart backed by pyqtgraph."""

    def __init__(
        self,
        primary_label: str = "Train",
        secondary_label: str = "Val",
        empty_text: str = "Loss chart will appear during training",
        title: str = "Training Loss",
        y_label: str = "Loss",
    ) -> None:
        """Create an interactive chart widget.

        Args:
            primary_label: Label for the primary series.
            secondary_label: Label for the secondary series.
            empty_text: Text shown before samples arrive.
            title: Plot title.
            y_label: Y-axis label.
        """

        super().__init__()
        self.setObjectName("LossChart")
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.primary_label = primary_label
        self.secondary_label = secondary_label
        self.empty_text = empty_text
        self.title = title
        self.y_label = y_label
        self.train_points: list[tuple[int, float]] = []
        self.val_points: list[tuple[int, float]] = []
        self.max_draw_points = 260
        self.max_marker_points = 32
        self._last_hover_text = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("#141414")
        self.plot.setTitle(title, color="#eeeeee", size="11pt")
        self.plot.setLabel("bottom", "Optimizer step", color="#d7d7d7")
        self.plot.setLabel("left", y_label, color="#d7d7d7")
        self.plot.showGrid(x=True, y=True, alpha=0.28)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.getAxis("bottom").setPen(pg.mkPen("#6a6a6a"))
        self.plot.getAxis("left").setPen(pg.mkPen("#6a6a6a"))
        self.plot.getAxis("bottom").setTextPen(pg.mkPen("#cfcfcf"))
        self.plot.getAxis("left").setTextPen(pg.mkPen("#cfcfcf"))
        self.plot.getPlotItem().setContentsMargins(10, 8, 10, 8)
        self.legend = self.plot.addLegend(offset=(12, 8), brush=pg.mkBrush(20, 20, 20, 180), pen=pg.mkPen("#444444"))
        self.primary_curve = self.plot.plot([], [], pen=pg.mkPen("#f5b041", width=2), name=primary_label)
        self.secondary_curve = self.plot.plot([], [], pen=pg.mkPen("#b6d77a", width=2), name=secondary_label)
        self.primary_scatter = pg.ScatterPlotItem(size=4, brush=pg.mkBrush("#f5b041"), pen=pg.mkPen("#ffd27a"))
        self.secondary_scatter = pg.ScatterPlotItem(size=4, brush=pg.mkBrush("#b6d77a"), pen=pg.mkPen("#d8f39b"))
        self.plot.addItem(self.primary_scatter)
        self.plot.addItem(self.secondary_scatter)
        self.empty_label = pg.TextItem(empty_text, color="#9a9a9a", anchor=(0.5, 0.5))
        self.plot.addItem(self.empty_label)
        self.plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        layout.addWidget(self.plot)
        self._refresh_plot()

    def clear(self) -> None:
        """Remove all plotted loss values."""

        self.train_points.clear()
        self.val_points.clear()
        self._refresh_plot()

    def set_points(
        self,
        primary_points: list[tuple[int, float]],
        secondary_points: Optional[list[tuple[int, float]]] = None,
    ) -> None:
        """Replace chart points, usually when scrubbing timeline history.

        Args:
            primary_points: Primary series points.
            secondary_points: Optional secondary series points.
        """

        self.train_points = list(primary_points)
        self.val_points = list(secondary_points or [])
        self._refresh_plot()

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
        self.train_points = self.train_points[-2000:]
        self.val_points = self.val_points[-2000:]
        self._refresh_plot()

    def add_values(self, step: int, primary_value: Optional[float], secondary_value: Optional[float] = None) -> None:
        """Add a generic metric sample.

        Args:
            step: Optimizer step for the sample.
            primary_value: Primary series value.
            secondary_value: Optional secondary series value.
        """

        self.add_metrics(step, primary_value, secondary_value)

    def _refresh_plot(self) -> None:
        """Refresh curves, points, and empty-state text."""

        primary_points = self._decimate_points(self.train_points, self.max_draw_points)
        secondary_points = self._decimate_points(self.val_points, self.max_draw_points)
        primary_x = [step for step, _ in primary_points]
        primary_y = [value for _, value in primary_points]
        secondary_x = [step for step, _ in secondary_points]
        secondary_y = [value for _, value in secondary_points]
        self.primary_curve.setData(primary_x, primary_y)
        self.secondary_curve.setData(secondary_x, secondary_y)
        primary_marker_points = self._decimate_points(primary_points, self.max_marker_points)
        secondary_marker_points = self._decimate_points(secondary_points, self.max_marker_points)
        self.primary_scatter.setData([step for step, _ in primary_marker_points], [value for _, value in primary_marker_points])
        self.secondary_scatter.setData([step for step, _ in secondary_marker_points], [value for _, value in secondary_marker_points])
        all_points = self.train_points + self.val_points
        self.empty_label.setVisible(not all_points)
        if all_points:
            min_step = min(step for step, _ in all_points)
            max_step = max(step for step, _ in all_points)
            min_value = min(value for _, value in all_points)
            max_value = max(value for _, value in all_points)
            if min_step == max_step:
                max_step += 1
            if min_value == max_value:
                min_value -= 0.5
                max_value += 0.5
            padding = max((max_value - min_value) * 0.08, 1e-9)
            self.plot.setXRange(min_step, max_step, padding=0.04)
            self.plot.setYRange(min_value - padding, max_value + padding, padding=0)
        else:
            self.plot.setXRange(0, 1, padding=0)
            self.plot.setYRange(0, 1, padding=0)
            self.empty_label.setPos(0.5, 0.5)

    @staticmethod
    def _decimate_points(points: list[tuple[int, float]], max_points: int) -> list[tuple[int, float]]:
        """Return a readable sample of points for drawing.

        Args:
            points: Source points.
            max_points: Maximum points to draw.

        Returns:
            Decimated points retaining the first and last point.
        """

        if len(points) <= max_points:
            return points
        stride = max(1, math.ceil(len(points) / max_points))
        sampled = points[::stride]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        return sampled

    def _on_mouse_moved(self, position: QPointF) -> None:
        """Show nearest point values while hovering over the plot.

        Args:
            position: Scene mouse position.
        """

        plot_item = self.plot.getPlotItem()
        if not plot_item.sceneBoundingRect().contains(position):
            self._last_hover_text = ""
            self.plot.setToolTip("")
            return
        mouse_point = plot_item.vb.mapSceneToView(position)
        nearest = self._nearest_point(float(mouse_point.x()), float(mouse_point.y()))
        if nearest is None:
            self._last_hover_text = ""
            self.plot.setToolTip("")
            return
        series, step, value = nearest
        text = f"{self.title}\n{series}\nStep: {step:,}\n{self.y_label}: {value:.6g}"
        if text != self._last_hover_text:
            self._last_hover_text = text
            self.plot.setToolTip(text)
        QToolTip.showText(QCursor.pos(), text, self.plot)

    def _nearest_point(self, x_value: float, y_value: float) -> Optional[tuple[str, int, float]]:
        """Return the nearest plotted point to a hover position.

        Args:
            x_value: Hover x value in data coordinates.
            y_value: Hover y value in data coordinates.

        Returns:
            Series label, step, and value when a point is close enough.
        """

        points: list[tuple[str, int, float]] = [
            (self.primary_label, step, value) for step, value in self.train_points
        ] + [
            (self.secondary_label, step, value) for step, value in self.val_points
        ]
        if not points:
            return None
        view_range = self.plot.getPlotItem().vb.viewRange()
        x_span = max(view_range[0][1] - view_range[0][0], 1e-9)
        y_span = max(view_range[1][1] - view_range[1][0], 1e-9)
        nearest = min(
            points,
            key=lambda item: ((item[1] - x_value) / x_span) ** 2 + ((item[2] - y_value) / y_span) ** 2,
        )
        distance = ((nearest[1] - x_value) / x_span) ** 2 + ((nearest[2] - y_value) / y_span) ** 2
        return nearest if distance < 0.01 else None
