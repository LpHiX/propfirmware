import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, cast

import h5py
import numpy as np

QtCore: Any
QtGui: Any
QtWidgets: Any
pg: Any
ne: Any

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except ImportError:
    QtCore = None
    QtGui = None
    QtWidgets = None
    pg = None

try:
    import numexpr as ne  # pyright: ignore[reportMissingImports]
except ImportError:
    ne = None

BaseWindow: type = cast(type, QtWidgets.QMainWindow) if QtWidgets is not None else object
BaseWidget: type = cast(type, QtWidgets.QWidget) if QtWidgets is not None else object


DEFAULT_LOGS_ROOT = Path(r"D:\Projects\propbackend_logs")


@dataclass(frozen=True)
class ChannelRecord:
    name: str
    unit: str
    signal_type: str
    samples: int
    time: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class DerivedChannelDefinition:
    name: str
    expression: str
    unit: str


class H5FileTab(BaseWidget):
    def __init__(self) -> None:
        super().__init__()

        self.current_file: Path | None = None
        self.base_channels: Dict[str, ChannelRecord] = {}
        self.channels: Dict[str, ChannelRecord] = {}
        self.derived_channels: Dict[str, DerivedChannelDefinition] = {}
        self.selected_channels: set[str] = set()
        self.unit_plots: Dict[str, Any] = {}
        self.unit_curves: Dict[str, Dict[str, Any]] = {}
        self.unit_regions: Dict[str, Any] = {}
        self.unit_region_actions: Dict[str, Any] = {}
        self.unit_metric_actions: Dict[str, Dict[str, Any]] = {}
        self.active_region_unit: str | None = None
        self.color_cycle = itertools.cycle(
            [
                "#1f77b4",
                "#d62728",
                "#2ca02c",
                "#ff7f0e",
                "#17becf",
                "#e377c2",
                "#bcbd22",
                "#7f7f7f",
            ]
        )
        self.channel_color: Dict[str, str] = {}
        self.max_points_per_curve = 6000
        self._table_mutation = False
        self._derived_config_dir = Path(__file__).resolve().parent / "h5viewer_config"
        self._derived_config_file = self._derived_config_dir / "derived_channels.json"
        self.metric_labels: Dict[str, str] = {
            "integral": "Integral",
            "min": "Min",
            "max": "Max",
            "mean": "Mean",
            "rms": "RMS",
            "std": "Std Dev",
        }
        self.metric_order = ["integral", "min", "max", "mean", "rms", "std"]
        self.enabled_metrics: set[str] = {"integral"}

        self._load_derived_config()
        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(7)
        root.addWidget(splitter, 1)

        left_panel = QtWidgets.QFrame()
        left_panel.setMinimumWidth(320)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        self.file_label = QtWidgets.QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        left_layout.addWidget(self.file_label)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Filter channels (name / unit / signal type)")
        left_layout.addWidget(self.search_edit)

        self.use_raw_checkbox = QtWidgets.QCheckBox("Use raw data when available")
        self.use_raw_checkbox.setChecked(False)
        left_layout.addWidget(self.use_raw_checkbox)

        self.link_x_checkbox = QtWidgets.QCheckBox("Link X pan/zoom across unit plots")
        self.link_x_checkbox.setChecked(True)
        left_layout.addWidget(self.link_x_checkbox)

        derived_box = QtWidgets.QGroupBox("Derived channels")
        derived_layout = QtWidgets.QVBoxLayout(derived_box)
        derived_layout.setContentsMargins(8, 8, 8, 8)
        derived_layout.setSpacing(6)

        self.derived_name_edit = QtWidgets.QLineEdit()
        self.derived_name_edit.setPlaceholderText("Name (e.g. flow_lps)")
        derived_layout.addWidget(self.derived_name_edit)

        self.derived_expr_edit = QtWidgets.QLineEdit()
        self.derived_expr_edit.setPlaceholderText("Expression with channel names (e.g. flow_main / 60)")
        derived_layout.addWidget(self.derived_expr_edit)

        self.derived_unit_edit = QtWidgets.QLineEdit()
        self.derived_unit_edit.setPlaceholderText("Output unit (e.g. L/s)")
        derived_layout.addWidget(self.derived_unit_edit)

        derived_button_row = QtWidgets.QHBoxLayout()
        self.add_derived_button = QtWidgets.QPushButton("Add / Update")
        self.remove_derived_button = QtWidgets.QPushButton("Remove Selected")
        derived_button_row.addWidget(self.add_derived_button)
        derived_button_row.addWidget(self.remove_derived_button)
        derived_layout.addLayout(derived_button_row)

        self.derived_list = QtWidgets.QListWidget()
        self.derived_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.derived_list.setMaximumHeight(115)
        derived_layout.addWidget(self.derived_list)

        self.derived_help_label = QtWidgets.QLabel("Use exact channel names in expressions. Operators: + - * / **")
        self.derived_help_label.setWordWrap(True)
        derived_layout.addWidget(self.derived_help_label)

        left_layout.addWidget(derived_box)

        self.channel_table = QtWidgets.QTableWidget(0, 5)
        self.channel_table.setHorizontalHeaderLabels(["Plot", "Channel", "Unit", "Signal", "Samples"])
        self.channel_table.verticalHeader().setVisible(False)
        self.channel_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.channel_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.channel_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.channel_table.setAlternatingRowColors(True)
        self.channel_table.setSortingEnabled(True)
        header = self.channel_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        left_layout.addWidget(self.channel_table, 1)

        self.clear_button = QtWidgets.QPushButton("Clear all plotted channels")
        left_layout.addWidget(self.clear_button)

        self.region_info_label = QtWidgets.QLabel("Region: none")
        left_layout.addWidget(self.region_info_label)

        self.analysis_table = QtWidgets.QTableWidget(0, 0)
        self.analysis_table.verticalHeader().setVisible(False)
        self.analysis_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.analysis_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.analysis_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.analysis_table.setAlternatingRowColors(True)
        self.analysis_table.setMaximumHeight(185)
        self._rebuild_analysis_table_columns()
        left_layout.addWidget(self.analysis_table)

        splitter.addWidget(left_panel)

        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.plot_container = QtWidgets.QWidget()
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(8)
        self.plot_layout.addStretch(1)

        self.plot_scroll.setWidget(self.plot_container)
        splitter.addWidget(self.plot_scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([450, 1100])

    def _connect_signals(self) -> None:
        self.search_edit.textChanged.connect(self._apply_table_filter)
        self.channel_table.cellClicked.connect(self._on_table_clicked)
        self.channel_table.itemChanged.connect(self._on_table_item_changed)
        self.use_raw_checkbox.stateChanged.connect(self._reload_current_file_data)
        self.link_x_checkbox.stateChanged.connect(self._relink_unit_plots)
        self.clear_button.clicked.connect(self._clear_all_channels)
        self.add_derived_button.clicked.connect(self._add_or_update_derived_channel)
        self.remove_derived_button.clicked.connect(self._remove_selected_derived_channel)
        self.derived_list.itemSelectionChanged.connect(self._on_derived_selected)

    def load_file(self, path: Path) -> None:
        try:
            self.base_channels = self._read_channels(path, use_raw=self.use_raw_checkbox.isChecked())
            self.channels = dict(self.base_channels)
            self._apply_derived_channels_to_loaded_data()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open failed", f"Failed to open file:\n{exc}")
            return

        self.current_file = path
        self.selected_channels.clear()
        self.unit_plots.clear()
        self.unit_curves.clear()
        self.unit_regions.clear()
        self.unit_region_actions.clear()
        self.unit_metric_actions.clear()
        self.channel_color.clear()
        self.active_region_unit = None
        self._clear_plot_widgets()
        self._clear_analysis_table()

        self.file_label.setText(f"Loaded: {path}")
        self._refresh_derived_list()
        self._populate_channel_table()

    def _reload_current_file_data(self) -> None:
        if self.current_file is None:
            return
        selected_before = set(self.selected_channels)
        self.load_file(self.current_file)
        for channel in selected_before:
            if channel in self.channels:
                self._set_table_checked(channel, True)
                self._set_channel_enabled(channel, True)

    def _read_channels(self, path: Path, use_raw: bool) -> Dict[str, ChannelRecord]:
        channels: Dict[str, ChannelRecord] = {}

        with h5py.File(path, "r") as h5:
            if "channels" not in h5 or not isinstance(h5["channels"], h5py.Group):
                return channels

            for channel_name in h5["channels"].keys():
                group = h5["channels"][channel_name]
                if not isinstance(group, h5py.Group):
                    continue
                if "time" not in group or "data" not in group:
                    continue

                time_values = np.asarray(group["time"][:], dtype=float)
                if use_raw and "raw" in group:
                    data_values = np.asarray(group["raw"][:], dtype=float)
                else:
                    data_values = np.asarray(group["data"][:], dtype=float)

                if len(time_values) != len(data_values):
                    size = min(len(time_values), len(data_values))
                    time_values = time_values[:size]
                    data_values = data_values[:size]

                channels[channel_name] = ChannelRecord(
                    name=channel_name,
                    unit=str(group.attrs.get("unit", "unknown")) or "unknown",
                    signal_type=str(group.attrs.get("signal_type", "unknown")) or "unknown",
                    samples=len(time_values),
                    time=time_values,
                    values=data_values,
                )

        return channels

    def _populate_channel_table(self) -> None:
        self.channel_table.setSortingEnabled(False)
        self.channel_table.setRowCount(0)

        for row, name in enumerate(sorted(self.channels.keys())):
            channel = self.channels[name]
            self.channel_table.insertRow(row)

            check_item = QtWidgets.QTableWidgetItem("")
            check_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            check_item.setCheckState(QtCore.Qt.Unchecked)
            check_item.setData(QtCore.Qt.UserRole, channel.name)
            self.channel_table.setItem(row, 0, check_item)

            self.channel_table.setItem(row, 1, QtWidgets.QTableWidgetItem(channel.name))
            self.channel_table.setItem(row, 2, QtWidgets.QTableWidgetItem(channel.unit))
            self.channel_table.setItem(row, 3, QtWidgets.QTableWidgetItem(channel.signal_type))
            self.channel_table.setItem(row, 4, QtWidgets.QTableWidgetItem(str(channel.samples)))

        self.channel_table.setSortingEnabled(True)
        self._apply_table_filter()

    def _apply_table_filter(self) -> None:
        query = self.search_edit.text().strip().lower()
        for row in range(self.channel_table.rowCount()):
            name_item = self.channel_table.item(row, 1)
            unit_item = self.channel_table.item(row, 2)
            signal_item = self.channel_table.item(row, 3)
            haystack = " ".join(
                [
                    name_item.text().lower() if name_item else "",
                    unit_item.text().lower() if unit_item else "",
                    signal_item.text().lower() if signal_item else "",
                ]
            )
            self.channel_table.setRowHidden(row, query not in haystack)

    def _on_table_clicked(self, row: int, column: int) -> None:
        if column == 0:
            return

        check_item = self.channel_table.item(row, 0)
        if check_item is None:
            return

        channel_name = check_item.data(QtCore.Qt.UserRole)
        if not isinstance(channel_name, str):
            return

        new_checked = check_item.checkState() != QtCore.Qt.Checked
        self._set_table_checked(channel_name, new_checked)

    def _on_table_item_changed(self, item) -> None:
        if self._table_mutation:
            return
        if item.column() != 0:
            return

        channel_name = item.data(QtCore.Qt.UserRole)
        if not isinstance(channel_name, str):
            return

        enabled = item.checkState() == QtCore.Qt.Checked
        self._set_channel_enabled(channel_name, enabled)

    def _set_table_checked(self, channel_name: str, checked: bool) -> None:
        self._table_mutation = True
        try:
            for row in range(self.channel_table.rowCount()):
                check_item = self.channel_table.item(row, 0)
                if check_item is None:
                    continue
                if check_item.data(QtCore.Qt.UserRole) == channel_name:
                    check_item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
                    break
        finally:
            self._table_mutation = False

        self._set_channel_enabled(channel_name, checked)

    def _set_channel_enabled(self, channel_name: str, enabled: bool) -> None:
        channel = self.channels.get(channel_name)
        if channel is None:
            return

        had_no_channels = len(self.selected_channels) == 0

        if enabled:
            self.selected_channels.add(channel_name)
            self._ensure_unit_plot(channel.unit)
            self._add_curve(channel)
            if had_no_channels:
                self._view_all_plots()
        else:
            self.selected_channels.discard(channel_name)
            self._remove_curve(channel)
            self._prune_empty_unit_plots()

        if self.unit_regions:
            self._refresh_analysis_table_from_active_regions()

    def _ensure_unit_plot(self, unit: str) -> None:
        if unit in self.unit_plots:
            return

        plot = pg.PlotWidget(title=f"Unit: {unit}")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("left", f"Value ({unit})")
        plot.setLabel("bottom", "Time (s)")
        plot.addLegend(offset=(8, 8))

        view_box = plot.getViewBox()
        toggle_action = QtGui.QAction("Select Region", plot)
        toggle_action.triggered.connect(lambda _checked=False, unit_name=unit: self._toggle_region_for_unit(unit_name))
        metrics_menu = QtWidgets.QMenu("Region Metrics", plot)
        metric_actions: Dict[str, Any] = {}
        for metric_key in self.metric_order:
            metric_action = QtGui.QAction(self.metric_labels[metric_key], plot)
            metric_action.setCheckable(True)
            metric_action.setChecked(metric_key in self.enabled_metrics)
            metric_action.triggered.connect(
                lambda checked, key=metric_key: self._set_metric_enabled(key, bool(checked))
            )
            metrics_menu.addAction(metric_action)
            metric_actions[metric_key] = metric_action

        if hasattr(view_box, "menu") and view_box.menu is not None:
            view_box.menu.addSeparator()
            view_box.menu.addAction(toggle_action)
            view_box.menu.addMenu(metrics_menu)

        self.plot_layout.insertWidget(max(0, self.plot_layout.count() - 1), plot, 1)
        self.unit_plots[unit] = plot
        self.unit_curves[unit] = {}
        self.unit_region_actions[unit] = toggle_action
        self.unit_metric_actions[unit] = metric_actions
        self._relink_unit_plots()

    def _add_curve(self, channel: ChannelRecord) -> None:
        if channel.name not in self.channel_color:
            self.channel_color[channel.name] = next(self.color_cycle)

        plot = self.unit_plots[channel.unit]
        x_values, y_values = self._downsample_for_display(channel.time, channel.values)
        pen = pg.mkPen(self.channel_color[channel.name], width=2)
        curve = plot.plot(x_values, y_values, pen=pen, name=channel.name)
        curve.setClipToView(True)
        curve.setDownsampling(auto=True, method="peak")
        self.unit_curves[channel.unit][channel.name] = curve

    def _downsample_for_display(self, x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x_values) <= self.max_points_per_curve:
            return x_values, y_values

        step = max(1, len(x_values) // self.max_points_per_curve)
        return x_values[::step], y_values[::step]

    def _remove_curve(self, channel: ChannelRecord) -> None:
        curves = self.unit_curves.get(channel.unit, {})
        curve = curves.pop(channel.name, None)
        if curve is not None:
            self.unit_plots[channel.unit].removeItem(curve)

    def _prune_empty_unit_plots(self) -> None:
        removed_any_plot = False
        for unit in list(self.unit_plots.keys()):
            if self.unit_curves.get(unit):
                continue
            plot = self.unit_plots.pop(unit)
            self.unit_curves.pop(unit, None)
            self.unit_regions.pop(unit, None)
            self.unit_region_actions.pop(unit, None)
            self.unit_metric_actions.pop(unit, None)
            self.plot_layout.removeWidget(plot)
            plot.deleteLater()
            removed_any_plot = True
        self._relink_unit_plots()
        if self.active_region_unit not in self.unit_plots:
            self.active_region_unit = None
        if removed_any_plot:
            if self.unit_regions:
                self._refresh_analysis_table_from_active_regions()
            else:
                self._clear_analysis_table()

    def _relink_unit_plots(self) -> None:
        plots = list(self.unit_plots.values())
        if not plots:
            return

        for plot in plots:
            plot.setXLink(None)

        if not self.link_x_checkbox.isChecked():
            return

        leader = plots[0]
        for plot in plots[1:]:
            plot.setXLink(leader)

    def _view_all_plots(self) -> None:
        for plot in self.unit_plots.values():
            plot.enableAutoRange(axis=pg.ViewBox.XYAxes)
            plot.autoRange()

    def _clear_all_channels(self) -> None:
        self.selected_channels.clear()
        self.unit_curves.clear()
        self.unit_plots.clear()
        self.unit_regions.clear()
        self.unit_region_actions.clear()
        self.unit_metric_actions.clear()
        self.channel_color.clear()
        self.active_region_unit = None

        self._clear_plot_widgets()
        self._clear_analysis_table()

        for row in range(self.channel_table.rowCount()):
            self._table_mutation = True
            check_item = self.channel_table.item(row, 0)
            if check_item is not None:
                check_item.setCheckState(QtCore.Qt.Unchecked)
            self._table_mutation = False

    def _clear_plot_widgets(self) -> None:
        while self.plot_layout.count() > 1:
            item = self.plot_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _toggle_region_for_unit(self, unit: str) -> None:
        if unit in self.unit_regions:
            self._remove_region_for_unit(unit)
            return
        self._create_region_for_unit(unit)

    def _create_region_for_unit(self, unit: str) -> None:
        plot = self.unit_plots.get(unit)
        if plot is None:
            return

        x_start, x_end = self._region_default_bounds_for_unit(unit)
        region = pg.LinearRegionItem(values=[x_start, x_end], movable=True)
        region.setZValue(100)
        plot.addItem(region)
        region.sigRegionChanged.connect(lambda _=None, unit_name=unit: self._on_region_changed(unit_name))
        self.unit_regions[unit] = region

        action = self.unit_region_actions.get(unit)
        if action is not None:
            action.setText("Deselect Region")

        self._on_region_changed(unit)

    def _remove_region_for_unit(self, unit: str) -> None:
        plot = self.unit_plots.get(unit)
        region = self.unit_regions.pop(unit, None)
        if plot is not None and region is not None:
            plot.removeItem(region)

        action = self.unit_region_actions.get(unit)
        if action is not None:
            action.setText("Select Region")

        if self.active_region_unit == unit:
            self.active_region_unit = None

        if self.unit_regions:
            self._refresh_analysis_table_from_active_regions()
        else:
            self._clear_analysis_table()

    def _region_default_bounds_for_unit(self, unit: str) -> tuple[float, float]:
        plot = self.unit_plots.get(unit)
        if plot is not None:
            x_range = plot.getViewBox().viewRange()[0]
            x_min = float(x_range[0])
            x_max = float(x_range[1])
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                span = x_max - x_min
                return x_min + 0.25 * span, x_min + 0.75 * span

        curves = self.unit_curves.get(unit, {})
        for channel_name in curves.keys():
            record = self.channels.get(channel_name)
            if record is None or len(record.time) < 2:
                continue
            t0 = float(record.time[0])
            t1 = float(record.time[-1])
            if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
                span = t1 - t0
                return t0 + 0.25 * span, t0 + 0.75 * span

        return 0.0, 1.0

    def _on_region_changed(self, unit: str) -> None:
        region = self.unit_regions.get(unit)
        if region is None:
            return

        start, end = region.getRegion()
        if end < start:
            start, end = end, start

        self.active_region_unit = unit
        self._refresh_analysis_table_from_active_regions()

    def _refresh_analysis_table_from_active_regions(self) -> None:
        self.analysis_table.setRowCount(0)
        region_summaries: list[str] = []

        for unit in sorted(self.unit_regions.keys()):
            region = self.unit_regions.get(unit)
            if region is None:
                continue

            start, end = region.getRegion()
            if end < start:
                start, end = end, start

            region_summaries.append(f"{unit}: {start:.3f} to {end:.3f} s")
            self._append_analysis_rows_for_unit(unit, start, end)

        if region_summaries:
            self.region_info_label.setText("Regions: " + " | ".join(region_summaries))
        else:
            self.region_info_label.setText("Region: none")

    def _append_analysis_rows_for_unit(self, unit: str, start: float, end: float) -> None:
        channel_names = sorted(self.unit_curves.get(unit, {}).keys())

        duration = max(0.0, end - start)
        for channel_name in channel_names:
            record = self.channels.get(channel_name)
            if record is None or len(record.time) < 2:
                continue

            mask = (record.time >= start) & (record.time <= end)
            if int(np.count_nonzero(mask)) < 2:
                metric_values = self._empty_metric_values()
            else:
                window_time = record.time[mask]
                window_values = record.values[mask]
                metric_values = self._compute_metric_values(window_time, window_values, record.unit)

            row_index = self.analysis_table.rowCount()
            self.analysis_table.insertRow(row_index)
            self.analysis_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(channel_name))
            self.analysis_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(record.unit))
            self.analysis_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(f"{duration:.6g}"))
            col = 3
            for metric_key in self.metric_order:
                if metric_key in self.enabled_metrics:
                    self.analysis_table.setItem(row_index, col, QtWidgets.QTableWidgetItem(metric_values[metric_key]))
                    col += 1

    def _integral_unit_for(self, input_unit: str) -> str:
        if "/s" in input_unit:
            return input_unit.split("/s", 1)[0].strip() or "(unitless)"
        return f"{input_unit}*s"

    def _clear_analysis_table(self) -> None:
        self.analysis_table.setRowCount(0)
        self.region_info_label.setText("Region: none")

    def _rebuild_analysis_table_columns(self) -> None:
        headers = ["Channel", "Unit", "Duration (s)"]
        for metric_key in self.metric_order:
            if metric_key in self.enabled_metrics:
                headers.append(self.metric_labels[metric_key])

        self.analysis_table.setColumnCount(len(headers))
        self.analysis_table.setHorizontalHeaderLabels(headers)
        header = self.analysis_table.horizontalHeader()
        if len(headers) > 0:
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for col in range(1, len(headers)):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)

    def _set_metric_enabled(self, metric_key: str, enabled: bool) -> None:
        if metric_key not in self.metric_labels:
            return

        if enabled:
            self.enabled_metrics.add(metric_key)
        else:
            self.enabled_metrics.discard(metric_key)

        self._sync_metric_action_checks()
        self._rebuild_analysis_table_columns()
        if self.unit_regions:
            self._refresh_analysis_table_from_active_regions()
        else:
            self._clear_analysis_table()

    def _sync_metric_action_checks(self) -> None:
        for action_map in self.unit_metric_actions.values():
            for metric_key, action in action_map.items():
                should_check = metric_key in self.enabled_metrics
                old_state = action.blockSignals(True)
                action.setChecked(should_check)
                action.blockSignals(old_state)

    def _empty_metric_values(self) -> Dict[str, str]:
        return {key: "n/a" for key in self.metric_order}

    def _compute_metric_values(self, time_values: np.ndarray, signal_values: np.ndarray, input_unit: str) -> Dict[str, str]:
        values = self._empty_metric_values()

        integral_value = float(np.trapezoid(signal_values, time_values))
        values["integral"] = f"{integral_value:.6g} {self._integral_unit_for(input_unit)}"
        values["min"] = f"{float(np.min(signal_values)):.6g}"
        values["max"] = f"{float(np.max(signal_values)):.6g}"
        values["mean"] = f"{float(np.mean(signal_values)):.6g}"
        values["rms"] = f"{float(np.sqrt(np.mean(np.square(signal_values)))):.6g}"
        values["std"] = f"{float(np.std(signal_values)):.6g}"
        return values

    def _reload_if_file_loaded(self) -> None:
        if self.current_file is not None:
            self._reload_current_file_data()

    def _warn_derived(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Derived channel", message)

    def _load_derived_config(self) -> None:
        self.derived_channels = {}
        if not self._derived_config_file.exists():
            return

        try:
            payload = json.loads(self._derived_config_file.read_text(encoding="utf-8"))
        except Exception:
            return

        entries = payload.get("channels", []) if isinstance(payload, dict) else []
        if not isinstance(entries, list):
            return

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            expression = str(entry.get("expression", "")).strip()
            unit = str(entry.get("unit", "")).strip() or "derived"
            if not name or not expression:
                continue
            self.derived_channels[name] = DerivedChannelDefinition(name=name, expression=expression, unit=unit)

    def _save_derived_config(self) -> None:
        self._derived_config_dir.mkdir(parents=True, exist_ok=True)
        entries = [
            {"name": item.name, "expression": item.expression, "unit": item.unit}
            for item in sorted(self.derived_channels.values(), key=lambda d: d.name.lower())
        ]
        payload = {"channels": entries}
        self._derived_config_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _refresh_derived_list(self) -> None:
        self.derived_list.clear()
        for definition in sorted(self.derived_channels.values(), key=lambda d: d.name.lower()):
            text = f"{definition.name} = {definition.expression} [{definition.unit}]"
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, definition.name)
            self.derived_list.addItem(item)

    def _on_derived_selected(self) -> None:
        item = self.derived_list.currentItem()
        if item is None:
            return

        name = item.data(QtCore.Qt.UserRole)
        if not isinstance(name, str):
            return

        definition = self.derived_channels.get(name)
        if definition is None:
            return

        self.derived_name_edit.setText(definition.name)
        self.derived_expr_edit.setText(definition.expression)
        self.derived_unit_edit.setText(definition.unit)

    def _apply_derived_channels_to_loaded_data(self) -> None:
        for definition in sorted(self.derived_channels.values(), key=lambda d: d.name.lower()):
            result = self._build_derived_channel_record(definition, show_errors=False)
            if result is not None:
                self.channels[definition.name] = result

    def _add_or_update_derived_channel(self) -> None:
        name = self.derived_name_edit.text().strip()
        expression = self.derived_expr_edit.text().strip()
        unit = self.derived_unit_edit.text().strip() or "derived"

        if not name:
            self._warn_derived("Please provide a channel name.")
            return
        if not expression:
            self._warn_derived("Please provide an expression.")
            return
        if name in self.base_channels:
            self._warn_derived("Name conflicts with a source channel. Pick a different derived name.")
            return

        definition = DerivedChannelDefinition(name=name, expression=expression, unit=unit)
        result = self._build_derived_channel_record(definition, show_errors=True)
        if result is None:
            return

        self.derived_channels[name] = definition
        self._save_derived_config()
        self._refresh_derived_list()
        self._reload_if_file_loaded()

    def _remove_selected_derived_channel(self) -> None:
        item = self.derived_list.currentItem()
        if item is None:
            return

        name = item.data(QtCore.Qt.UserRole)
        if not isinstance(name, str):
            return

        if name in self.derived_channels:
            self.derived_channels.pop(name, None)
            self._save_derived_config()
            self._refresh_derived_list()
        self._reload_if_file_loaded()

    def _build_derived_channel_record(
        self,
        definition: DerivedChannelDefinition,
        show_errors: bool,
    ) -> ChannelRecord | None:
        if ne is None:
            if show_errors:
                self._warn_derived("numexpr is not installed. Install with: pip install numexpr")
            return None

        try:
            parsed_expression = ne.NumExpr(definition.expression)
            variable_names = {str(name) for name in parsed_expression.input_names}
        except Exception as exc:
            if show_errors:
                self._warn_derived(f"Invalid expression syntax:\n{exc}")
            return None
        values_context: Dict[str, np.ndarray] = {}
        reference_time: np.ndarray | None = None

        for variable in variable_names:
            record = self.base_channels.get(variable)
            if record is None:
                if show_errors:
                    self._warn_derived(f"Unknown channel in expression: {variable}")
                return None

            if reference_time is None:
                reference_time = record.time
            else:
                if len(reference_time) != len(record.time) or not np.array_equal(reference_time, record.time):
                    if show_errors:
                        self._warn_derived("All channels in a derived expression must share the same time axis.")
                    return None

            values_context[variable] = record.values

        if reference_time is None:
            if show_errors:
                self._warn_derived("Expression must reference at least one channel.")
            return None

        try:
            evaluated = ne.evaluate(definition.expression, local_dict=values_context)
            derived_values = np.asarray(evaluated, dtype=float)
        except Exception as exc:
            if show_errors:
                self._warn_derived(f"Failed to evaluate expression:\n{exc}")
            return None

        if derived_values.shape != reference_time.shape:
            if show_errors:
                self._warn_derived("Expression did not produce a single vector aligned to the input time axis.")
            return None

        return ChannelRecord(
            name=definition.name,
            unit=definition.unit,
            signal_type="derived",
            samples=len(reference_time),
            time=reference_time,
            values=derived_values,
        )


class H5ViewerWindow(BaseWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BoardState H5 Viewer")
        self.resize(1450, 900)

        self.current_folder: Path | None = None

        self._build_tabs()
        self._build_menu()
        self._connect_signals()

        if DEFAULT_LOGS_ROOT.exists() and DEFAULT_LOGS_ROOT.is_dir():
            self._set_folder(DEFAULT_LOGS_ROOT)
        else:
            self.statusBar().showMessage(f"Default folder not found: {DEFAULT_LOGS_ROOT}")

    def _build_tabs(self) -> None:
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.setCentralWidget(self.tabs)

        self.folder_tab = QtWidgets.QWidget()
        folder_layout = QtWidgets.QVBoxLayout(self.folder_tab)
        folder_layout.setContentsMargins(10, 10, 10, 10)
        folder_layout.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        self.folder_label = QtWidgets.QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.open_folder_button = QtWidgets.QPushButton("Choose Folder...")
        self.refresh_folder_button = QtWidgets.QPushButton("Refresh")
        top_row.addWidget(self.folder_label, 1)
        top_row.addWidget(self.open_folder_button)
        top_row.addWidget(self.refresh_folder_button)
        folder_layout.addLayout(top_row)

        self.file_tree = QtWidgets.QTreeWidget()
        self.file_tree.setColumnCount(2)
        self.file_tree.setHeaderLabels(["File", "Size (KiB)"])
        self.file_tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.file_tree.setUniformRowHeights(True)
        self.file_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.file_tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        folder_layout.addWidget(self.file_tree, 1)

        open_row = QtWidgets.QHBoxLayout()
        self.open_selected_button = QtWidgets.QPushButton("Open Selected File In New Tab")
        open_row.addStretch(1)
        open_row.addWidget(self.open_selected_button)
        folder_layout.addLayout(open_row)

        self.tabs.addTab(self.folder_tab, "Files")

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        open_action = QtGui.QAction("Open H5...", self)
        open_action.setShortcut(QtGui.QKeySequence.Open)
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)

        open_folder_action = QtGui.QAction("Open Folder...", self)
        open_folder_action.triggered.connect(self._open_folder_dialog)
        file_menu.addAction(open_folder_action)

        close_action = QtGui.QAction("Close Current Tab", self)
        close_action.triggered.connect(self._close_current_tab)
        file_menu.addAction(close_action)

        file_menu.addSeparator()

        exit_action = QtGui.QAction("Exit", self)
        exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _connect_signals(self) -> None:
        self.open_folder_button.clicked.connect(self._open_folder_dialog)
        self.refresh_folder_button.clicked.connect(self._refresh_folder_list)
        self.open_selected_button.clicked.connect(self._open_selected_from_list)
        self.file_tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        self.tabs.tabCloseRequested.connect(self._close_tab_by_index)

    def _open_file_dialog(self) -> None:
        start_dir = self.current_folder if self.current_folder is not None else DEFAULT_LOGS_ROOT
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open BoardState H5 File",
            str(start_dir),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if not path:
            return
        self._open_file_in_tab(Path(path))

    def _open_folder_dialog(self) -> None:
        start_dir = self.current_folder if self.current_folder is not None else DEFAULT_LOGS_ROOT
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose H5 Folder", str(start_dir))
        if not folder:
            return
        self._set_folder(Path(folder))

    def _set_folder(self, folder: Path) -> None:
        self.current_folder = folder
        self.folder_label.setText(str(folder))
        self._refresh_folder_list()
        self.tabs.setCurrentIndex(0)

    def _refresh_folder_list(self) -> None:
        self.file_tree.clear()
        if self.current_folder is None:
            return

        if not self.current_folder.exists() or not self.current_folder.is_dir():
            self.statusBar().showMessage("Selected folder is not available")
            return

        files = self._discover_h5_files(self.current_folder)
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

        groups: Dict[str, list[Path]] = {}
        group_latest_mtime: Dict[str, float] = {}
        for path in files:
            rel_path = path.relative_to(self.current_folder)
            parts = rel_path.parts
            if len(parts) <= 1:
                group_name = "(root)"
            else:
                group_name = parts[0]
            groups.setdefault(group_name, []).append(path)
            group_latest_mtime[group_name] = max(group_latest_mtime.get(group_name, 0.0), path.stat().st_mtime)

        # Sort groups by latest file timestamp (newest first).
        sorted_groups = sorted(groups.keys(), key=lambda name: group_latest_mtime.get(name, 0.0), reverse=True)
        for idx, group_name in enumerate(sorted_groups):
            group_item = QtWidgets.QTreeWidgetItem([group_name, ""])
            group_item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            group_item.setData(0, QtCore.Qt.UserRole, None)

            file_paths = sorted(groups[group_name], key=lambda p: p.stat().st_mtime, reverse=True)
            for path in file_paths:
                rel_path = path.relative_to(self.current_folder)
                if group_name == "(root)":
                    display_name = str(rel_path)
                else:
                    display_name = str(Path(*rel_path.parts[1:]))
                size_kib = int(path.stat().st_size / 1024)
                child = QtWidgets.QTreeWidgetItem([display_name, str(size_kib)])
                child.setData(0, QtCore.Qt.UserRole, str(path))
                child.setToolTip(0, str(rel_path))
                group_item.addChild(child)

            self.file_tree.addTopLevelItem(group_item)
            if idx == 0:
                group_item.setExpanded(True)

        self.statusBar().showMessage(
            f"Found {len(files)} H5 files in {self.current_folder} (including two-level subfolders)"
        )

    def _on_tree_item_double_clicked(self, item, _column: int) -> None:
        value = item.data(0, QtCore.Qt.UserRole)
        if isinstance(value, str):
            self._open_file_in_tab(Path(value))
            return

        item.setExpanded(not item.isExpanded())

    def _discover_h5_files(self, root: Path) -> list[Path]:
        files: list[Path] = []
        files.extend(root.glob("*.h5"))
        files.extend(root.glob("*.hdf5"))

        # Include direct child directories and their direct children (two levels deep).
        for child in root.iterdir():
            if not child.is_dir():
                continue
            files.extend(child.glob("*.h5"))
            files.extend(child.glob("*.hdf5"))

            for grandchild in child.iterdir():
                if not grandchild.is_dir():
                    continue
                files.extend(grandchild.glob("*.h5"))
                files.extend(grandchild.glob("*.hdf5"))

        return files

    def _open_selected_from_list(self) -> None:
        item = self.file_tree.currentItem()
        if item is None:
            return
        value = item.data(0, QtCore.Qt.UserRole)
        if not isinstance(value, str):
            return
        self._open_file_in_tab(Path(value))

    def _open_file_in_tab(self, path: Path) -> None:
        existing_index = self._find_file_tab(path)
        if existing_index is not None:
            self.tabs.setCurrentIndex(existing_index)
            return

        tab = H5FileTab()
        tab.load_file(path)
        tab_index = self.tabs.addTab(tab, path.name)
        self.tabs.setTabToolTip(tab_index, str(path))
        self.tabs.setCurrentIndex(tab_index)
        self.statusBar().showMessage(f"Opened {path}")

    def _find_file_tab(self, path: Path) -> int | None:
        for i in range(1, self.tabs.count()):
            widget = self.tabs.widget(i)
            if isinstance(widget, H5FileTab) and widget.current_file == path:
                return i
        return None

    def _close_current_tab(self) -> None:
        self._close_tab_by_index(self.tabs.currentIndex())

    def _close_tab_by_index(self, index: int) -> None:
        if index <= 0:
            return
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        if widget is not None:
            widget.deleteLater()


def main() -> int:
    parser = argparse.ArgumentParser(description="BoardStateLogger H5 interactive viewer")
    parser.add_argument("h5", nargs="?", help="Optional path to .h5 file")
    args = parser.parse_args()

    if QtWidgets is None or pg is None:
        print("This viewer requires PySide6 and pyqtgraph.")
        print("Install with: pip install pyside6 pyqtgraph")
        return 1

    app = QtWidgets.QApplication([])
    pg.setConfigOptions(antialias=False, foreground="#d8dee9", background="#111217")

    window = H5ViewerWindow()
    window.show()

    if args.h5:
        input_path = Path(args.h5)
        if input_path.exists():
            window._open_file_in_tab(input_path)
        else:
            QtWidgets.QMessageBox.warning(window, "Missing file", f"File does not exist:\n{input_path}")

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
