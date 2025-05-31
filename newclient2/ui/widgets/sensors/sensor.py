from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import QTimer, QDateTime
import pyqtgraph as pg
import numpy as np
from collections import deque

# Global variables for sensor colors and point counter
colors = ['y', 'r', 'c', 'b', 'm', 'y', 'k']
pt_number = 0

class SensorControllerWidget(QWidget):
    """Base class for all sensor controller widgets"""
    def __init__(self, config, sensor_default_data, board_name, sensor_name, sensor_type, udpmanager, testapp):
        global pt_number
        super().__init__()
        self.config = config
        self.sensor_default_data = sensor_default_data
        self.board_name = board_name
        self.sensor_name = sensor_name
        self.sensor_type = sensor_type  # "pts", "tcs", etc.
        self.udpmanager = udpmanager
        self.testapp = testapp
        
        # Define standard widths
        self.LABEL_WIDTH = 100
        
        # Create common layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Create common UI elements
        self.name_label = QLabel(f"{self.sensor_type}: {self.sensor_name}")
        self.name_label.setFixedWidth(self.LABEL_WIDTH + 20)
        self.layout.addWidget(self.name_label)

        self.values = {}
        self.MAX_HISTORY_SIZE = 10000
        
        for value_name, _ in self.sensor_default_data.items():
            self.values[value_name] = {"value": "No data"}
            self.values[value_name]["label"] = QLabel(f"{value_name}: No data")
            self.values[value_name]["label"].setStyleSheet("font-size: 40px;")
            self.values[value_name]["label"].setFixedWidth(self.LABEL_WIDTH + 100)
            self.values[value_name]["history"] = deque(maxlen=self.MAX_HISTORY_SIZE)
            
            # Preallocate arrays for improved performance
            self.values[value_name]["time_array"] = np.zeros(self.MAX_HISTORY_SIZE, dtype=np.float64)
            self.values[value_name]["value_array"] = np.zeros(self.MAX_HISTORY_SIZE, dtype=np.float64)
            self.values[value_name]["array_size"] = 0
            
            self.vertical_layout = QVBoxLayout()
            self.vertical_layout.addWidget(self.values[value_name]["label"])

            self.values[value_name]["plot"] = pg.PlotWidget(title=f"{self.sensor_type} {self.sensor_name} {value_name}")
            self.values[value_name]["curve"] = self.values[value_name]["plot"].plot(pen=pg.mkPen(colors[pt_number], width=2))
            if sensor_type == "pts":
                self.values[value_name]["pressurecurve"] = self.testapp.pressureplot.plot(
                    pen=pg.mkPen(colors[pt_number], width=2), 
                    name=sensor_name
                )
            pt_number += 1
            self.vertical_layout.addWidget(self.values[value_name]["plot"])

            self.layout.addLayout(self.vertical_layout)
            
            self.value_timer = QTimer(self)
            self.value_timer.timeout.connect(lambda vn=value_name: self.update_history(vn))
            self.value_timer.start(50)
        self.layout.addStretch()

    def update_history(self, value_name):
        value_history = self.values[value_name]["history"]
        time_array = self.values[value_name]["time_array"]
        value_array = self.values[value_name]["value_array"]

        if not value_history:
            return
        
        # Clean up old data points (older than 10 seconds)
        current_time = QDateTime.currentDateTime()
        cutoff_time = current_time.addSecs(-120)
        while value_history and value_history[0][0] < cutoff_time:
            value_history.popleft()
        
        if not value_history:   
            # Reset array size if history is empty
            self.values[value_name]["array_size"] = 0
            return
        
        # Copy data to preallocated arrays
        size = len(value_history)
        self.values[value_name]["array_size"] = size
        
        for i, (t, v) in enumerate(value_history):
            time_array[i] = t.toMSecsSinceEpoch()
            value_array[i] = v
        
        # Use preallocated arrays for plotting
        self.values[value_name]["curve"].setData(
            (time_array[max(0, size-200):size] - time_array[size-1]) / 1000, 
            value_array[max(0, size-200):size]
        )
        if self.sensor_type == "pts":
            self.values[value_name]["pressurecurve"].setData(
                (time_array[:size] - time_array[0]) / 1000, 
                value_array[:size]
            )

    def update_states(self, states):
        for value_name, value_dict in self.values.items():
            try:
                value = states[self.sensor_type][self.sensor_name][value_name]
                value_label = value_dict["label"]
                
                # Apply calibration if available
                if "adc" in self.config[self.sensor_type][self.sensor_name]:
                    gain = self.config[self.sensor_type][self.sensor_name]["gain"]
                    offset = self.config[self.sensor_type][self.sensor_name]["offset"]
                    value = (value - offset) * gain
                
                # Update the label
                value_label.setText(f"{value:.2f}")
                
                # Add to the history deque
                if value is None:
                    continue
                value_dict["history"].append((QDateTime.currentDateTime(), float(value)))
                
            except KeyError:
                value_label.setText("Value: No data")
