import socket
import sys
import json
import signal
import ctypes
from pathlib import Path

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QTextEdit, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QSlider, QGroupBox, QGridLayout, QLineEdit, QTabWidget)
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import QTimer, Qt
from PySide6.QtNetwork import QUdpSocket, QHostAddress
from PySide6.QtCore import QTimer, QByteArray, Slot
from PySide6.QtCore import Slot, QObject, Signal
from PySide6.QtCore import QDateTime
import pyqtgraph as pg
import numpy as np

PLOT_SECONDS = 30
PLOT_UPDATE_INTERVAL_MS = 100
REQUEST_STATES_INTERVAL_MS = 100
BACKEND_META_INTERVAL_MS = 100
SENSOR_PLOT_MIN_HEIGHT_PX = 90
SENSOR_PLOT_MAX_HEIGHT_PX = 220
LABEL_UPDATE_INTERVAL_MS = 100

class UDPManager(QObject):
    dataReceived = Signal(str)
    hardwareJsonStatusChanged = Signal(bool)
    
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = QUdpSocket(self)
        self.socket.readyRead.connect(self.process_datagrams)
        
        self.hardware_json_received = False
        
        QTimer.singleShot(0, self.request_hardware_json)
        
        # Timer to check if hardware JSON was received
        self.hardware_json_timer = QTimer(self)
        self.hardware_json_timer.timeout.connect(self.check_hardware_json)
        self.hardware_json_timer.start(1000)  # Check every 1 second    
    @Slot()
    def check_hardware_json(self):
        if not self.hardware_json_received:
            self.request_hardware_json()
    
    def set_hardware_json_received(self, received=True):
        if received != self.hardware_json_received:
            self.hardware_json_received = received
            self.hardwareJsonStatusChanged.emit(received)
            if received:
                # Stop checking once we've received it
                self.hardware_json_timer.stop()
                self.transmit_timer = QTimer(self)
                self.transmit_timer.timeout.connect(self.request_states)
                self.transmit_timer.start(REQUEST_STATES_INTERVAL_MS)
    
    @Slot()
    def request_states(self):
        #print("Requesting boards states from backend")
        data = json.dumps({"command":"get boards states", "data":{}})
        self.send(data)
        # ALso request desired states
        data = json.dumps({"command":"get boards desired states", "data":{}})
        self.send(data)

    @Slot()
    def request_hardware_json(self):
        data = json.dumps({"command":"get hardware json", "data":{}})
        self.send(data)

    @Slot()
    def process_datagrams(self):
        while self.socket.hasPendingDatagrams():
            datagram_size = self.socket.pendingDatagramSize()
            data, host, port = self.socket.readDatagram(datagram_size)
            self.dataReceived.emit(bytes(data.data()).decode())

    @Slot()
    def send(self, data):
        self.socket.writeDatagram(QByteArray(data.encode()), QHostAddress(self.host), self.port)

    @Slot()
    def send_actuator_command(self, board_name, actuator_types, actuator_name, command, value):
        message = {
            "command": "update desired state",
            "data":{
                "board_name": board_name,
                "message": {
                    actuator_types: {
                        actuator_name: {
                            command: value
                        }
                    }
                }
            }
        }
        self.send(json.dumps(message))

class ActuatorControllerWidget(QWidget):
    """Base class for all actuator controller widgets"""
    def __init__(self, config, board_name, actuator_name, actuator_type, udpmanager):
        super().__init__()
        self.config = config
        self.board_name = board_name
        self.actuator_name = actuator_name
        self.actuator_type = actuator_type  # "servos", "solenoids", etc.
        self.udpmanager = udpmanager
        
        # Define standard widths
        self.LABEL_WIDTH = 100
        self.BUTTON_WIDTH = 50
        self.INPUT_WIDTH = 40
        
        # Create common layout
        self.row_layout = QHBoxLayout()
        self.row_layout.setContentsMargins(2, 1, 2, 1)
        self.row_layout.setSpacing(5)
        self.setLayout(self.row_layout)
        
        # Create common UI elements
        self.name_label = QLabel(f"{self.get_actuator_type_name()}: {self.actuator_name}")
        self.name_label.setMinimumWidth(self.LABEL_WIDTH)
        self.row_layout.addWidget(self.name_label)
        
        # Armed state is common to most actuators
        self.armed_label = QLabel(f"Armed: No data")
        self.armed_label.setMinimumWidth(self.LABEL_WIDTH)
        self.row_layout.addWidget(self.armed_label)
        
        # Common arm/disarm buttons
        self.arm_button = QPushButton("Arm")
        self.arm_button.setMinimumWidth(self.BUTTON_WIDTH)
        self.arm_button.clicked.connect(lambda: self.send_command("armed", True))
        
        self.disarm_button = QPushButton("Disarm")
        self.disarm_button.setMinimumWidth(self.BUTTON_WIDTH)
        self.disarm_button.clicked.connect(lambda: self.send_command("armed", False))
        
        self.row_layout.addWidget(self.arm_button)
        self.row_layout.addWidget(self.disarm_button)
        
        # Add specific UI elements
        self.setup_specific_ui()
        self.row_layout.addStretch()
    
    def send_command(self, command, value):
        """Send command to the actuator"""
        self.udpmanager.send_actuator_command(
            self.board_name, 
            self.actuator_type, 
            self.actuator_name, 
            command, 
            value
        )
    
    def update_common_states(self, states):
        """Update common states like armed status"""
        try:
            armed = states[self.actuator_type][self.actuator_name]["armed"]
            self.armed_label.setText(f"Armed: {armed}")
        except KeyError:
            self.armed_label.setText("Armed: No data")
    
    def update_states(self, states):
        """Update widget with current states"""
        self.update_common_states(states)
        self.update_specific_states(states)

    def update_desired_states(self, desired_states):
        """Update widget with desired states"""
        self.update_specific_desired_states(desired_states)

    def setup_specific_ui(self):
        """Set up actuator-specific UI elements. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement setup_specific_ui()")
    
    def update_specific_states(self, states):
        """Update actuator-specific states. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement update_specific_states()")
    
    def get_actuator_type_name(self):
        """Return human-readable actuator type name. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement get_actuator_type_name()")

    def update_specific_desired_states(self, desired_states):
        """Update actuator-specific desired states. Override in subclasses. Does nothing if not implemented"""
        pass

class ServoControllerWidget(ActuatorControllerWidget):
    def __init__(self, config, board_name, servo_name, udpmanager):
        super().__init__(config, board_name, servo_name, "servos", udpmanager)
    
    def get_actuator_type_name(self):
        return "Servo"
    
    def setup_specific_ui(self):
        # Servo-specific UI elements
        self.angle_label = QLabel("Angle: No data")
        self.angle_label.setMinimumWidth(self.LABEL_WIDTH)
        self.row_layout.addWidget(self.angle_label)

        self.desired_angle_label = QLabel("Desired: No data")
        self.desired_angle_label.setMinimumWidth(self.LABEL_WIDTH)
        self.row_layout.addWidget(self.desired_angle_label)
        
        self.manual_angle = QLineEdit(self)
        self.manual_angle.setMinimumWidth(self.INPUT_WIDTH)
        self.manual_angle.setPlaceholderText("0")
        
        self.manual_angle_button = QPushButton("Set")
        self.manual_angle_button.setMinimumWidth(self.BUTTON_WIDTH)
        self.manual_angle_button.clicked.connect(self.set_manual_angle)

        self.zero_angle_button = QPushButton("Set 0")
        self.zero_angle_button.setMinimumWidth(self.BUTTON_WIDTH)
        self.zero_angle_button.clicked.connect(self.set_zero_angle)
        
        self.row_layout.addWidget(self.manual_angle)
        self.row_layout.addWidget(self.manual_angle_button)
        self.row_layout.addWidget(self.zero_angle_button)
    
    def update_specific_states(self, states):
        try:
            angle = states[self.actuator_type][self.actuator_name]["angle"]
            self.angle_label.setText(f"Angle: {angle}")
        except KeyError:
            self.angle_label.setText("Angle: No data")
    
    def update_specific_desired_states(self, desired_states):
        try:
            desired_angle = desired_states[self.actuator_type][self.actuator_name]["angle"]
            self.desired_angle_label.setText(f"Desired: {desired_angle}")
        except KeyError:
            self.desired_angle_label.setText("Desired: No data")

    def set_manual_angle(self):
        try:
            angle = int(self.manual_angle.text())
            # if angle < 0 or angle > 180:
                # raise ValueError("Angle must be between 0 and 180")
            self.send_command("angle", angle)
        except ValueError as e:
            print(f"Invalid angle: {e}")

    def set_zero_angle(self):
        self.send_command("angle", 0)
        self.manual_angle.setText("0")

class SolenoidControllerWidget(ActuatorControllerWidget):
    def __init__(self, config, board_name, actuator_name, udpmanager):
        super().__init__(config, board_name, actuator_name, "solenoids", udpmanager)
    
    def get_actuator_type_name(self):
        return "Solenoid"
    
    def setup_specific_ui(self):
        # Solenoid-specific UI elements
        self.powered_label = QLabel("Powered: No data")
        self.powered_label.setMinimumWidth(self.LABEL_WIDTH)
        self.row_layout.addWidget(self.powered_label)
        
        self.poweron_button = QPushButton("Power On")
        self.poweron_button.setMinimumWidth(self.BUTTON_WIDTH)
        self.poweron_button.clicked.connect(lambda: self.send_command("powered", True))
        
        self.poweroff_button = QPushButton("Power Off")
        self.poweroff_button.setMinimumWidth(self.BUTTON_WIDTH)
        self.poweroff_button.clicked.connect(lambda: self.send_command("powered", False))
        
        self.row_layout.addWidget(self.poweron_button)
        self.row_layout.addWidget(self.poweroff_button)
    
    def update_specific_states(self, states):
        #print(f"Updating states for {self.actuator_name}: {states}")
        try:
            powered = states[self.actuator_type][self.actuator_name]["powered"]
            self.powered_label.setText(f"Powered: {powered}")
        except KeyError:
            self.powered_label.setText("Powered: No data")

class PyroControllerWidget(ActuatorControllerWidget):
    def __init__(self, config, board_name, actuator_name, udpmanager):
        super().__init__(config, board_name, actuator_name, "pyros", udpmanager)
    
    def get_actuator_type_name(self):
        return "Pyro"
    
    def setup_specific_ui(self):
        # Pyro-specific UI elements
        self.powered_label = QLabel("Powered: No data")
        self.powered_label.setMinimumWidth(self.LABEL_WIDTH)
        self.row_layout.addWidget(self.powered_label)
        
        self.poweron_button = QPushButton("Power On")
        self.poweron_button.setMinimumWidth(self.BUTTON_WIDTH)
        self.poweron_button.clicked.connect(lambda: self.send_command("powered", True))
        
        self.poweroff_button = QPushButton("Power Off")
        self.poweroff_button.setMinimumWidth(self.BUTTON_WIDTH)
        self.poweroff_button.clicked.connect(lambda: self.send_command("powered", False))
        
        self.row_layout.addWidget(self.poweron_button)
        self.row_layout.addWidget(self.poweroff_button)
    
    def update_specific_states(self, states):
        #print(f"Updating states for {self.actuator_name}: {states}")
        try:
            powered = states[self.actuator_type][self.actuator_name]["powered"]
            self.powered_label.setText(f"Powered: {powered}")
        except KeyError:
            self.powered_label.setText("Powered: No data")
# High-contrast colors for dark pyqtgraph backgrounds.
colors = ['y', 'r', 'c', 'b', 'm', 'g', 'w']
pt_number = 0
UNIT_COLOR_PALETTE = [
    '#f6e05e',  # yellow
    '#00e5ff',  # cyan
    '#ff5f57',  # red
    '#4cff7f',  # green
    '#c084fc',  # violet
    '#ffffff',  # white
    '#ffa94d',  # orange
    '#74c0fc',  # light blue
]

class SensorControllerWidget(QWidget):
    """"Base class for all sensor controller widgets"""
    def __init__(self, config, sensor_default_data, board_name, sensor_name, sensor_type, udpmanager, testapp, unit, unit_color):
        super().__init__()
        self.config = config
        self.sensor_default_data = sensor_default_data
        self.board_name = board_name
        self.sensor_name = sensor_name
        self.sensor_type = sensor_type  # "pts", "tcs", etc.
        self.unit = unit
        self.unit_color = unit_color
        self.udpmanager = udpmanager
        self.testapp = testapp
        
        # Define standard widths
        self.LABEL_WIDTH = 100
        
        # Create common layout
        self.sensor_layout = QVBoxLayout()
        self.sensor_layout.setContentsMargins(2, 2, 2, 2)
        self.sensor_layout.setSpacing(2)
        self.setLayout(self.sensor_layout)

        self.values = {}
        self.plot_widgets = []
        # Keep enough points for plotting while limiting memory and copy cost.
        self.MAX_HISTORY_SIZE = max(1000, int(PLOT_SECONDS * 120))
        
        for value_name, _ in self.sensor_default_data.items():
            label_prefix = f"{self.sensor_name}"
            self.values[value_name] = {"value": "No data"}
            self.values[value_name]["label"] = QLabel(f"{label_prefix}: No data")
            self.values[value_name]["label"].setStyleSheet("font-size: 20px;")
            self.values[value_name]["label"].setMinimumWidth(self.LABEL_WIDTH + 100)
            self.values[value_name]["label"].setFixedHeight(30)
            self.values[value_name]["label_prefix"] = label_prefix
            self.values[value_name]["last_label_ms"] = 0.0

            # Fixed-size ring buffer for plotting history
            self.values[value_name]["time_array"] = np.zeros(self.MAX_HISTORY_SIZE, dtype=np.float64)
            self.values[value_name]["value_array"] = np.zeros(self.MAX_HISTORY_SIZE, dtype=np.float64)
            self.values[value_name]["head"] = 0
            self.values[value_name]["count"] = 0

            self.values[value_name]["plot"] = pg.PlotWidget()
            self.values[value_name]["plot"].setFixedHeight(SENSOR_PLOT_MIN_HEIGHT_PX)
            self.values[value_name]["plot"].setMouseEnabled(x=True, y=True)
            self.values[value_name]["plot"].showGrid(x=True, y=True, alpha=0.15)
            self._install_auto_axis_action(self.values[value_name]["plot"])
            self.values[value_name]["curve"] = self.values[value_name]["plot"].plot(pen=pg.mkPen(self.unit_color, width=2))
            self.plot_widgets.append(self.values[value_name]["plot"])
            self.values[value_name]["curve"].setClipToView(True)
            self.values[value_name]["curve"].setDownsampling(auto=True, method="peak")

            self.vertical_layout = QVBoxLayout()
            self.vertical_layout.setContentsMargins(0, 0, 0, 0)
            self.vertical_layout.setSpacing(3)
            self.vertical_layout.addWidget(self.values[value_name]["plot"])
            self.vertical_layout.addWidget(self.values[value_name]["label"])


            self.sensor_layout.addLayout(self.vertical_layout)

        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update_all_histories)
        self.plot_timer.start(PLOT_UPDATE_INTERVAL_MS)
        self.sensor_layout.addStretch()

    def cleanup(self):
        if hasattr(self, "plot_timer") and self.plot_timer is not None:
            self.plot_timer.stop()

    def set_plot_height(self, plot_height):
        for plot_widget in self.plot_widgets:
            plot_widget.setFixedHeight(plot_height)

    def _install_auto_axis_action(self, plot_widget):
        plot_item = plot_widget.getPlotItem()
        view_box = plot_item.getViewBox()
        menu = view_box.menu if view_box.menu is not None else view_box.getMenu()
        if menu is None:
            return

        auto_axis_action = QAction("Auto Axis", plot_widget)
        auto_axis_action.triggered.connect(
            lambda _checked=False, item=plot_item: self._enable_auto_axis(item)
        )
        menu.addAction(auto_axis_action)

    def _enable_auto_axis(self, plot_item):
        plot_item.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)
        plot_item.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

    def update_all_histories(self):
        for value_name in self.values:
            self.update_history(value_name)

    def update_history(self, value_name):
        value_dict = self.values[value_name]
        count = value_dict["count"]
        if count == 0:
            return

        time_array = value_dict["time_array"]
        value_array = value_dict["value_array"]
        head = value_dict["head"]

        if count < self.MAX_HISTORY_SIZE:
            times = time_array[:count]
            values = value_array[:count]
        else:
            times = np.concatenate((time_array[head:], time_array[:head]))
            values = np.concatenate((value_array[head:], value_array[:head]))

        start_time = times[-1] - (PLOT_SECONDS * 1000)
        start_idx = np.searchsorted(times, start_time, side="left")
        plot_times = (times[start_idx:] - times[-1]) / 1000.0
        plot_values = values[start_idx:]

        value_dict["curve"].setData(
            plot_times,
            plot_values
        )

    def update_states(self, states):
        timestamp_ms = float(QDateTime.currentMSecsSinceEpoch())
        for value_name, value_dict in self.values.items():
            value_label = value_dict["label"]
            try:
                raw_value = states[self.sensor_type][self.sensor_name][value_name]
                value = raw_value
                
                
                # Apply calibration if available

                
                # Update the label
                if value is None:
                    if timestamp_ms - value_dict["last_label_ms"] >= LABEL_UPDATE_INTERVAL_MS:
                        value_label.setText("Value: No data")
                        value_dict["last_label_ms"] = timestamp_ms
                else:
                    if "adc" in self.config[self.sensor_type][self.sensor_name]:
                        gain = self.config[self.sensor_type][self.sensor_name]["gain"]
                        offset = self.config[self.sensor_type][self.sensor_name]["offset"]
                        value = (value - offset) * gain
                        if isinstance(raw_value, (int, float)) and float(raw_value).is_integer():
                            raw_display = f"{int(raw_value)}"
                        else:
                            raw_display = f"{raw_value:.0f}"
                        if timestamp_ms - value_dict["last_label_ms"] >= LABEL_UPDATE_INTERVAL_MS:
                            value_label.setText(
                                f"<span style='font-size: 14px; font-weight: 600; color: #e8e8e8;'>{value_dict['label_prefix']} </span>"
                                f"<span style='font-size: 20px; color: {self.unit_color};'>{value:.3f}</span> "
                                f"<span style='font-size: 14px; color: {self.unit_color};'>{self.unit}</span> "
                                f"<span style='font-size: 13px; color: #8f8f8f;'>{raw_display}mV</span>"
                            )
                            value_dict["last_label_ms"] = timestamp_ms
                    else:
                        if timestamp_ms - value_dict["last_label_ms"] >= LABEL_UPDATE_INTERVAL_MS:
                            value_label.setText(
                                f"<span style='font-size: 14px; font-weight: 600; color: #e8e8e8;'>{value_dict['label_prefix']} </span>"
                                f"<span style='font-size: 20px; color: {self.unit_color};'>{value:.2f}</span> "
                                f"<span style='font-size: 14px; color: {self.unit_color};'>{self.unit}</span>"
                            )
                            value_dict["last_label_ms"] = timestamp_ms
                
                # Add to the history deque
                if value is None:
                    continue

                idx = value_dict["head"]
                value_dict["time_array"][idx] = timestamp_ms
                value_dict["value_array"][idx] = float(value)
                value_dict["head"] = (idx + 1) % self.MAX_HISTORY_SIZE
                if value_dict["count"] < self.MAX_HISTORY_SIZE:
                    value_dict["count"] += 1
                
            except KeyError:
                value_label.setText("Value: No data")

class PropertyTestApp(QMainWindow):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.hardware_json = None
        self.state_defaults: dict = {}
        self.hardware_types: list[str] = []
        self.actuator_list = []
        self.sensor_list = []
        self.actuator_group_widget = None
        self.sensor_group_widget = None
        self.unit_color_map = {}
        self.sensor_plot_row_count = 0
        self.sensor_unit_group_count = 0


        self.setWindowTitle("Propulsion Test Client")
        self.setWindowIcon(QIcon("assets/propclient.ico"))

        self.main_tab = QWidget()
        self.main_layout = QHBoxLayout(self.main_tab)
        self.setCentralWidget(self.main_tab)  


        #container = QWidget()
        #container.setLayout(self.main_layout)


        self.data_waiting_label = QLabel("Waiting for data...")

        self.last_update_time = None
        self.last_state_update_label = QLabel("Last state update: Never")
        self.last_state_update_label.setMinimumWidth(200)
        self.backend_state_timer = QTimer(self)
        self.backend_state_timer.timeout.connect(self.backend_state_coroutine)
        self.backend_state_timer.start(BACKEND_META_INTERVAL_MS)
        self.command_response = QTextEdit(self)
        self.command_response.setReadOnly(True)
        self.command_response.setMinimumHeight(170)
        self.command_response.setPlaceholderText("Command/response output")

        self.preset_commands_layout = QVBoxLayout()
        self.command_buttons_layout = QHBoxLayout()
        self.command_buttons_right_layout = QVBoxLayout()

        self.abort_button = QPushButton("Abort")
        self.abort_button.setStyleSheet("font-size: 60px;")
        self.abort_button.setMinimumHeight(180)
        self.abort_button.setMinimumWidth(170)
        self.abort_button.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"abort engine", "data":{}})))
        self.command_buttons_layout.addWidget(self.abort_button)

        self.hotfire_button = QPushButton("Hotfire")
        self.hotfire_button.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"start hotfire sequence", "data":{}})))
        self.command_buttons_right_layout.addWidget(self.hotfire_button)

        self.return_from_idle_button = QPushButton("Return to Idle")
        self.return_from_idle_button.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"return to idle", "data":{}})))
        self.command_buttons_right_layout.addWidget(self.return_from_idle_button)

        self.reload_hardware_json = QPushButton("Reload Hardware JSON")
        self.reload_hardware_json.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"reload hardware json", "data":{}})))
        self.command_buttons_right_layout.addWidget(self.reload_hardware_json)

        self.get_new_hardware_json = QPushButton("Reload UI")
        self.get_new_hardware_json.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"get hardware json", "data":{}})))
        self.command_buttons_right_layout.addWidget(self.get_new_hardware_json)

        self.command_buttons_right_layout.addStretch()
        self.command_buttons_layout.addLayout(self.command_buttons_right_layout)
        self.preset_commands_layout.addLayout(self.command_buttons_layout)



        self.hotfiretime_str = "Request timer didn't load"
        self.hotfiretime_label = QLabel(self.hotfiretime_str)
        self.hotfiretime_label.setStyleSheet("font-size: 60px;")
        self.preset_commands_layout.addWidget(self.hotfiretime_label)

        self.datetime_str = "Request timer didn't load"
        self.datetime_label = QLabel(f"Backend Time: {self.datetime_str}")

        self.statemachine_str = "Request timer didn't load"
        self.statemachine_label = QLabel("Machine state: {self.statemachine_str}")

        self.preset_commands_layout.addWidget(self.statemachine_label)
        self.preset_commands_layout.addWidget(self.datetime_label)
        self.preset_commands_layout.addWidget(self.last_state_update_label)
        self.preset_commands_layout.addWidget(self.command_response)
        

        self.preset_commands_layout.addStretch()

        self.control_area = QVBoxLayout()
        self.sensor_area = QVBoxLayout()
        self.control_area.setSpacing(2)
        self.sensor_area.setSpacing(4)

        self.left_panel_layout = QVBoxLayout()
        self.left_panel_layout.setSpacing(8)
        self.left_panel_layout.addLayout(self.preset_commands_layout)
        self.left_panel_layout.addLayout(self.control_area)

        # Add two dark red server control buttons
        self.reboot_button = QPushButton("Reboot PI")
        self.reboot_button.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"reboot pi", "data":{}})))
        self.reboot_button.setStyleSheet("background-color: darkred; color: white;")

        self.shutdown_button = QPushButton("Shutdown PI")
        self.shutdown_button.setStyleSheet("background-color: darkred; color: white;")
        self.shutdown_button.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"shutdown pi", "data":{}})))

        self.pi_buttons_layout = QHBoxLayout()
        self.pi_buttons_layout.addWidget(self.reboot_button)
        self.pi_buttons_layout.addWidget(self.shutdown_button)
        self.left_panel_layout.addLayout(self.pi_buttons_layout)
        

        self.right_panel_layout = QVBoxLayout()
        self.right_panel_layout.setSpacing(6)
        self.right_panel_layout.addWidget(self.data_waiting_label)
        self.right_panel_layout.addLayout(self.sensor_area)

        self.main_layout.addLayout(self.left_panel_layout)
        self.main_layout.addLayout(self.right_panel_layout)

        self.main_layout.setStretch(0, 3)
        self.main_layout.setStretch(1, 7)
        self.right_panel_layout.setStretch(0, 0)
        self.right_panel_layout.setStretch(1, 10)
        
        self.udp_manager = UDPManager(self.host, self.port)
        self.udp_manager.dataReceived.connect(self.handle_data_received)

        self.commands_responses = {
            "get hardware json": self.hardware_json_received,
            "reload hardware json": self.reload_hardware_json_received,
            "get boards states": self.boards_states_received,
            "get boards desired states": self.boards_desired_states_received,
            "get state": self.machinestate_received,
            "get time": self.time_received
        }

    def _clear_dynamic_ui(self):
        for actuator in self.actuator_list:
            actuator.setParent(None)
            actuator.deleteLater()
        self.actuator_list.clear()

        for sensor in self.sensor_list:
            try:
                sensor.cleanup()
            except Exception:
                pass
            sensor.setParent(None)
            sensor.deleteLater()
        self.sensor_list.clear()
        self.sensor_plot_row_count = 0
        self.sensor_unit_group_count = 0

        if self.actuator_group_widget is not None:
            self.control_area.removeWidget(self.actuator_group_widget)
            self.actuator_group_widget.setParent(None)
            self.actuator_group_widget.deleteLater()
            self.actuator_group_widget = None

        if self.sensor_group_widget is not None:
            self.sensor_area.removeWidget(self.sensor_group_widget)
            self.sensor_group_widget.setParent(None)
            self.sensor_group_widget.deleteLater()
            self.sensor_group_widget = None

    def _unit_color(self, unit):
        unit_key = unit or "value"
        if unit_key not in self.unit_color_map:
            idx = len(self.unit_color_map) % len(UNIT_COLOR_PALETTE)
            self.unit_color_map[unit_key] = UNIT_COLOR_PALETTE[idx]
        return self.unit_color_map[unit_key]

    def _build_dynamic_ui_from_hardware(self):
        if self.hardware_json is None:
            return
        hardware_json = self.hardware_json

        actuator_group = QGroupBox("Actuators")
        actuator_layout = QVBoxLayout()
        actuator_layout.setContentsMargins(4, 4, 4, 4)
        actuator_layout.setSpacing(2)

        default_actuator_types = {
            "servos": ServoControllerWidget,
            "solenoids": SolenoidControllerWidget,
            "pyros": PyroControllerWidget
        }

        for board_name, board_config in hardware_json["boards"].items():
            if board_config.get("is_actuator", False):
                for actuator_type, actuator_widget_class in default_actuator_types.items():
                    if actuator_type in board_config:
                        for actuator_name, _ in board_config[actuator_type].items():
                            actuator_controller_widget = actuator_widget_class(board_config, board_name, actuator_name, self.udp_manager)
                            actuator_layout.addWidget(actuator_controller_widget)
                            self.actuator_list.append(actuator_controller_widget)
        actuator_layout.addStretch()
        actuator_group.setLayout(actuator_layout)
        self.control_area.addWidget(actuator_group)
        self.actuator_group_widget = actuator_group

        sensor_group = QGroupBox("Sensors")
        sensor_mainlayout = QVBoxLayout()
        sensor_mainlayout.setContentsMargins(4, 4, 4, 4)
        sensor_mainlayout.setSpacing(6)
        columns = 3

        sensors_by_unit = {}
        for board_name, board_config in hardware_json["boards"].items():
            if not board_config.get("is_actuator", False):
                for sensor_type, sensor_default_data in self.state_defaults.items():
                    if sensor_type in board_config and isinstance(board_config[sensor_type], dict):
                        for sensor_name, sensor_cfg in board_config[sensor_type].items():
                            unit = ""
                            if isinstance(sensor_cfg, dict):
                                unit = str(sensor_cfg.get("unit", ""))
                            unit = unit or "value"
                            unit_color = self._unit_color(unit)
                            sensor_controller_widget = SensorControllerWidget(
                                board_config,
                                sensor_default_data,
                                board_name,
                                sensor_name,
                                sensor_type,
                                self.udp_manager,
                                self,
                                unit,
                                unit_color,
                            )
                            sensors_by_unit.setdefault(unit, []).append(sensor_controller_widget)
                            self.sensor_list.append(sensor_controller_widget)

        bottom_row_widgets = []
        for unit, widgets in sensors_by_unit.items():
            if not widgets:
                continue

            # Multi-sensor units get their own dedicated row; singletons share the bottom grid.
            if len(widgets) == 1:
                bottom_row_widgets.extend(widgets)
                continue

            type_group = QGroupBox(unit)
            type_layout = QGridLayout()
            type_layout.setContentsMargins(2, 2, 2, 2)
            type_layout.setHorizontalSpacing(6)
            type_layout.setVerticalSpacing(4)

            for idx, sensor_widget in enumerate(widgets):
                type_layout.addWidget(sensor_widget, idx // columns, idx % columns)
            for col in range(columns):
                type_layout.setColumnStretch(col, 1)

            type_group.setLayout(type_layout)
            sensor_mainlayout.addWidget(type_group)
            self.sensor_plot_row_count += (len(widgets) + columns - 1) // columns
            self.sensor_unit_group_count += 1

        if bottom_row_widgets:
            bottom_group = QGroupBox("single channels")
            bottom_layout = QGridLayout()
            bottom_layout.setContentsMargins(2, 2, 2, 2)
            bottom_layout.setHorizontalSpacing(6)
            bottom_layout.setVerticalSpacing(4)
            for idx, sensor_widget in enumerate(bottom_row_widgets):
                bottom_layout.addWidget(sensor_widget, idx // columns, idx % columns)
            for col in range(columns):
                bottom_layout.setColumnStretch(col, 1)
            bottom_group.setLayout(bottom_layout)
            sensor_mainlayout.addWidget(bottom_group)
            self.sensor_plot_row_count += (len(bottom_row_widgets) + columns - 1) // columns
            self.sensor_unit_group_count += 1

        sensor_group.setLayout(sensor_mainlayout)
        self.sensor_area.addWidget(sensor_group)
        self.sensor_group_widget = sensor_group
        QTimer.singleShot(0, self._apply_adaptive_sensor_plot_heights)

    def _apply_adaptive_sensor_plot_heights(self):
        if self.sensor_group_widget is None or self.sensor_plot_row_count <= 0:
            return

        group_height = self.sensor_group_widget.height()
        if group_height <= 0:
            return

        # Reserve space for section headers and per-plot value labels; fill the remainder with equal plot heights.
        sensors_header_px = 26
        unit_group_header_px = 24
        unit_group_spacing_px = 6
        label_block_px = 34
        reserved_px = (
            sensors_header_px
            + (self.sensor_unit_group_count * unit_group_header_px)
            + (max(0, self.sensor_unit_group_count - 1) * unit_group_spacing_px)
            + (self.sensor_plot_row_count * label_block_px)
            + 16
        )
        available_plot_px = max(0, group_height - reserved_px)
        raw_height = available_plot_px // self.sensor_plot_row_count if self.sensor_plot_row_count else SENSOR_PLOT_MIN_HEIGHT_PX
        target_plot_height = max(SENSOR_PLOT_MIN_HEIGHT_PX, min(SENSOR_PLOT_MAX_HEIGHT_PX, raw_height))

        for sensor_widget in self.sensor_list:
            sensor_widget.set_plot_height(target_plot_height)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_adaptive_sensor_plot_heights()

    def reload_hardware_json_received(self, response):
        try:
            response_json = json.loads(response)
            if "response" not in response_json:
                return
        except json.JSONDecodeError:
            return
        self.udp_manager.request_hardware_json()
    def time_received(self, responsestr):
        try:
            response = json.loads(responsestr)["response"]
        except json.JSONDecodeError:
            print("Time is not valid JSON")
            return
        if "date_time" in response:
            self.datetime_str = response["date_time"]
            self.datetime_label.setText(f"Backend Time: {self.datetime_str}")
        else:
            print("No date_time in response")
        if "hotfire_time" in response:
            self.hotfiretime_str = response["hotfire_time"]
            self.hotfiretime_label.setText(f"{self.hotfiretime_str}")
        else:
            print("No hotfire_time in response")

    def machinestate_received(self, response):
        try:
            self.statemachine_str = json.loads(response)["response"]
            self.statemachine_label.setText(f"Machine state: {self.statemachine_str}")
        except json.JSONDecodeError:
            print("Machine state is not valid JSON")
            return
    def handle_data_received(self, data):
        try:
            data = json.loads(data)
            #print(f"Decoded data: {data}")
            if "command" not in data:
                print("No command in data json, backend is being weird")
                return
            if "response" not in data:
                print("No response in data json, backend is being weird")
                return
            
            command = data["command"]
            response = data["response"]
            #print(type(response))
            if command in self.commands_responses:
                self.commands_responses[command](response)
            else:
                self.command_response.append(json.dumps(data, indent=4))
        except json.JSONDecodeError:
            print("Failed to decode JSON data")
    def hardware_json_received(self, response):
        global pt_number
        pt_number = 0
        self.udp_manager.set_hardware_json_received(True)
        if self.data_waiting_label is not None:
            self.data_waiting_label.setParent(None)
            self.data_waiting_label.deleteLater()
            self.data_waiting_label = None
        try :
            response = json.loads(response)
        except json.JSONDecodeError:
            print("Hardware json is not valid JSON")
            return
        
        self.hardware_json = response
        if "boards" not in self.hardware_json:
            print("No boards in hardware json")
            return
        
        if "state_defaults" not in self.hardware_json:
            print("No state defaults in hardware json")
            return
        
        self.state_defaults: dict = self.hardware_json['state_defaults']
        self.hardware_types = list(self.state_defaults.keys())

        self._clear_dynamic_ui()
        self._build_dynamic_ui_from_hardware()

    def boards_desired_states_received(self, response):
        if self.hardware_json is None:
            return
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            print("Boards desired states is not valid JSON")
            return
        
        for board_name, states in response.items():
            if board_name in self.hardware_json["boards"]:
                for actuator in self.actuator_list:
                    if actuator.board_name == board_name:
                        actuator.update_desired_states(states)
            else:
                print(f"Board {board_name} not found in hardware json")

    def boards_states_received(self, response):
        if self.hardware_json is None:
            return
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            print("Boards states is not valid JSON")
            return
        
        for board_name, states in response.items():
            if board_name in self.hardware_json["boards"]:
                for actuator in self.actuator_list:
                    if actuator.board_name == board_name:
                        actuator.update_states(states)
                for sensor in self.sensor_list:
                    if sensor.board_name == board_name:
                        sensor.update_states(states)
            else:
                print(f"Board {board_name} not found in hardware json")
        self.last_update_time = QDateTime.currentDateTime()
    def backend_state_coroutine(self):
        self.udp_manager.send(json.dumps({"command":"get state", "data":{}}))
        self.udp_manager.send(json.dumps({"command":"get time", "data":{}}))
        if self.last_update_time is not None:
            current_time = QDateTime.currentDateTime()
            elapsed_time = self.last_update_time.msecsTo(current_time)
            self.last_state_update_label.setText(f"Last state update: {elapsed_time/1000:.1f} seconds ago")
            if elapsed_time > 5000:
                self.last_state_update_label.setStyleSheet("color: red")
            else:
                self.last_state_update_label.setStyleSheet("color: white")
        else:
            self.last_state_update_label.setStyleSheet("color: red")

if __name__ == "__main__":
    host = "192.168.137.179"

    # find with:
    # hostname -I
    #
    # output:
    # martin@raspberrypi:~/propbackend $ hostname -I
    # 192.168.137.179 172.26.163.215 2a0c:5bc0:40:2e26:deda:e82b:ed5e:b0b2 

    # host = "127.0.0.1"
    port = 8888

    if sys.platform == "win32":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("propfirmware.newclient.propclient")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app_icon = Path("assets/propclient.ico")
    if app_icon.exists():
        app.setWindowIcon(QIcon(str(app_icon)))

    window = PropertyTestApp(host, port)
    window.show()

    def signal_handler(sig, frame):
        print("Exiting...")
        app.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    timer = QTimer()
    timer.start(100)  # Small interval to check signals
    timer.timeout.connect(lambda: None)

    sys.exit(app.exec())