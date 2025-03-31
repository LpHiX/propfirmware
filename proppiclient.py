import socket
import sys
import json
import signal

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QTextEdit, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QSlider, QGroupBox, QGridLayout, QLineEdit)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtNetwork import QUdpSocket, QHostAddress
from PySide6.QtCore import QTimer, QByteArray, Slot
from PySide6.QtCore import Slot, QObject, Signal
from PySide6.QtCore import QDateTime
import pyqtgraph as pg

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

        self.transmit_timer = QTimer(self)
        self.transmit_timer.timeout.connect(self.request_states)
        self.transmit_timer.start(10) #10ms = 100Hz
    
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
    
    @Slot()
    def request_states(self):
        #print("Requesting boards states from backend")
        data = json.dumps({"command":"get boards states", "data":{}})
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
            self.dataReceived.emit(bytes(data).decode())

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
        self.LABEL_WIDTH = 150
        self.BUTTON_WIDTH = 100
        self.INPUT_WIDTH = 100
        
        # Create common layout
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        # Create common UI elements
        self.name_label = QLabel(f"{self.get_actuator_type_name()}: {self.actuator_name}")
        self.name_label.setFixedWidth(self.LABEL_WIDTH + 150)
        self.layout.addWidget(self.name_label)
        
        # Armed state is common to most actuators
        self.armed_label = QLabel(f"Armed: No data")
        self.armed_label.setFixedWidth(self.LABEL_WIDTH)
        self.layout.addWidget(self.armed_label)
        
        # Common arm/disarm buttons
        self.arm_button = QPushButton("Arm")
        self.arm_button.setFixedWidth(self.BUTTON_WIDTH)
        self.arm_button.clicked.connect(lambda: self.send_command("armed", True))
        
        self.disarm_button = QPushButton("Disarm")
        self.disarm_button.setFixedWidth(self.BUTTON_WIDTH)
        self.disarm_button.clicked.connect(lambda: self.send_command("armed", False))
        
        self.layout.addWidget(self.arm_button)
        self.layout.addWidget(self.disarm_button)
        
        # Add specific UI elements
        self.setup_specific_ui()
        self.layout.addStretch()
    
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
    
    def setup_specific_ui(self):
        """Set up actuator-specific UI elements. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement setup_specific_ui()")
    
    def update_specific_states(self, states):
        """Update actuator-specific states. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement update_specific_states()")
    
    def get_actuator_type_name(self):
        """Return human-readable actuator type name. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement get_actuator_type_name()")

class ServoControllerWidget(ActuatorControllerWidget):
    def __init__(self, config, board_name, servo_name, udpmanager):
        super().__init__(config, board_name, servo_name, "servos", udpmanager)
    
    def get_actuator_type_name(self):
        return "Servo"
    
    def setup_specific_ui(self):
        # Servo-specific UI elements
        self.angle_label = QLabel("Angle: No data")
        self.angle_label.setFixedWidth(self.LABEL_WIDTH)
        self.layout.addWidget(self.angle_label)
        
        self.manual_angle = QLineEdit(self)
        self.manual_angle.setFixedWidth(self.INPUT_WIDTH)
        self.manual_angle.setPlaceholderText("0")
        
        self.manual_angle_button = QPushButton("Set Angle")
        self.manual_angle_button.setFixedWidth(self.BUTTON_WIDTH)
        self.manual_angle_button.clicked.connect(self.set_manual_angle)
        
        self.layout.addWidget(self.manual_angle)
        self.layout.addWidget(self.manual_angle_button)
    
    def update_specific_states(self, states):
        try:
            angle = states[self.actuator_type][self.actuator_name]["angle"]
            self.angle_label.setText(f"Angle: {angle}")
        except KeyError:
            self.angle_label.setText("Angle: No data")
    
    def set_manual_angle(self):
        try:
            angle = int(self.manual_angle.text())
            if angle < 0 or angle > 180:
                raise ValueError("Angle must be between 0 and 180")
            self.send_command("angle", angle)
        except ValueError as e:
            print(f"Invalid angle: {e}")

class SolenoidControllerWidget(ActuatorControllerWidget):
    def __init__(self, config, board_name, actuator_name, udpmanager):
        super().__init__(config, board_name, actuator_name, "solenoids", udpmanager)
    
    def get_actuator_type_name(self):
        return "Solenoid"
    
    def setup_specific_ui(self):
        # Solenoid-specific UI elements
        self.powered_label = QLabel("Powered: No data")
        self.powered_label.setFixedWidth(self.LABEL_WIDTH)
        self.layout.addWidget(self.powered_label)
        
        self.poweron_button = QPushButton("Power On")
        self.poweron_button.setFixedWidth(self.BUTTON_WIDTH)
        self.poweron_button.clicked.connect(lambda: self.send_command("powered", True))
        
        self.poweroff_button = QPushButton("Power Off")
        self.poweroff_button.setFixedWidth(self.BUTTON_WIDTH)
        self.poweroff_button.clicked.connect(lambda: self.send_command("powered", False))
        
        self.layout.addWidget(self.poweron_button)
        self.layout.addWidget(self.poweroff_button)
    
    def update_specific_states(self, states):
        try:
            powered = states[self.actuator_type][self.actuator_name]["powered"]
            self.powered_label.setText(f"Powered: {powered}")
        except KeyError:
            self.powered_label.setText("Powered: No data")

class SensorControllerWidget(QWidget):
    """"Base class for all sensor controller widgets"""
    def __init__(self, config, sensor_default_data, board_name, sensor_name, sensor_type, udpmanager):
        super().__init__()
        self.config = config
        self.sensor_default_data = sensor_default_data
        self.board_name = board_name
        self.sensor_name = sensor_name
        self.sensor_type = sensor_type  # "pts", "tcs", etc.
        self.udpmanager = udpmanager
        
        # Define standard widths
        self.LABEL_WIDTH = 100
        
        # Create common layout
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        # Create common UI elements
        self.name_label = QLabel(f"{self.sensor_type}: {self.sensor_name}")
        self.name_label.setFixedWidth(self.LABEL_WIDTH + 150)
        self.layout.addWidget(self.name_label)

        self.values = {}
        for value_name, _ in self.sensor_default_data.items():
            self.values[value_name] = {"value": "No data"}
            self.values[value_name]["label"] = QLabel(f"{value_name}: No data")
            self.values[value_name]["label"].setFixedWidth(self.LABEL_WIDTH)
            self.values[value_name]["history"] = []
            self.vertical_layout = QVBoxLayout()
            self.vertical_layout.addWidget(self.values[value_name]["label"])

            # self.plot = pg.PlotWidget(title=f"{self.sensor_type} {self.sensor_name} {value_name}")
            # self.vertical_layout.addWidget(self.plot)

            self.layout.addLayout(self.vertical_layout)
            
            # self.value_timer = QTimer(self)
            # self.value_timer.timeout.connect(self.update_history(value_name))
            # self.value_timer.start(10)
        self.layout.addStretch()

    # def update_history(self, value_name):
    #     value_history = self.values[value_name]["history"]
    #     for timestamp, value in enumerate(value_history):
    #         while value_history[0][0] < QDateTime.currentDateTime().addSecs(-10):
    #             del value_history[0]
    #         while len(value_history) > 100:
    #             del value_history[0]
    #     if len(value_history) == 0:
    #         return
    #     time_arr, sensor_arr = zip(*value_history)
    #     print(f"Time array: {time_arr} Sensor array: {sensor_arr}")
    #     self.plot.clear()
    #     self.plot.plot(time_arr, sensor_arr, pen='g')

    def update_states(self, states):
        for value_name, value_dict in self.values.items():
            try:
                value = states[self.sensor_type][self.sensor_name][value_name]
                value_label = value_dict["label"]
                if "adc" in self.config[self.sensor_type][self.sensor_name]:
                    gain = self.config[self.sensor_type][self.sensor_name]["gain"]
                    offset = self.config[self.sensor_type][self.sensor_name]["offset"]
                    value = (value - offset) * gain
                    value_label.setText(f"Value: {value}")
                else:
                    value_label.setText(f"Value: {value}")
                self.values[self.sensor_name]["history"].append((QDateTime.currentDateTime(), value))
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


        self.setWindowTitle("Property Test App")
        #self.setGeometry(100, 100, 400, 300)
        #self.text_edit = QTextEdit(self)
        self.data_waiting_label = QLabel("Waiting for data...")

        self.last_update_time = None
        self.last_state_update_label = QLabel("Last state update: Never")
        self.last_state_update_label.setFixedWidth(200)
        self.last_update_timer = QTimer(self)
        self.last_update_timer.timeout.connect(self.update_last_state_update_label)
        self.last_update_timer.start(100)
        self.manual_command = QTextEdit(self)
        self.manual_response = QTextEdit(self)
        self.manual_response.setReadOnly(True)
        self.manual_command_button = QPushButton("Send Command")
        self.manual_command_button.clicked.connect(self.send_manual_command)
        self.manual_command_layout = QVBoxLayout()
        self.manual_command_layout.addWidget(self.last_state_update_label)
        self.manual_command_layout.addWidget(self.manual_command)
        self.manual_command_layout.addWidget(self.manual_command_button)
        self.manual_command_layout.addWidget(self.manual_response)

        self.hotfire_button = QPushButton("Hotfire")
        self.hotfire_button.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"start hotfire sequence", "data":{}})))
        self.preset_commands_layout = QVBoxLayout()
        self.preset_commands_layout.addWidget(self.hotfire_button)
        self.preset_commands_layout.addStretch()

        self.commands_layout = QHBoxLayout()
        self.commands_layout.addLayout(self.manual_command_layout)
        self.commands_layout.addLayout(self.preset_commands_layout)

        self.control_area = QVBoxLayout()
        self.sensor_area = QVBoxLayout()

        self.actuators_sensors_area = QVBoxLayout()
        self.actuators_sensors_area.addLayout(self.control_area)
        self.actuators_sensors_area.addLayout(self.sensor_area)

        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.data_waiting_label)
        self.main_layout.addLayout(self.actuators_sensors_area)
        self.main_layout.addLayout(self.commands_layout)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)  

        self.udp_manager = UDPManager(self.host, self.port)
        self.udp_manager.dataReceived.connect(self.handle_data_received)

        self.commands_responses = {
            "get hardware json": self.hardware_json_received,
            "get boards states": self.boards_states_received,
        }
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
                self.manual_response.append(json.dumps(data, indent=4))
        except json.JSONDecodeError:
            print("Failed to decode JSON data")
    def hardware_json_received(self, response):
        self.udp_manager.set_hardware_json_received(True)
        self.main_layout.removeWidget(self.data_waiting_label)
        self.data_waiting_label.setParent(None)
        self.data_waiting_label.deleteLater()
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
        
        actuator_group = QGroupBox("Actuators")
        actuator_layout = QVBoxLayout()
        self.actuator_list.clear()

        default_actuator_types = {
            "servos": ServoControllerWidget,
            "solenoids": SolenoidControllerWidget,
        }

        for board_name, board_config in self.hardware_json["boards"].items():
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
        
        sensor_group = QGroupBox("Sensors")
        sensor_layout = QVBoxLayout()
        self.sensor_list.clear()
        for board_name, board_config in self.hardware_json["boards"].items():
            if not board_config.get("is_actuator", False):
                for sensor_type, sensor_default_data in self.state_defaults.items():
                    if sensor_type in board_config and isinstance(board_config[sensor_type], dict):
                        for sensor_name, _ in board_config[sensor_type].items():
                            sensor_controller_widget = SensorControllerWidget(board_config, sensor_default_data, board_name, sensor_name, sensor_type, self.udp_manager)
                            sensor_layout.addWidget(sensor_controller_widget)
                            self.sensor_list.append(sensor_controller_widget)
        sensor_layout.addStretch()
        sensor_group.setLayout(sensor_layout)
        self.sensor_area.addWidget(sensor_group)

    def boards_states_received(self, response):
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
    def send_manual_command(self):
        command = self.manual_command.toPlainText()
        if not command:
            return
        #self.manual_response.clear()
        try:
            command_json = json.loads(command)
            if "command" not in command_json:
                self.manual_response.append("No command in JSON")
                return
            if "data" not in command_json:
                self.manual_response.append("No data in JSON")
                return
            self.udp_manager.send(command)
        except json.JSONDecodeError:
            self.manual_response.append("Invalid JSON format")
    def update_last_state_update_label(self):
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
    host = "192.168.137.216"
    port = 8888

    app = QApplication(sys.argv)
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