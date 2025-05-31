from PySide6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                              QLabel, QPushButton, QTextEdit, QGroupBox)
from PySide6.QtCore import QTimer, QDateTime
import json

from ui.sensor_widget import SensorControllerWidget
# Import other necessary classes

class MainWindow(QMainWindow):
    def __init__(self, data_manager, udp_manager):
        super().__init__()
        self.data_manager = data_manager
        self.udp_manager = udp_manager
        
        # Initialize UI elements
        self.statemachine_str = "No data"
        self.datetime_str = "No data"
        self.hotfiretime_str = "No data"
        
        # Setup UI components
        self.setup_ui()
        
        # Setup timers
        self.backend_state_timer = QTimer(self)
        self.backend_state_timer.timeout.connect(self.backend_state_coroutine)
        self.backend_state_timer.start(10)
        
        # Connect signals
        self.udp_manager.dataReceived.connect(self.handle_manual_command_response)
    
    def setup_ui(self):
        self.setWindowTitle("Propulsion Test Client")
        
        # Main layout
        main_widget = QWidget()
        self.main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # Setup sensor area
        self.sensor_area = QVBoxLayout()
        
        # Setup status labels
        self.statemachine_label = QLabel(f"Machine state: {self.statemachine_str}")
        self.datetime_label = QLabel(f"Backend Time: {self.datetime_str}")
        self.hotfiretime_label = QLabel(f"{self.hotfiretime_str}")
        self.last_state_update_label = QLabel("Last state update: Never")
        
        # Setup manual command area
        self.manual_command = QTextEdit()
        self.manual_response = QTextEdit()
        self.manual_response.setReadOnly(True)
        self.manual_command_button = QPushButton("Send Command")
        self.manual_command_button.clicked.connect(self.send_manual_command)
        
        # Add other UI initialization as needed
        # ...

    def setup_sensor_widgets(self, board_name, board_config):
        sensor_group = QGroupBox("Sensors")
        sensor_mainlayout = QVBoxLayout()
        
        for sensor_type in self.data_manager.hardware_types:
            if sensor_type in board_config and isinstance(board_config[sensor_type], dict):
                sensor_layout = QHBoxLayout()
                sensor_mainlayout.addLayout(sensor_layout)
                for sensor_name, _ in board_config[sensor_type].items():
                    sensor_widget = SensorControllerWidget(
                        board_config,
                        self.data_manager.state_defaults.get(sensor_type, {}),
                        board_name,
                        sensor_name,
                        sensor_type,
                        self.udp_manager,
                        self
                    )
                    sensor_layout.addWidget(sensor_widget)
                    self.data_manager.register_sensor_widget(
                        board_name, sensor_type, sensor_name, sensor_widget
                    )

        sensor_mainlayout.addStretch()
        sensor_group.setLayout(sensor_mainlayout)
        self.sensor_area.addWidget(sensor_group)
    
    def update_board_states(self, board_name, states):
        # Update actuator widgets for this board
        for key, widget in self.data_manager.actuator_widgets.items():
            if widget.board_name == board_name:
                widget.update_states(states)
        
        # Update sensor widgets for this board
        for key, widget in self.data_manager.sensor_widgets.items():
            if widget.board_name == board_name:
                widget.update_states(states)
    
    def update_machine_state(self, state):
        self.statemachine_str = state
        self.statemachine_label.setText(f"Machine state: {self.statemachine_str}")
    
    def update_time(self, datetime_str, hotfiretime_str):
        self.datetime_str = datetime_str
        self.datetime_label.setText(f"Backend Time: {self.datetime_str}")
        
        self.hotfiretime_str = hotfiretime_str
        self.hotfiretime_label.setText(f"{self.hotfiretime_str}")
    
    def backend_state_coroutine(self):
        self.udp_manager.send(json.dumps({"command":"get state", "data":{}}))
        self.udp_manager.send(json.dumps({"command":"get time", "data":{}}))
        
        elapsed_time = self.data_manager.get_last_update_elapsed_time()
        if elapsed_time is not None:
            self.last_state_update_label.setText(f"Last state update: {elapsed_time/1000:.1f} seconds ago")
            if elapsed_time > 5000:
                self.last_state_update_label.setStyleSheet("color: red")
            else:
                self.last_state_update_label.setStyleSheet("color: white")
        else:
            self.last_state_update_label.setText("Last state update: Never")
            self.last_state_update_label.setStyleSheet("color: red")
    
    def handle_manual_command_response(self, command, response_str):
        # Only update the manual response text area for commands that aren't
        # automatically handled by the data manager
        standard_commands = ["get hardware json", "get boards states", "get state", "get time"]
        if command not in standard_commands:
            try:
                response = json.loads(response_str)
                self.manual_response.append(json.dumps({
                    "command": command,
                    "response": response
                }, indent=4))
            except json.JSONDecodeError:
                self.manual_response.append(f"Invalid JSON response for command: {command}")
    
    def send_manual_command(self):
        command = self.manual_command.toPlainText()
        if not command:
            return
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
