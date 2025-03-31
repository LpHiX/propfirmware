import socket
import sys
import json
import time
import threading

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QTextEdit, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QSlider, QGroupBox, QGridLayout)
from PySide6.QtCore import QTimer, Qt

class PropertyTestApp(QMainWindow):
    def __init__(self, udp_client):
        super().__init__()
        self.setWindowTitle("Property Test App")
        self.setGeometry(100, 100, 800, 600)
        self.udp_client = udp_client
        self.hardware_json = None

        # Create widgets
        self.text_edit = QTextEdit(self)
        self.response_text_edit = QTextEdit(self)
        self.response_text_edit.setReadOnly(True)
        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)
        
        # Create control area
        self.control_area = QVBoxLayout()
        self.sensor_area = QVBoxLayout()
        
        # Update data button
        self.update_data_button = QPushButton("Update Data", self)
        self.update_data_button.clicked.connect(self.request_board_states)

        # Layouts
        main_layout = QVBoxLayout()
        
        text_layout = QHBoxLayout()
        text_layout.addWidget(self.text_edit)
        text_layout.addWidget(self.response_text_edit)
        
        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.update_data_button)
        
        main_layout.addLayout(text_layout)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(self.control_area)
        main_layout.addLayout(self.sensor_area)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Request hardware json on startup
        QTimer.singleShot(100, self.request_hardware_json)

    def request_hardware_json(self):
        try:
            request = {
                "command": "get hardware json",
                "data": {}
            }
            self.udp_client.sock.sendto(json.dumps(request).encode(), 
                                       (self.udp_client.host, self.udp_client.port))
            response, _ = self.udp_client.sock.recvfrom(4096)
            response_text = response.decode()
            print(f"Hardware JSON Received: {response_text}")
            self.hardware_json = json.loads(response_text)
            self.build_controls_from_hardware()
        except Exception as e:
            print(f"Error requesting hardware JSON: {e}")
            
    def build_controls_from_hardware(self):
        if not self.hardware_json or 'boards' not in self.hardware_json:
            print("No valid hardware JSON available")
            return
            
        # Clear existing controls
        self.clear_layout(self.control_area)
        self.clear_layout(self.sensor_area)
        
        # Build UI for each board
        for board in self.hardware_json.get('boards', []):
            board_name = board.get('board_name', 'Unknown Board')
            
            # Create actuator controls if applicable
            if board.get('is_actuator', False):
                actuator_group = QGroupBox(f"Actuator Controls: {board_name}")
                actuator_layout = QGridLayout()
                row = 0
                
                # Add servo controls
                for servo_name, servo_data in board.get('servos', {}).items():
                    label = QLabel(f"{servo_name}:")
                    angle_slider = QSlider(Qt.Horizontal)
                    angle_slider.setMinimum(servo_data.get('minangle', 0))
                    angle_slider.setMaximum(servo_data.get('maxangle', 180))
                    angle_slider.setValue(servo_data.get('safe_angle', 0))
                    angle_label = QLabel(f"{angle_slider.value()}°")
                    arm_button = QPushButton("Arm")
                    
                    actuator_layout.addWidget(label, row, 0)
                    actuator_layout.addWidget(angle_slider, row, 1)
                    actuator_layout.addWidget(angle_label, row, 2)
                    actuator_layout.addWidget(arm_button, row, 3)
                    row += 1
                    
                # Add solenoid controls
                for solenoid_name, solenoid_data in board.get('solenoids', {}).items():
                    label = QLabel(f"{solenoid_name}:")
                    power_button = QPushButton("Power On")
                    arm_button = QPushButton("Arm")
                    
                    actuator_layout.addWidget(label, row, 0)
                    actuator_layout.addWidget(power_button, row, 1)
                    actuator_layout.addWidget(arm_button, row, 3)
                    row += 1
                
                # Add pyro controls
                for pyro_name, pyro_data in board.get('pyros', {}).items():
                    label = QLabel(f"{pyro_name}:")
                    fire_button = QPushButton("Fire")
                    arm_button = QPushButton("Arm")
                    
                    actuator_layout.addWidget(label, row, 0)
                    actuator_layout.addWidget(fire_button, row, 1)
                    actuator_layout.addWidget(arm_button, row, 3)
                    row += 1
                
                actuator_group.setLayout(actuator_layout)
                self.control_area.addWidget(actuator_group)
            
            # Create sensor display area for all boards
            sensor_group = QGroupBox(f"Sensor Data: {board_name}")
            sensor_layout = QGridLayout()
            row = 0
            
            # Add pressure transducer displays
            for pt_name, pt_data in board.get('pts', {}).items():
                label = QLabel(f"{pt_name}:")
                value_label = QLabel("N/A")
                value_label.setObjectName(f"sensor_{board_name}_{pt_name}")
                
                sensor_layout.addWidget(label, row, 0)
                sensor_layout.addWidget(value_label, row, 1)
                row += 1
            
            # Add thermocouple displays
            for tc_name, tc_data in board.get('tcs', {}).items():
                label = QLabel(f"{tc_name}:")
                value_label = QLabel("N/A")
                value_label.setObjectName(f"sensor_{board_name}_{tc_name}")
                
                sensor_layout.addWidget(label, row, 0)
                sensor_layout.addWidget(value_label, row, 1)
                row += 1
            
            # Add GPS displays
            for gps_name, gps_data in board.get('gps', {}).items():
                label = QLabel(f"{gps_name}:")
                value_label = QLabel("N/A")
                value_label.setObjectName(f"sensor_{board_name}_{gps_name}")
                
                sensor_layout.addWidget(label, row, 0)
                sensor_layout.addWidget(value_label, row, 1)
                row += 1
            
            sensor_group.setLayout(sensor_layout)
            self.sensor_area.addWidget(sensor_group)
        
    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())
    
    def request_board_states(self):
        try:
            request = {
                "command": "get boards states",
                "data": {}
            }
            self.udp_client.sock.sendto(json.dumps(request).encode(), 
                                      (self.udp_client.host, self.udp_client.port))
            response, _ = self.udp_client.sock.recvfrom(4096)
            response_text = response.decode()
            print(f"Board States Received: {response_text}")
            self.update_sensor_values(json.loads(response_text))
        except Exception as e:
            print(f"Error requesting board states: {e}")
    
    def update_sensor_values(self, board_states):
        if not board_states or 'boards' not in board_states:
            print("No valid board states available")
            return
            
        for board in board_states.get('boards', []):
            board_name = board.get('board_name', 'Unknown Board')
            
            # Update pressure transducer values
            for pt_name, pt_data in board.get('pts', {}).items():
                value_label = self.findChild(QLabel, f"sensor_{board_name}_{pt_name}")
                if value_label:
                    value_label.setText(f"{pt_data.get('raw', 'N/A')}")
            
            # Update thermocouple values
            for tc_name, tc_data in board.get('tcs', {}).items():
                value_label = self.findChild(QLabel, f"sensor_{board_name}_{tc_name}")
                if value_label:
                    value_label.setText(f"{tc_data.get('raw', 'N/A')}")
            
            # Update GPS values
            for gps_name, gps_data in board.get('gps', {}).items():
                value_label = self.findChild(QLabel, f"sensor_{board_name}_{gps_name}")
                if value_label:
                    value_label.setText(str(gps_data.get('channel', 'N/A')))

    def send_message(self):
        message = self.text_edit.toPlainText()
        try:
            self.udp_client.sock.sendto(message.encode(), (self.udp_client.host, self.udp_client.port))
            response, server = self.udp_client.sock.recvfrom(4096)
            response_text = response.decode()
            print(f"Received: {response_text}")
            self.response_text_edit.setPlainText(json.dumps(json.loads(response_text), indent=4))
        except socket.timeout:
            print("Request timed out")
        except Exception as e:
            print(f"Error: {e}")

class UDPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(5)

if __name__ == "__main__":
    host = "192.168.137.216"
    port = 8888
    client = UDPClient(host, port)

    app = QApplication(sys.argv)
    window = PropertyTestApp(client)
    window.show()
    sys.exit(app.exec())