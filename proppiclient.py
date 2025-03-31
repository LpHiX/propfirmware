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

class UDPManager(QObject):
    dataReceived = Signal(str)
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.socket = QUdpSocket(self)
        self.socket.readyRead.connect(self.process_datagrams)

        QTimer.singleShot(0, self.request_hardware_json)

        self.transmit_timer = QTimer(self)
        self.transmit_timer.timeout.connect(self.request_states)
        self.transmit_timer.start(10) #10ms = 100Hz

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
    def send_actuator_command(self, board_name, actuator_type, actuator_name, command, value):
        message = {
            "command": "update desired state",
            "data":{
                "board_name": board_name,
                "message": {
                    actuator_type: {
                        actuator_name: {
                            command: value
                        }
                    }
                }
            }
        }
        self.send(json.dumps(message))

class ServoControllerWidget(QWidget):
    def __init__(self, config, board_name, servo_name, udpmanager):
        super().__init__()
        self.config = config
        self.board_name = board_name
        self.servo_name = servo_name
        self.udpmanager: UDPManager = udpmanager

        try:
            angle = self.config["boards"][self.board_name]["servos"][self.servo_name]["angle"]
        except KeyError:
            angle = "No data"
        
        try:
            armed = self.config["boards"][self.board_name]["servos"][self.servo_name]["armed"]
        except KeyError:
            armed = "No data"

        self.name_label = QLabel(f"Servo: {self.servo_name}")
        self.angle_label = QLabel(f"Angle: {angle}")
        self.armed_label = QLabel(f"Armed: {armed}")

        self.manual_angle = QLineEdit(self)
        self.manual_angle.setPlaceholderText("0")
        self.manual_angle_button = QPushButton("Set Angle")
        self.manual_angle_button.clicked.connect(self.set_manual_angle)

        self.arm_button = QPushButton("Arm")
        self.arm_button.clicked.connect(lambda: self.udpmanager.send_actuator_command(self.board_name, "servos", self.servo_name, "armed", True))
        self.disarm_button = QPushButton("Disarm")
        self.disarm_button.clicked.connect(lambda: self.udpmanager.send_actuator_command(self.board_name, "servos", self.servo_name, "armed", False))
        layout = QHBoxLayout()

        layout.addWidget(self.name_label)
        layout.addWidget(self.angle_label)
        layout.addWidget(self.armed_label)
        layout.addWidget(self.manual_angle)
        layout.addWidget(self.manual_angle_button)
        layout.addWidget(self.arm_button)
        layout.addWidget(self.disarm_button)
        self.setLayout(layout)

    def update_states(self, states):

        try:
            angle = states["servos"][self.servo_name]["angle"]
        except KeyError:
            angle = "No data"
        
        #print(f'{self.servo_name},{angle}')
        self.angle_label.setText(f"Angle: {angle}")

        try:
            armed = states["servos"][self.servo_name]["armed"]
        except KeyError:
            armed = "No data"

        self.armed_label.setText(f"Armed: {armed}")

    def set_manual_angle(self, angle):
        try:
            angle = int(self.manual_angle.text())
            if angle < 0 or angle > 180:
                raise ValueError("Angle must be between 0 and 180")
            self.udpmanager.send_actuator_command(self.board_name, "servos", self.servo_name, "angle", angle)
        except ValueError as e:
            print(f"Invalid angle: {e}")

class PropertyTestApp(QMainWindow):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.hardware_json = None
        self.actuator_list = []
        
        self.setWindowTitle("Property Test App")
        #self.setGeometry(100, 100, 400, 300)
        #self.text_edit = QTextEdit(self)
        self.data_waiting_label = QLabel("Waiting for data... (NEED TO RESTART PROGRAM, RECONNECTION NOT IMPLEMENTED)")
        #Need to implement reconnection logic here

        # self.manual_command = QTextEdit(self)
        # self.manual_response = QTextEdit(self)
        # self.manual_response.setReadOnly(True)
        # self.manual_command_button = QPushButton("Send Command")
        # self.manual_command_button.clicked.connect(self.send_manual_command)
        # self.manual_command_layout = QVBoxLayout()

        self.control_area = QVBoxLayout()
        self.sensor_area = QVBoxLayout()

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.data_waiting_label)
        self.main_layout.addLayout(self.control_area)
        self.main_layout.addLayout(self.sensor_area)

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
                print(f"Recieved \"{command}\" with response: {response}")
        except json.JSONDecodeError:
            print("Failed to decode JSON data")
    def hardware_json_received(self, response):
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
        
        actuator_group = QGroupBox("Actuators")
        actuator_layout = QVBoxLayout()
        self.actuator_list.clear()

        for board_name, board_config in self.hardware_json["boards"].items():
            if board_config.get("is_actuator", False):
                for servo_name, _ in board_config.get('servos', {}).items():
                    servo_controller_widget = ServoControllerWidget(self.hardware_json, board_name, servo_name, self.udp_manager)
                    actuator_layout.addWidget(servo_controller_widget)
                    self.actuator_list.append(servo_controller_widget)
            
        actuator_group.setLayout(actuator_layout)
        self.control_area.addWidget(actuator_group)
        print(len(self.actuator_list))

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
            else:
                print(f"Board {board_name} not found in hardware json")
        pass
    

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