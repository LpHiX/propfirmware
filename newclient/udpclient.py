from PySide6.QtCore import QTimer, Qt
from PySide6.QtNetwork import QUdpSocket, QHostAddress
from PySide6.QtCore import QTimer, QByteArray, Slot
from PySide6.QtCore import Slot, QObject, Signal
from PySide6.QtCore import QDateTime
import json

class UDPClient(QObject):
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

    def send(self, data):
        self.socket.writeDatagram(QByteArray(data.encode()), QHostAddress(self.host), self.port)

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