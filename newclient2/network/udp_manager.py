from PySide6.QtCore import QObject, Signal, QTimer, QByteArray, Slot
from PySide6.QtNetwork import QUdpSocket, QHostAddress
import json

class UDPManager(QObject):
    dataReceived = Signal(str, str)  # command, response_json_str
    
    def __init__(self, host, port, data_manager):
        super().__init__()
        self.host = host
        self.port = port
        self.data_manager = data_manager
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
            if received:
                # Stop checking once we've received it
                self.hardware_json_timer.stop()
                self.transmit_timer = QTimer(self)
                self.transmit_timer.timeout.connect(self.request_states)
                self.transmit_timer.start(10) #10ms = 100Hz
    
    @Slot()
    def request_states(self):
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
            data_str = bytes(data).decode()
            
            try:
                data_json = json.loads(data_str)
                if "command" in data_json and "response" in data_json:
                    command = data_json["command"]
                    response = data_json["response"]
                    
                    # Process through data manager based on command
                    if command == "get hardware json":
                        if self.data_manager.set_hardware_json(json.dumps(response)):
                            self.set_hardware_json_received(True)
                    elif command == "get boards states":
                        self.data_manager.update_board_states(json.dumps(response))
                    elif command == "get state":
                        self.data_manager.update_machine_state(json.dumps(response))
                    elif command == "get time":
                        self.data_manager.update_system_time(json.dumps(response))
                    
                    # Emit the signal for any other listeners
                    self.dataReceived.emit(command, json.dumps(response))
            except json.JSONDecodeError:
                print("Failed to decode JSON data")

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
