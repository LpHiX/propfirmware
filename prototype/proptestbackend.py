# Two goals:
# Send UART command packets to actuator esp32
# Send UART packet request to DAQ esp32, then receive and process the response

import argparse
import serial
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QTextEdit, QVBoxLayout, QHBoxLayout, QWidget)
from PySide6.QtCore import QTimer

class PropertyTestApp(QMainWindow):
    def __init__(self, serial_device=None):
        super().__init__()
        self.serial_device = serial_device
        self.serial_connection = None
        
        if serial_device:
            try:
                self.serial_connection = serial.Serial(serial_device, baudrate=115200, timeout=1)
                print("Connected to device:", serial_device)
            except serial.SerialException as e:
                print(f"Error connecting to device: {e}")
        else:
            print("No device specified. Running in debug mode.")
        
        self.init_ui()
        
        # Set up timer for reading from serial port
        self.serial_timer = QTimer()
        self.serial_timer.timeout.connect(self.read_serial)
        self.serial_timer.start(100)  # Read every 100 ms
    
    def init_ui(self):
        self.setWindowTitle("Property Testing Interface")
        self.setGeometry(100, 100, 800, 500)
        
        # Create main layout widget
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel with button
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.send_button = QPushButton("Send Command")
        self.send_button.clicked.connect(self.send_command)
        left_layout.addWidget(self.send_button)
        
        # Add a spacer to push button to the top
        left_layout.addStretch()
        
        # Right panel with serial monitor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.serial_monitor = QTextEdit()
        self.serial_monitor.setReadOnly(True)
        right_layout.addWidget(self.serial_monitor)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 3)  # Give the right panel more space
        
        self.setCentralWidget(central_widget)
        
    def send_command(self):
        # Example command - this would be replaced with actual commands
        command = b'TEST_COMMAND\n'
        self.serial_monitor.append(f"Sending: {command.decode().strip()}")
        
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.write(command)
            except Exception as e:
                self.serial_monitor.append(f"Error sending command: {e}")
        else:
            self.serial_monitor.append("Debug mode: Command simulated")
    
    def read_serial(self):
        if self.serial_connection and self.serial_connection.is_open:
            try:
                if self.serial_connection.in_waiting:
                    data = self.serial_connection.readline()
                    self.serial_monitor.append(f"Received: {data.decode().strip()}")
            except Exception as e:
                self.serial_monitor.append(f"Error reading from serial: {e}")
    
    def closeEvent(self, event):
        # Clean up the serial connection when closing the application
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        event.accept()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Property Testing Backend")
    parser.add_argument("--device", "-d", help="Serial device path (e.g. COM3)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    app = QApplication(sys.argv)
    window = PropertyTestApp(args.device)
    window.show()
    sys.exit(app.exec())