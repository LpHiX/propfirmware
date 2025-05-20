#!/usr/bin/env python3

import sys
import serial
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QGridLayout
from PySide6.QtCore import QTimer
import time
import pyqtgraph as pg

class NineDOFPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up the window
        self.setWindowTitle("9DOF Sensor Plotter")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create grid layout for plots
        plot_layout = QGridLayout()
        main_layout.addLayout(plot_layout)
        
        # Setup serial connection
        try:
            self.serial_port = serial.Serial('COM6', 9600, timeout=1)
            print("Connected to COM6")
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            self.serial_port = None
        
        # Initialize data storage
        self.max_points = 100  # Maximum number of points to display
        self.timestamps = []
        self.accel_data = [[], [], []]  # x, y, z
        self.gyro_data = [[], [], []]   # x, y, z
        self.mag_data = [[], [], []]    # x, y, z
        
        # Create plots (3x3 grid)
        self.plots = []
        self.curves = []
        
        titles = [
            "Acceleration X (m/s²)", "Acceleration Y (m/s²)", "Acceleration Z (m/s²)",
            "Gyroscope X (rad/s)", "Gyroscope Y (rad/s)", "Gyroscope Z (rad/s)",
            "Magnetometer X", "Magnetometer Y", "Magnetometer Z"
        ]
        
        for i in range(9):
            row, col = divmod(i, 3)
            plot = pg.PlotWidget(title=titles[i])
            plot.showGrid(x=True, y=True)
            plot.setLabel('left', 'Value')
            plot.setLabel('bottom', 'Time (s)')
            plot_layout.addWidget(plot, row, col)
            self.plots.append(plot)
            self.curves.append(plot.plot(pen=pg.mkPen(color=(255, 0, 0), width=2)))
        
        # Setup timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(1)  # Update every 10 seconds
        
        # Start time
        self.start_time = time.time()
    
    def update_plots(self):
        if self.serial_port is None or not self.serial_port.is_open:
            return
        
        # Check if there's data available
        if self.serial_port.in_waiting:
            # Read all available data
            try:
                while self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    
                    # Parse the data (expected format: "Raw:accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z")
                    if line.startswith("Raw:"):
                        values = line[4:].split(',')
                        if len(values) == 9:
                            current_time = time.time() - self.start_time
                            self.timestamps.append(current_time)
                            
                            # Add accelerometer data
                            self.accel_data[0].append(float(values[0]))  # x
                            self.accel_data[1].append(float(values[1]))  # y
                            self.accel_data[2].append(float(values[2]))  # z
                            
                            # Add gyroscope data
                            self.gyro_data[0].append(float(values[3]))  # x
                            self.gyro_data[1].append(float(values[4]))  # y
                            self.gyro_data[2].append(float(values[5]))  # z
                            
                            # Add magnetometer data
                            self.mag_data[0].append(float(values[6]))  # x
                            self.mag_data[1].append(float(values[7]))  # y
                            self.mag_data[2].append(float(values[8]))  # z
                            
                            # Limit data points
                            if len(self.timestamps) > self.max_points:
                                self.timestamps = self.timestamps[-self.max_points:]
                                for i in range(3):
                                    self.accel_data[i] = self.accel_data[i][-self.max_points:]
                                    self.gyro_data[i] = self.gyro_data[i][-self.max_points:]
                                    self.mag_data[i] = self.mag_data[i][-self.max_points:]
                
                # Update plots after reading all available data
                self.update_plot_data()
            except Exception as e:
                print(f"Error reading/processing serial data: {e}")
    
    def update_plot_data(self):
        # Only update if we have data
        if not self.timestamps:
            return
        
        # Update accelerometer plots
        for i in range(3):
            self.curves[i].setData(self.timestamps, self.accel_data[i])
        
        # Update gyroscope plots
        for i in range(3):
            self.curves[i + 3].setData(self.timestamps, self.gyro_data[i])
        
        # Update magnetometer plots
        for i in range(3):
            self.curves[i + 6].setData(self.timestamps, self.mag_data[i])
    
    def closeEvent(self, event):
        # Close serial port when application closes
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("Serial port closed")
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NineDOFPlotter()
    window.show()
    sys.exit(app.exec())
