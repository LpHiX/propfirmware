import numpy as np
import serial
import time
import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui

class IMUCalibration(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Remove the line disabling OpenGL to enable 3D plotting
        # pg.setConfigOptions(useOpenGL=False)
        
        # Data storage
        self.accel_data = np.empty((0, 3))
        self.gyro_data = np.empty((0, 3))
        self.mag_data = np.empty((0, 3))
        
        # Setup UI
        self.setup_ui()
        
        # Serial port parameters
        self.ser = None
        self.max_retry_attempts = 5
        self.retry_count = 0
        self.retry_delay = 2  # seconds
        self.update_freq = 1  # Only update visualization every N points
        self.point_count = 0
        self.max_points = 5000  # Maximum points to display
        
        # Start data collection
        self.start_serial_connection()

    def setup_ui(self):
        self.setWindowTitle('IMU Calibration')
        self.setGeometry(100, 100, 1200, 800)  # Increased window size
        
        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout (vertical)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Upper section for plots (grid layout)
        plot_layout = QtWidgets.QGridLayout()
        main_layout.addLayout(plot_layout)
        
        # Create 3D GLViewWidget for scatter plot (upper left)
        self.plot_widget = gl.GLViewWidget()
        plot_layout.addWidget(self.plot_widget, 0, 0)
        
        # Add grid and axes for 3D orientation
        grid = gl.GLGridItem()
        grid.setSize(x=3000, y=3000, z=1)
        grid.setSpacing(x=200, y=200, z=10)
        self.plot_widget.addItem(grid)
        
        # Add X, Y, Z axes for reference
        axis_size = 100
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_size, 0, 0]]), color=(1, 0, 0, 1), width=2)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_size, 0]]), color=(0, 1, 0, 1), width=2)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_size]]), color=(0, 0, 1, 1), width=2)
        self.plot_widget.addItem(x_axis)
        self.plot_widget.addItem(y_axis)
        self.plot_widget.addItem(z_axis)
        
        # Create 3D scatter plot for magnetometer data
        self.scatter = gl.GLScatterPlotItem(pos=np.empty((0, 3)), color=(1, 0, 0, 1), size=5)
        self.plot_widget.addItem(self.scatter)
        
        # Create 2D projection plots
        # XY projection (upper right)
        self.xy_plot = pg.PlotWidget(title="XY Projection")
        self.xy_plot.setLabel('left', 'Y')
        self.xy_plot.setLabel('bottom', 'X')
        self.xy_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(255, 0, 0, 120), size=5)
        self.xy_plot.addItem(self.xy_scatter)
        self.xy_plot.setAspectLocked()
        plot_layout.addWidget(self.xy_plot, 0, 1)
        
        # XZ projection (lower left)
        self.xz_plot = pg.PlotWidget(title="XZ Projection")
        self.xz_plot.setLabel('left', 'Z')
        self.xz_plot.setLabel('bottom', 'X')
        self.xz_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(0, 255, 0, 120), size=5)
        self.xz_plot.addItem(self.xz_scatter)
        self.xz_plot.setAspectLocked()
        plot_layout.addWidget(self.xz_plot, 1, 0)
        
        # YZ projection (lower right)
        self.yz_plot = pg.PlotWidget(title="YZ Projection")
        self.yz_plot.setLabel('left', 'Z')
        self.yz_plot.setLabel('bottom', 'Y')
        self.yz_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(0, 0, 255, 120), size=5)
        self.yz_plot.addItem(self.yz_scatter)
        self.yz_plot.setAspectLocked()
        plot_layout.addWidget(self.yz_plot, 1, 1)
        
        # Status display at bottom
        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        main_layout.addWidget(self.status_text)
        
        # Setup update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # Update every 50ms
        
        # Show initial window
        self.show()

    def log_message(self, message):
        """Add message to status display"""
        self.status_text.append(message)
        #print(message)
        
    def start_serial_connection(self):
        """Start background thread for serial connection"""
        self.serial_thread = SerialThread(self)
        self.serial_thread.new_data_signal.connect(self.process_data)
        self.serial_thread.log_signal.connect(self.log_message)
        self.serial_thread.start()
        
    def process_data(self, values):
        """Process new data received from serial port"""
        # Append new data points
        self.accel_data = np.vstack((self.accel_data, [values[0:3]]))
        self.gyro_data = np.vstack((self.gyro_data, [values[3:6]]))
        self.mag_data = np.vstack((self.mag_data, [values[6:9]]))
        
        # Limit maximum number of points to prevent slowdown
        if len(self.mag_data) > self.max_points:
            self.accel_data = self.accel_data[-self.max_points:]
            self.gyro_data = self.gyro_data[-self.max_points:]
            self.mag_data = self.mag_data[-self.max_points:]
    
    def update_plot(self):
        """Update the visualization"""
        if len(self.mag_data) > 0:
            # Update the 3D scatter plot of the magnetometer data
            self.scatter.setData(pos=self.mag_data)
            
            # Update 2D projections
            # Extract x, y, z coordinates for easier processing
            x_data = self.mag_data[:, 0]
            y_data = self.mag_data[:, 1]
            z_data = self.mag_data[:, 2]
            
            # Update XY projection
            self.xy_scatter.setData(x=x_data, y=y_data)
            
            # Update XZ projection
            self.xz_scatter.setData(x=x_data, y=z_data)
            
            # Update YZ projection
            self.yz_scatter.setData(x=y_data, y=z_data)
            
            # Update plot limits if needed
            if len(self.mag_data) > 1:
                # Calculate min/max values for each axis
                xmin, xmax = np.min(x_data), np.max(x_data)
                ymin, ymax = np.min(y_data), np.max(y_data)
                zmin, zmax = np.min(z_data), np.max(z_data)


                self.log_message(f'compass.setCalibration({xmin},{xmax},{ymin},{ymax},{zmin},{zmax});')
                # heading = self.calculate_tilt_compensated_heading()
                # self.log_message(f"Heading: {heading:.1f}° (tilt-compensated)")

                # Add some margin to the limits
                margin = 0.1
                x_margin = margin * (xmax - xmin)
                y_margin = margin * (ymax - ymin)
                z_margin = margin * (zmax - zmin)
                
                # # Set ranges for 2D plots
                # self.xy_plot.setRange(xRange=[xmin-x_margin, xmax+x_margin], 
                #                       yRange=[ymin-y_margin, ymax+y_margin])
                # self.xz_plot.setRange(xRange=[xmin-x_margin, xmax+x_margin], 
                #                       yRange=[zmin-z_margin, zmax+z_margin])
                # self.yz_plot.setRange(xRange=[ymin-y_margin, ymax+y_margin], 
                #                       yRange=[zmin-z_margin, zmax+z_margin])
                
                # Calculate center point of data for 3D view
                center_x = (xmax + xmin) / 2
                center_y = (ymax + ymin) / 2
                center_z = (zmax + zmin) / 2
                
                # Calculate appropriate distance based on data span
                x_span = xmax - xmin 
                y_span = ymax - ymin
                z_span = zmax - zmin
                distance = max(x_span, y_span, z_span) * 1.5  # Add some margin
                
                # Set camera position to view all data
                self.plot_widget.setCameraPosition(pos=QtGui.QVector3D(center_x, center_y, center_z + distance))
                self.plot_widget.opts['center'] = QtGui.QVector3D(center_x, center_y, center_z)

    def calculate_tilt_compensated_heading(self):
        """Calculate tilt-compensated magnetic heading"""
        if len(self.mag_data) == 0 or len(self.accel_data) == 0:
            return None
            
        # Get latest magnetometer and accelerometer readings
        mag = self.mag_data[-1]
        accel = self.accel_data[-1]
        
        # Normalize accelerometer data (gives us gravity direction)
        accel_norm = accel / np.linalg.norm(accel)
        
        # Calculate pitch and roll from accelerometer
        # (Device coordinate system: X forward, Y right, Z down)
        pitch = np.arcsin(-accel_norm[0])
        roll = np.arcsin(accel_norm[1] / np.cos(pitch))
        
        # Tilt compensated magnetic field X and Y components
        mag_x = mag[0] * np.cos(pitch) + mag[2] * np.sin(pitch)
        mag_y = (mag[0] * np.sin(roll) * np.sin(pitch) + 
                 mag[1] * np.cos(roll) - 
                 mag[2] * np.sin(roll) * np.cos(pitch))
        
        # Calculate heading
        heading = np.arctan2(mag_y, mag_x) * 180 / np.pi
        
        # Convert to 0-360 range
        heading = (heading + 360) % 360
        
        return heading

class SerialThread(QtCore.QThread):
    new_data_signal = QtCore.pyqtSignal(list)
    log_signal = QtCore.pyqtSignal(str)
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        
    def run(self):
        max_retry_attempts = 5
        retry_count = 0
        retry_delay = 2  # seconds
        
        while retry_count < max_retry_attempts:
            ser = None
            try:
                self.log_signal.emit(f"Opening serial port... (attempt {retry_count+1}/{max_retry_attempts})")
                ser = serial.Serial('COM8', 115200, timeout=1)
                self.log_signal.emit("Serial port opened, waiting for data")
                retry_count = 0  # Reset count on successful connection
                
                while True:
                    line = ser.readline().decode('utf-8').strip()
                    if not line.startswith("Raw:"):
                        continue
                        
                    try: 
                        data_part = line[4:].strip()
                        values = [float(x) for x in data_part.split(",")]
                        if len(values) != 9:
                            continue
                        
                        # Emit new data signal
                        self.new_data_signal.emit(values)
                        
                        # Print current data point
                        
                    except Exception as e:
                        self.log_signal.emit(f"Error parsing line: {line} - {e}")
                        continue
                        
            except serial.SerialException as e:
                self.log_signal.emit(f"Serial error: {e}")
                retry_count += 1
                self.log_signal.emit(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            except Exception as e:
                self.log_signal.emit(f"Unexpected error: {e}")
                break
            finally:
                if ser is not None and ser.is_open:
                    ser.close()
                    self.log_signal.emit("Serial port closed.")
        
        # Only reached if we've exceeded max retries
        self.log_signal.emit("Maximum reconnection attempts reached or program terminated.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = IMUCalibration()
    sys.exit(app.exec_())