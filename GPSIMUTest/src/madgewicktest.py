import numpy as np
import serial
import time
import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
from scipy.spatial.transform import Rotation

class MadgwickFilter:
    def __init__(self, beta=0.1, sample_freq=100.0):
        # Filter gain (beta)
        self.beta = beta
        # Sampling frequency in Hz
        self.sample_freq = sample_freq
        # Quaternion of sensor frame relative to earth frame (scalar last)
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        # Time tracking
        self.last_update = time.time()

    def update(self, gyro, accel, mag=None, dt=None):
        """
        Update orientation with gyroscope, accelerometer, and magnetometer data
        
        Args:
            gyro: gyroscope measurements in rad/s (x, y, z)
            accel: accelerometer measurements in m/s^2 (x, y, z)
            mag: magnetometer measurements in arbitrary units (x, y, z), optional
            dt: time step in seconds (optional)
        
        Returns:
            Quaternion representing the updated orientation (scalar last)
        """
        # Calculate time step if not provided
        if dt is None:
            now = time.time()
            dt = now - self.last_update
            self.last_update = now
            
        # Use provided sample frequency if dt is too small or zero
        if dt <= 0:
            print("invalid dt")
            dt = 1.0 / self.sample_freq
        
        # Normalize accelerometer measurement
        if np.linalg.norm(accel) > 0:
            accel = accel / np.linalg.norm(accel)
        
        # Use magnetometer data if available
        if mag is not None and np.linalg.norm(mag) > 0:
            mag = mag / np.linalg.norm(mag)
            return self._update_with_mag(gyro, accel, mag, dt)
        else:
            return self._update_with_accel(gyro, accel, dt)
    
    def _update_with_accel(self, gyro, accel, dt):
        """Update using gyroscope and accelerometer only"""
        q = self.q  # short name local variable for readability
        
        # Auxiliary variables to avoid repeated calculations
        q1, q2, q3, q0 = q[0], q[1], q[2], q[3]
        
        # Gradient descent algorithm corrective step
        # Modified objective function for "Z is up" convention (note the sign change)
        f = np.array([
            2*(q1*q3 - q0*q2) - accel[0],
            2*(q0*q1 + q2*q3) - accel[1],
            2*(0.5 - q1*q1 - q2*q2) + accel[2]  # Changed sign here from - to +
        ])
        
        J = np.array([
            [-2*q2,  2*q3, -2*q0,  2*q1],
            [ 2*q1,  2*q0,  2*q3,  2*q2],
            [    0, -4*q1, -4*q2,     0]
        ])
        
        # Normalize gradient
        gradient = J.T @ f
        if np.linalg.norm(gradient) > 0:
            gradient = gradient / np.linalg.norm(gradient)
        
        # Rate of change of quaternion from gyroscope
        qDot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])
        
        # Apply feedback step
        qDot -= self.beta * gradient
        
        # Integrate to yield quaternion
        q += qDot * dt
        
        # Normalize quaternion
        if np.linalg.norm(q) > 0:
            q = q / np.linalg.norm(q)
        
        self.q = q
        return q
    
    def _update_with_mag(self, gyro, accel, mag, dt):
        """Update using gyroscope, accelerometer and magnetometer"""
        q = self.q  # short name local variable for readability
        
        # Auxiliary variables to avoid repeated calculations
        q1, q2, q3, q0 = q[0], q[1], q[2], q[3]
        
        # Reference direction of Earth's magnetic field
        h = np.array([
            2*mag[0]*(0.5 - q2*q2 - q3*q3) + 2*mag[1]*(q1*q2 - q0*q3) + 2*mag[2]*(q1*q3 + q0*q2),
            2*mag[0]*(q1*q2 + q0*q3) + 2*mag[1]*(0.5 - q1*q1 - q3*q3) + 2*mag[2]*(q2*q3 - q0*q1),
            2*mag[0]*(q1*q3 - q0*q2) + 2*mag[1]*(q2*q3 + q0*q1) + 2*mag[2]*(0.5 - q1*q1 - q2*q2)
        ])
        
        b = np.array([np.sqrt(h[0]*h[0] + h[1]*h[1]), 0, h[2]])
        
        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q1*q3 - q0*q2) - accel[0],
            2*(q0*q1 + q2*q3) - accel[1],
            2*(0.5 - q1*q1 - q2*q2) + accel[2],  # Changed sign here from - to +
            2*b[0]*(0.5 - q2*q2 - q3*q3) + 2*b[2]*(q1*q3 - q0*q2) - mag[0],
            2*b[0]*(q1*q2 - q0*q3) + 2*b[2]*(q0*q1 + q2*q3) - mag[1],
            2*b[0]*(q0*q2 + q1*q3) + 2*b[2]*(0.5 - q1*q1 - q2*q2) - mag[2]
        ])
        
        J = np.array([
            [-2*q2,                 2*q3,                  -2*q0,                  2*q1],
            [ 2*q1,                 2*q0,                   2*q3,                  2*q2],
            [    0,                -4*q1,                  -4*q2,                     0],
            [-2*b[2]*q2,      2*b[2]*q3,    -4*b[0]*q2-2*b[2]*q0,  -4*b[0]*q3+2*b[2]*q1],
            [-2*b[0]*q3+2*b[2]*q1, -2*b[0]*q2-2*b[2]*q0,  2*b[0]*q1+2*b[2]*q3,  -2*b[0]*q0+2*b[2]*q2],
            [2*b[0]*q2,           2*b[0]*q3-4*b[2]*q1,     2*b[0]*q0-4*b[2]*q2,     2*b[0]*q1]
        ])
        
        # Normalize gradient
        gradient = J.T @ f
        if np.linalg.norm(gradient) > 0:
            gradient = gradient / np.linalg.norm(gradient)
        
        # Rate of change of quaternion from gyroscope
        qDot = 0.5 * np.array([
            -q1*gyro[0] - q2*gyro[1] - q3*gyro[2],
            q0*gyro[0] + q2*gyro[2] - q3*gyro[1],
            q0*gyro[1] - q1*gyro[2] + q3*gyro[0],
            q0*gyro[2] + q1*gyro[1] - q2*gyro[0]
        ])
        
        # Apply feedback step
        qDot -= self.beta * gradient
        
        # Integrate to yield quaternion
        q += qDot * dt
        
        # Normalize quaternion
        if np.linalg.norm(q) > 0:
            q = q / np.linalg.norm(q)
        
        self.q = q
        return q

class MadgwickStateEstimator:
    def __init__(self, beta=0.1, sample_freq=100.0):
        """
        Initialize Madgwick-based state estimator
        
        Args:
            beta: Filter gain parameter (default: 0.1)
            sample_freq: Expected sample frequency in Hz (default: 100.0)
        """
        # Initialize Madgwick filter
        self.madgwick = MadgwickFilter(beta, sample_freq)
        
        # State vector: position (3), velocity (3), orientation (4), angular velocity (3)
        self.state = np.zeros(13)
        self.state[9] = 1.0  # Initial quaternion w component (scalar last format)
        
        # Additional variables for tracking
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        
    def update(self, accel, gyro, mag=None, dt=None):
        """
        Update state estimation with new sensor readings
        """
        # Initialize with identity quaternion if needed
        if np.allclose(self.madgwick.q, [0, 0, 0, 1]):
            # Calculate initial orientation based on accelerometer
            # This assumes the device is approximately level
            norm_accel = accel / np.linalg.norm(accel)
            
            # Create rotation from [0,0,1] to measured gravity direction
            if norm_accel[2] > 0.9:  # If mostly pointing up already
                self.madgwick.q = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                # Calculate rotation from [0,0,1] to current accel direction
                v = np.cross([0, 0, 1], norm_accel)
                s = np.linalg.norm(v)
                c = np.dot([0, 0, 1], norm_accel)
                
                if s > 0.001:  # Avoid division by zero
                    v = v / s
                    self.madgwick.q = np.array([
                        v[0] * np.sin(np.arccos(c) / 2),
                        v[1] * np.sin(np.arccos(c) / 2),
                        v[2] * np.sin(np.arccos(c) / 2),
                        np.cos(np.arccos(c) / 2)
                    ])
        
        # Run Madgwick with original inputs
        quaternion = self.madgwick.update(gyro, accel, mag, dt)
        
        # Store quaternion in state vector (scalar last format)
        self.state[6:10] = quaternion
        
        # Store gyro values in the state
        self.state[10:13] = gyro
        
        # Simple integration for position and velocity (if needed)
        if dt is not None:
            # Use orientation to transform acceleration
            rot = Rotation.from_quat(quaternion)
            gravity_direction = rot.apply([0, 0, 9.81])
            #print(f"Gravity direction: {gravity_direction}")
            accel_earth = rot.apply(accel)
            accel_earth[2] -= 9.81  # Remove gravity (assuming Z is up)
            
            # Simple threshold to reduce noise
            accel_threshold = 0.1  # m/s²
            accel_earth = np.where(np.abs(accel_earth) < accel_threshold, 0, accel_earth)
            
            # Update velocity and position with damping
            self.velocity += accel_earth * dt
            self.velocity *= 0.95  # Apply damping to prevent drift
            self.position += self.velocity * dt
            
            # Update state vector
            self.state[0:3] = self.position
            self.state[3:6] = self.velocity
        
        # Return the updated state
        return self.state
        
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions in scalar-last format [x,y,z,w]"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([x, y, z, w])

class IMUCalibration(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Data storage
        self.accel_data = np.zeros((0, 3))
        self.gyro_data = np.zeros((0, 3))
        self.mag_data = np.zeros((0, 3))
        self.last_update_time = time.time()
        
        # Initialize orientation (identity quaternion)
        self.orientation = Rotation.identity()
        
        # Initialize Madgwick filter for sensor fusion
        self.initialize_madgwick()
        
        # Setup UI
        self.setup_ui()
        
        # Serial port parameters
        self.ser = None
        self.max_retry_attempts = 5
        self.retry_count = 0
        self.retry_delay = 2  # seconds
        
        # Start data collection
        self.start_serial_connection()
    
    def initialize_madgwick(self):
        """Initialize the Madgwick state estimator"""
        # Create with beta parameter and estimated sample rate
        beta = 0.4  # Filter gain - higher values reduce lag but may increase noise
        sample_freq = 100.0  # Estimated sample frequency in Hz
        self.state_estimator = MadgwickStateEstimator(beta, sample_freq)
    
    def process_data(self, values):
        """Process new data received from serial port"""
        # Get current time for integration
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Extract sensor readings
        accel_values = values[0:3]
        gyro_values = values[3:6]
        mag_values = values[6:9] if len(values) >= 9 else np.zeros(3)
        
        # Apply calibration offsets for better accuracy
        gyro_bias = np.array([0.04, 0.0, 0.0])  # Adjust these values based on your sensor calibration
        gyro_values = gyro_values - gyro_bias
        
        # Apply threshold filter to gyro (ignore very small movements that are likely just noise)
        gyro_threshold = 0.01  # radians/sec
        gyro_values = np.where(np.abs(gyro_values) < gyro_threshold, 0.0, gyro_values)
        
        # Append new data points to history
        self.accel_data = np.vstack((self.accel_data, [accel_values]))
        self.gyro_data = np.vstack((self.gyro_data, [gyro_values]))
        if len(values) >= 9:
            self.mag_data = np.vstack((self.mag_data, [mag_values]))
        
        # Update state estimation with Madgwick filter
        if dt > 0 and dt < 0.1:  # Reasonable time step check
            try:
                # Update state with measurements
                state = self.state_estimator.update(accel_values, gyro_values, mag_values, dt)
                
                # Extract and normalize quaternion
                quat = state[6:10]
                quat = quat / np.linalg.norm(quat)  # Ensure quaternion is normalized
                
                # Update orientation
                self.orientation = Rotation.from_quat(quat)
                
                # Log quaternion values less frequently to reduce console spam
                if hasattr(self, 'log_counter'):
                    self.log_counter += 1
                else:
                    self.log_counter = 0
                
                if self.log_counter % 10 == 0:  # Log only every 10th update
                    qx, qy, qz, qw = quat
                    self.log_message(f"Quaternion: x={qx:.3f}, y={qy:.3f}, z={qz:.3f}, w={qw:.3f}")
            except Exception as e:
                self.log_message(f"Madgwick update error: {e}")
    
    def setup_ui(self):
        self.setWindowTitle('IMU Orientation')
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        plot_layout = QtWidgets.QGridLayout()
        main_layout.addLayout(plot_layout)
        
        self.plot_widget = gl.GLViewWidget()
        plot_layout.addWidget(self.plot_widget, 0, 0, 2, 2)
        
        grid = gl.GLGridItem()
        grid.setSize(x=3000, y=3000, z=1)
        grid.setSpacing(x=200, y=200, z=10)
        self.plot_widget.addItem(grid)
        
        axis_size = 100
        x_axis_ref = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_size, 0, 0]]), color=(0.5, 0, 0, 1), width=1)
        y_axis_ref = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_size, 0]]), color=(0, 0.5, 0, 1), width=1)
        z_axis_ref = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_size]]), color=(0, 0, 0.5, 1), width=1)
        self.plot_widget.addItem(x_axis_ref)
        self.plot_widget.addItem(y_axis_ref)
        self.plot_widget.addItem(z_axis_ref)
        
        self.x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [1, 0, 0]]), color=(1, 0, 0, 1), width=3)
        self.y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 1, 0]]), color=(0, 1, 0, 1), width=3)
        self.z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 1]]), color=(0, 0, 1, 1), width=3)
        self.plot_widget.addItem(self.x_axis)
        self.plot_widget.addItem(self.y_axis)
        self.plot_widget.addItem(self.z_axis)

        self.accel_data_plot = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=0.1, color=(1, 0, 1, 1), pxMode=False)
        self.mag_plot = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=0.1, color=(1, 1, 1, 1), pxMode=False)
        self.plot_widget.addItem(self.accel_data_plot)
        self.plot_widget.addItem(self.mag_plot)
        
        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        main_layout.addWidget(self.status_text)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)
        
        self.plot_widget.setCameraPosition(distance=5)
        
        self.show()

    def log_message(self, message):
        self.status_text.append(message)
        
    def start_serial_connection(self):
        self.serial_thread = SerialThread(self)
        self.serial_thread.new_data_signal.connect(self.process_data)
        self.serial_thread.log_signal.connect(self.log_message)
        self.serial_thread.start()
        
    def update_plot(self):
        if hasattr(self, 'orientation'):
            rot_matrix = self.orientation.as_matrix()
            
            x_end = np.dot(rot_matrix, np.array([1, 0, 0]))
            y_end = np.dot(rot_matrix, np.array([0, 1, 0]))
            z_end = np.dot(rot_matrix, np.array([0, 0, 1]))
            
            self.x_axis.setData(pos=np.array([[0, 0, 0], x_end]))
            self.y_axis.setData(pos=np.array([[0, 0, 0], y_end]))
            self.z_axis.setData(pos=np.array([[0, 0, 0], z_end]))

            self.accel_data_plot.setData(pos=self.accel_data[-1]/np.linalg.norm(self.accel_data[-1]), color=(1, 0, 0, 1))
            self.mag_plot.setData(pos=self.mag_data[-1]/np.linalg.norm(self.mag_data[-1]), color=(0, 1, 0, 1))
            
            euler = self.orientation.as_euler('xyz', degrees=True)
            self.log_message(f"Orientation (xyz): Roll: {euler[0]:.1f}°, Pitch: {euler[1]:.1f}°, Yaw: {euler[2]:.1f}°")

    def calculate_tilt_compensated_heading(self):
        if len(self.accel_data) == 0:
            return None
            
        accel = self.accel_data[-1]
        
        euler = self.orientation.as_euler('xyz', degrees=True)
        
        heading = euler[2]
        
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
        retry_delay = 2
        
        while retry_count < max_retry_attempts:
            ser = None
            try:
                self.log_signal.emit(f"Opening serial port... (attempt {retry_count+1}/{max_retry_attempts})")
                ser = serial.Serial('COM6', 115200, timeout=1)
                self.log_signal.emit("Serial port opened, waiting for data")
                retry_count = 0
                
                while True:
                    line = ser.readline().decode('utf-8').strip()
                    if not line.startswith("Raw:"):
                        continue
                        
                    try: 
                        data_part = line[4:].strip()
                        values = [float(x) for x in data_part.split(",")]
                        if len(values) != 9:
                            continue
                        print(line)
                        self.new_data_signal.emit(values)
                        
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
        
        self.log_signal.emit("Maximum reconnection attempts reached or program terminated.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = IMUCalibration()
    sys.exit(app.exec_())