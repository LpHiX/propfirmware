import numpy as np
import serial
import time
import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtGui
from scipy.spatial.transform import Rotation
from scipy.linalg import cholesky  # Add this import for the UKF

class UKF:
    def __init__(self, n_states, n_measurements, process_noise_matrix, measurement_noise_matrix, alpha=0.001, beta=2.0, kappa=0):
        self.n = n_states # State vector dimension
        self.m = n_measurements # Measurement vector dimension
        self.x = np.zeros(self.n) # State vector
        self.P = np.eye(self.n)  # State covariance matrix
        self.Q = process_noise_matrix
        self.R = measurement_noise_matrix
        
        # Common parameters for UKF https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambd = alpha**2 * (self.n + kappa) - self.n
        
        self.Wm = np.full(2 * self.n + 1, 0.5 / (self.n + self.lambd))
        self.Wc = np.full(2 * self.n + 1, 0.5 / (self.n + self.lambd))
        self.Wm[0] = self.lambd / (self.n + self.lambd)
        self.Wc[0] = (self.lambd / (self.n + self.lambd)) + (1 - self.alpha**2 + self.beta)


    def generate_sigma_points(self):
        self.sigma_points = np.zeros((2 * self.n + 1, self.n))
        S = cholesky((self.n + self.lambd) * self.P)
        self.sigma_points[0] = self.x
        for i in range(self.n):
            self.sigma_points[i + 1] = self.x + S[i]
            self.sigma_points[i + 1 + self.n] = self.x - S[i]
        return self.sigma_points

    def predict(self, process_model, dt):
        sigma_points = self.generate_sigma_points()
        transformed_sigma_points = np.zeros_like(sigma_points)
        for i in range(2 * self.n + 1):
            transformed_sigma_points[i] = process_model(sigma_points[i], dt)

        x_pred = np.dot(self.Wm, transformed_sigma_points)
        P_pred = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = transformed_sigma_points[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)

        # Add process noise covariance
        P_pred += self.Q
        self.x = x_pred
        self.P = P_pred
        self.sigma_points = transformed_sigma_points
    
    def update(self, measurement_model, measurement):
        transformed_sigma_points = np.zeros((2 * self.n + 1, self.m))
        for i in range(2 * self.n + 1):
            transformed_sigma_points[i] = measurement_model(self.sigma_points[i])

        y_pred = np.dot(self.Wm, transformed_sigma_points)
        Pyy = np.zeros((self.m, self.m))
        for i in range(2 * self.n + 1):
            diff = transformed_sigma_points[i] - y_pred
            Pyy += self.Wc[i] * np.outer(diff, diff)

        # Add measurement noise covariance
        Pyy += self.R

        # Cross-covariance matrix
        Pxy = np.zeros((self.n, self.m))
        for i in range(2 * self.n + 1):
            diff_x = self.sigma_points[i] - self.x
            diff_y = transformed_sigma_points[i] - y_pred
            Pxy += self.Wc[i] * np.outer(diff_x, diff_y)

        # Ensure numerical stability with robust matrix inversion
        try:
            # Add small regularization to ensure invertibility
            Pyy_reg = Pyy + np.eye(self.m) * 1e-6
            K = Pxy @ np.linalg.inv(Pyy_reg)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse for stability
            K = Pxy @ np.linalg.pinv(Pyy)

        # Update state and covariance
        self.x += K @ (measurement - y_pred)
        
        # Joseph form of covariance update (more numerically stable)
        I_KH = np.eye(self.n) - K @ Pxy.T @ np.linalg.inv(self.P)
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Ensure the covariance matrix stays symmetric and positive definite
        self.P = (self.P + self.P.T) / 2
        
        # Enforce minimum eigenvalues to prevent covariance collapse
        eigvals, eigvecs = np.linalg.eigh(self.P)
        min_eig = 1e-6
        eigvals = np.maximum(eigvals, min_eig)
        self.P = eigvecs @ np.diag(eigvals) @ eigvecs.T

class HopperStateEstimator:
    def __init__(self):
        # Define state vector:
        # [x, y, z, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz]
        # where:
        # - x,y,z: position in ENU frame (meters)
        # - vx,vy,vz: velocity in ENU frame (m/s)
        # - qx,qy,qz,qw: quaternion representing orientation (SCALAR-LAST format)
        # - wx,wy,wz: angular velocity (rad/s)
        self.n_states = 13

        # Define process noise covariance matrix (Q) (THIS NEEDS TUNING)
        self.Q = np.eye(self.n_states) * 0.01
        self.Q[0:3, 0:3] *= 0.001 # Position noise is smaller (GPT said so)
        self.Q[6:10, 6:10] *= 0.001 # Quaternion noise is smaller (GPT said so)

        # Define measurement vector:
        # accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z,
        self.n_measurements = 9

        # Define measurement noise covariance matrix (R) (THIS NEEDS TUNING)
        self.R = np.eye(self.n_measurements)
        self.R[0:3, 0:3] *= 0.1 #Accelerometer noise (in m/s^2)
        self.R[3:6, 3:6] *= 0.01 #Gyro noise (in rad/s)
        self.R[6:9, 6:9] *= 0.01 #Magnetometer noise (in uT)

        self.ukf = UKF(self.n_states, self.n_measurements, self.Q, self.R)

        self.ukf.x = np.zeros(self.n_states) # Initialize state vector to zeros
        self.ukf.x[9] = 1.0 # Set initial quaternion to identity (qw=1, scalar-last)

        self.ukf.P = np.eye(self.n_states) * 10 # Initialize covariance matrix (High uncertainty)

    
    def process_model(self, state, dt):
        """Process model for UKF prediction step - Quaternion based orientation estimation"""
        new_state = np.copy(state)

        # Extract state variables
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        qx, qy, qz, qw = state[6:10]  # Scalar-last format
        wx, wy, wz = state[10:13]

        # Normalize quaternion to prevent accumulation of numerical errors
        quat_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        qx, qy, qz, qw = qx/quat_norm, qy/quat_norm, qz/quat_norm, qw/quat_norm
        
        # 1. Update position based on velocity
        new_state[0:3] = state[0:3] + state[3:6] * dt
        
        # 2. Update quaternion based on angular velocity
        # This is the key to stable orientation estimation
        if np.linalg.norm([wx, wy, wz]) > 1e-8:  # Only update if there's meaningful rotation
            # Convert angular velocity to quaternion derivative
            wx_half_dt = 0.5 * wx * dt
            wy_half_dt = 0.5 * wy * dt
            wz_half_dt = 0.5 * wz * dt
            
            # Quaternion update formula based on integration of quaternion derivative
            new_qw = qw - wx_half_dt*qx - wy_half_dt*qy - wz_half_dt*qz
            new_qx = qx + wx_half_dt*qw - wy_half_dt*qz + wz_half_dt*qy
            new_qy = qy + wx_half_dt*qz + wy_half_dt*qw - wz_half_dt*qx
            new_qz = qz - wx_half_dt*qy + wy_half_dt*qx + wz_half_dt*qw
        else:
            # No rotation, keep quaternion the same
            new_qw, new_qx, new_qy, new_qz = qw, qx, qy, qz
        
        # Normalize the resulting quaternion
        q_norm = np.sqrt(new_qw*new_qw + new_qx*new_qx + new_qy*new_qy + new_qz*new_qz)
        new_state[6:10] = np.array([new_qx, new_qy, new_qz, new_qw]) / q_norm
        
        # 3. Update velocity - we keep it simple since most IMU fusion cares mainly about orientation
        # For a stationary or slow-moving platform with no external forces, velocity stays close to zero
        # Slight decay factor prevents velocity drift
        decay_factor = 0.95  # Velocity decay factor (slight damping)
        new_state[3:6] = state[3:6] #* decay_factor
        
        # 4. Update angular velocity - also add damping to prevent wild gyro predictions
        # For most consumer IMUs, angular velocity changes relatively slowly
        angular_decay = 0.9  # Angular velocity decay factor
        new_state[10:13] = state[10:13] #* angular_decay
        
        return new_state
    
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions in scalar-last format"""
        x1, y1, z1, w1 = q1  # Unpack as scalar-last (x,y,z,w)
        x2, y2, z2, w2 = q2
        
        # Quaternion multiplication formula for scalar-last format
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        
        return np.array([x, y, z, w])  # Return as scalar-last

    def measurement_model(self, state):
        """Convert state vector to expected sensor measurements"""
        # Extract state components
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        qx, qy, qz, qw = state[6:10]  # Scalar-last format
        wx, wy, wz = state[10:13]

        # Normalize quaternion to ensure valid rotation matrix
        quat = np.array([qx, qy, qz, qw])
        quat = quat / np.linalg.norm(quat)
        
        # Create rotation from quaternion
        rot = Rotation.from_quat(quat)  # In scalar-last format
        R_b2e = rot.as_matrix()  # Rotation matrix from body to ENU
        R_e2b = R_b2e.T  # Rotation matrix from ENU to body

        # Initialize measurement vector [ax, ay, az, gx, gy, gz, mx, my, mz]
        measurement = np.zeros(self.n_measurements)

        # 1. Expected accelerometer measurements (body frame)
        # In a static or slow-moving case, accelerometer measures gravity vector
        g_enu = np.array([0, 0, 9.81])  # Gravity vector in ENU frame (positive Z up)
        accel_body = R_e2b @ g_enu  # Transform to body frame
        
        # Add contribution from linear acceleration of the body (a = v_dot)
        # This is important for dynamic cases, but could be muted for more stability
        linear_accel_factor = 0.2  # Scale factor - reduce impact of velocity changes (0.0 to ignore completely)
        linear_accel_enu = np.array([vx, vy, vz]) * linear_accel_factor
        linear_accel_body = R_e2b @ linear_accel_enu
        
        measurement[0:3] = accel_body - linear_accel_body  # Note: subtract because acceleration due to motion opposes gravity
        
        # 2. Expected gyroscope measurements (body frame angular velocities)
        # Gyro directly measures angular velocity in the body frame
        # Add a small low-pass filter effect for prediction stability
        measurement[3:6] = state[10:13]  # Use the angular velocity state directly
        
        # 3. Expected magnetometer measurements (body frame)
        # Ideally we would use the actual local magnetic field model
        # For most locations this is a reasonable approximation for North Hemisphere
        # Adjust the inclination angle (67°) based on your geographical location
        mag_field_enu = np.array([22.0, 5.5, 42.2])  # Adjusted based on typical Northern Hemisphere field components
        # Normalize to remove magnitude dependency, since we care only about direction
        mag_field_enu = mag_field_enu / np.linalg.norm(mag_field_enu)
        
        # Transform to body frame 
        mag_field_body = R_e2b @ mag_field_enu  # Transform to body frame
        measurement[6:9] = mag_field_body
        
        return measurement
    
    def predict(self, dt):
        """Predict state forward in time"""
        self.ukf.predict(self.process_model, dt)
    
    def update(self, accel, gyro, mag):
        """Update state with sensor measurements"""
        z = np.concatenate([accel, gyro, mag])
        self.ukf.update(self.measurement_model, z)
        
        return self.ukf.x


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
        
        
        # Initialize UKF for sensor fusion
        self.initialize_ukf()
        
        # Setup UI
        self.setup_ui()
        
        # Serial port parameters
        self.ser = None
        self.max_retry_attempts = 5
        self.retry_count = 0
        self.retry_delay = 2  # seconds
        
        # Start data collection
        self.start_serial_connection()
    
    def initialize_ukf(self):
        """Initialize the UKF state estimator with better parameters"""
        self.state_estimator = HopperStateEstimator()
        
        # Initialize with identity quaternion orientation
        self.state_estimator.ukf.x[6:9] = 0.0  # qx, qy, qz = 0
        self.state_estimator.ukf.x[9] = 1.0    # qw = 1
        
        # Better tuning of UKF parameters - crucial for stable performance
        # Process noise matrix tuning
        self.state_estimator.ukf.Q = np.eye(self.state_estimator.n_states) * 0.001  # Start with low process noise
        self.state_estimator.ukf.Q[0:3, 0:3] *= 0.0001  # Very low position process noise (position changes slowly)
        self.state_estimator.ukf.Q[3:6, 3:6] *= 0.001   # Low velocity process noise
        self.state_estimator.ukf.Q[6:10, 6:10] *= 0.01  # Moderate quaternion process noise
        self.state_estimator.ukf.Q[10:13, 10:13] *= 0.1 # Higher angular velocity process noise
        
        # Measurement noise matrix tuning - critical for stability
        self.state_estimator.ukf.R = np.eye(self.state_estimator.n_measurements)
        self.state_estimator.ukf.R[0:3, 0:3] *= 10.0     # High accelerometer noise (they're noisy in practice)
        self.state_estimator.ukf.R[3:6, 3:6] *= 0.1      # Lower gyro noise (gyros are more reliable)
        self.state_estimator.ukf.R[6:9, 6:9] *= 20.0     # Very high magnetometer noise (mag is least reliable)
        
        # UKF alpha parameter - smaller means sigma points are closer to mean
        # This can help stability with non-linear systems
        self.state_estimator.ukf.alpha = 0.1
        
        # Set initial state uncertainties
        self.state_estimator.ukf.P = np.eye(self.state_estimator.n_states)
        self.state_estimator.ukf.P[0:3, 0:3] *= 0.1      # Low position uncertainty (we know we start at origin)
        self.state_estimator.ukf.P[3:6, 3:6] *= 0.1      # Low velocity uncertainty (we know we start stationary)
        self.state_estimator.ukf.P[6:10, 6:10] *= 0.001  # Very low quaternion uncertainty (we start at identity)
        self.state_estimator.ukf.P[10:13, 10:13] *= 0.1  # Low angular velocity uncertainty

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
        
        # Apply calibration offsets for better accuracy (commonly needed for consumer IMUs)
        # Gyroscope bias compensation - helps with drift
        gyro_bias = np.array([0.0, 0.0, 0.0])  # Adjust these values based on your sensor calibration
        gyro_values = gyro_values - gyro_bias
        
        # Apply threshold filter to gyro (ignore very small movements that are likely just noise)
        gyro_threshold = 0.01  # radians/sec
        gyro_values = np.where(np.abs(gyro_values) < gyro_threshold, 0.0, gyro_values)
        
        # Append new data points to history
        self.accel_data = np.vstack((self.accel_data, [accel_values]))
        self.gyro_data = np.vstack((self.gyro_data, [gyro_values]))
        if len(values) >= 9:
            self.mag_data = np.vstack((self.mag_data, [mag_values]))
        
        # # Apply moving average filter to sensor data for smoothing
        # window_size = 3
        # if len(self.accel_data) >= window_size:
        #     accel_values = np.mean(self.accel_data[-window_size:], axis=0)
        #     gyro_values = np.mean(self.gyro_data[-window_size:], axis=0)
        #     if len(self.mag_data) >= window_size:
        #         mag_values = np.mean(self.mag_data[-window_size:], axis=0)
        
        # Update UKF state estimation
        if dt > 0 and dt < 0.1:  # More restrictive sanity check on time delta
            try:
                # Predict state forward
                self.state_estimator.predict(dt)
                
                # Update with measurements
                state = self.state_estimator.update(accel_values, gyro_values, mag_values)
                
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
                self.log_message(f"UKF update error: {e}")
    
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