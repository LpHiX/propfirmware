import numpy as np
import serial
import time
from collections import deque
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys

# Constants
COM_PORT = 'COM6'
BAUD_RATE = 115200
GYRO_BIAS = np.array([0.04, 0.0, -0.04])  # Given gyro bias
G = 9.81  # Gravity constant
LONDON_MAG_INCLINATION = 66  # Magnetic inclination in London (degrees)

ACCEL_OFFSETS = np.array([0.397600, -0.117200, -0.084255])
ACCEL_SCALE_FACTORS = np.array([1.002334, 0.990214, 1.021738])

# State vector representation:
# x[0:4] = quaternion (w,x,y,z)
# x[4:7] = gyro bias
# x[7:10] = velocity (vx, vy, vz) in world frame
# x[10:13] = position (px, py, pz) in world frame
# x[13:16] = acceleration (ax, ay, az) in world frame

def quaternion_normalize(q):
    """Normalize a quaternion to unit length"""
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_to_matrix(q):
    """Convert quaternion to rotation matrix"""
    q = quaternion_normalize(q)
    w, x, y, z = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees"""
    w, x, y, z = quaternion_normalize(q)
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.degrees(np.pi / 2 * np.sign(sinp))
    else:
        pitch = np.degrees(np.arcsin(sinp))
        
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))
    
    return roll, pitch, yaw

def state_transition_fn(x, dt):
    """State transition function for UKF"""
    quat = x[0:4]
    gyro_bias = x[4:7]
    velocity = x[7:10] if len(x) > 7 else np.zeros(3)
    position = x[10:13] if len(x) > 10 else np.zeros(3)
    acceleration = x[13:16] if len(x) > 13 else np.zeros(3)
    
    # Normalize quaternion
    quat = quaternion_normalize(quat)
    
    # Get corrected gyro rate (applying known bias for prediction)
    gyro_rate = GYRO_BIAS - gyro_bias
    
    # Convert to quaternion rotation
    angle = np.linalg.norm(gyro_rate) * dt
    if angle > 1e-10:
        axis = gyro_rate / np.linalg.norm(gyro_rate)
        dq = np.array([
            np.cos(angle/2),
            axis[0] * np.sin(angle/2),
            axis[1] * np.sin(angle/2),
            axis[2] * np.sin(angle/2)
        ])
    else:
        dq = np.array([1.0, 0.0, 0.0, 0.0])
    
    # Update orientation quaternion
    new_quat = quaternion_multiply(quat, dq)
    new_quat = quaternion_normalize(new_quat)
    
    # Gyro bias evolves as a random walk (remains unchanged in prediction)
    new_bias = gyro_bias
    
    # Apply natural velocity decay (damping) to model friction/drag
    damping = 0.98
    
    # Update velocity based on current acceleration and damping
    new_velocity = velocity * damping + acceleration * dt
    
    # Update position based on velocity and acceleration (second-order integration)
    new_position = position + velocity * dt + 0.5 * acceleration * dt * dt
    
    # Model acceleration as a damped process (tends to decay to zero)
    accel_damping = 0.9  # Stronger damping for acceleration
    new_acceleration = acceleration * accel_damping
    
    # Return new state
    return np.concatenate([new_quat, new_bias, new_velocity, new_position, new_acceleration])

def measurement_fn(x):
    """Measurement function for UKF - predicts sensor readings from state"""
    quat = x[0:4]
    gyro_bias = x[4:7]
    velocity = x[7:10]
    
    # Get rotation matrix from quaternion
    R = quaternion_to_matrix(quat)
    
    # Expected accelerometer reading (gravity in sensor frame)
    # In world frame, gravity is [0, 0, G]
    accel_expected = R.T @ np.array([0, 0, G])
    
    # Expected gyroscope reading (known bias - estimated bias)
    gyro_expected = GYRO_BIAS - gyro_bias
    
    # Expected magnetometer reading
    inclination_rad = np.radians(LONDON_MAG_INCLINATION)
    mag_field_world = np.array([
        np.cos(inclination_rad), 
        0, 
        -np.sin(inclination_rad)
    ])
    mag_scaling = 1500
    mag_field_world *= mag_scaling
    mag_expected = R.T @ mag_field_world
    
    # Include estimated acceleration in world frame (just zero as baseline)
    # In practice, this would be derived from your dynamics model
    accel_world_expected = np.zeros(3)
    
    # Combine all measurements
    return np.concatenate([accel_expected, gyro_expected, mag_expected, accel_world_expected])

def parse_raw_data(line):
    """Parse a line of raw sensor data"""
    if not line.startswith("Raw:"):
        return None
    
    data_part = line[4:]  # Remove "Raw:" prefix
    values = data_part.strip().split(',')
    
    if len(values) != 9:
        return None
    
    try:
        # Extract sensor readings
        accel_raw = np.array([float(values[0]), float(values[1]), float(values[2])])
        gyro = np.array([float(values[3]), float(values[4]), float(values[5])])
        mag = np.array([float(values[6]), float(values[7]), float(values[8])])
        
        accel = (accel_raw - ACCEL_OFFSETS) / ACCEL_SCALE_FACTORS
        return (accel, gyro, mag)
    except:
        return None

class PCBVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB Orientation & Position Tracking with UKF")
        self.resize(1000, 800)
        
        # Initialization timer - to allow filter to converge before tracking position/velocity
        self.start_time = time.time()
        self.init_period = 1000  # 10 seconds initialization period
        self.initialized = False
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create status display
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMaximumHeight(30)
        layout.addWidget(self.status_label)
        
        # Create 3D view widget
        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)
        
        # Create grid for reference
        grid = gl.GLGridItem()
        grid.setSize(1, 1, 1)
        grid.setSpacing(0.01, 0.01, 0.01)
        self.view.addItem(grid)
        
        # Setup coordinate system
        self.setup_world_axes()
        
        # Create PCB model
        self.pcb, self.pcb_axes = self.create_pcb_model()
        self.view.addItem(self.pcb)
        for axis in self.pcb_axes:
            self.view.addItem(axis)
        
        # Create velocity and acceleration vectors
        # self.velocity_arrow = gl.GLLinePlotItem(color=(1, 1, 0, 1), width=3)  # Yellow
        # self.accel_arrow = gl.GLLinePlotItem(color=(0.8, 0, 0.8, 1), width=3)  # Purple
        # self.view.addItem(self.velocity_arrow)
        # self.view.addItem(self.accel_arrow)
        
        # Create trail
        self.trail_points = deque(maxlen=100)
        self.trail = gl.GLLinePlotItem(color=(1, 0, 0, 1), width=2)
        self.view.addItem(self.trail)
        
        # Initialize UKF
        self.init_ukf()
        
        # Setup serial connection
        try:
            self.ser = serial.Serial(COM_PORT, BAUD_RATE)
            print(f"Connected to {COM_PORT} at {BAUD_RATE} baud")
            self.status_label.setText(f"Connected to {COM_PORT}")
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            self.status_label.setText(f"Error: {e}")
            self.ser = None
            
        # Time tracking for integration
        self.prev_time = time.time()
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_visualization)
        self.timer.start(16)  # ~60 fps
    
    def setup_world_axes(self):
        # X axis (red)
        x_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0.5, 0, 0]]), color=(1, 0, 0, 1), width=2)
        self.view.addItem(x_axis)
        
        # Y axis (green)
        y_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0.5, 0]]), color=(0, 1, 0, 1), width=2)
        self.view.addItem(y_axis)
        
        # Z axis (blue)
        z_axis = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, 0.5]]), color=(0, 0, 1, 1), width=2)
        self.view.addItem(z_axis)
    
    def create_pcb_model(self):
        # PCB dimensions in meters
        l, w, h = 0.02, 0.03, 0.01  # Length, width, height
        
        # Define box vertices
        vertices = np.array([
            [-l/2, -w/2, -h/2],  # 0: bottom-left-back
            [l/2, -w/2, -h/2],   # 1: bottom-right-back
            [l/2, w/2, -h/2],    # 2: bottom-right-front
            [-l/2, w/2, -h/2],   # 3: bottom-left-front
            [-l/2, -w/2, h/2],   # 4: top-left-back
            [l/2, -w/2, h/2],    # 5: top-right-back
            [l/2, w/2, h/2],     # 6: top-right-front
            [-l/2, w/2, h/2],    # 7: top-left-front
        ])
        
        # Define faces for the mesh
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom face
            [4, 5, 6], [4, 6, 7],  # Top face
            [0, 1, 5], [0, 5, 4],  # Back face
            [2, 3, 7], [2, 7, 6],  # Front face
            [0, 3, 7], [0, 7, 4],  # Left face
            [1, 2, 6], [1, 6, 5]   # Right face
        ])
        
        # Store faces for later use in updates
        self.pcb_faces = faces
        
        # Create mesh item for PCB
        mesh = gl.GLMeshItem(vertexes=vertices, faces=faces, smooth=False, drawEdges=True,
                            edgeColor=(0, 0, 1, 1), color=(0.5, 0.5, 1, 0.8))
        
        # Create axes for PCB
        arrow_length = 0.1
        
        # PCB axes
        pcb_x = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [arrow_length, 0, 0]]), color=(1, 0, 0, 1), width=2)
        pcb_y = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, arrow_length, 0]]), color=(0, 1, 0, 1), width=2)
        pcb_z = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, arrow_length]]), color=(0, 0, 1, 1), width=2)
        
        return mesh, [pcb_x, pcb_y, pcb_z]
    
    def init_ukf(self):
        # Initialize UKF
        n_state = 16  # 4 quaternion + 3 gyro bias + 3 velocity + 3 position + 3 acceleration
        n_measurement = 12  # 3 accel + 3 gyro + 3 mag + 3 accel_world
        
        # Create sigma points
        points = MerweScaledSigmaPoints(n=n_state, alpha=0.01, beta=2., kappa=-1)
        
        # Initialize filter
        self.ukf = UnscentedKalmanFilter(dim_x=n_state, dim_z=n_measurement, dt=0.01, 
                                        fx=state_transition_fn, hx=measurement_fn, points=points)
        
        # Initial state (identity quaternion, zero gyro bias, zero velocity, zero position, zero acceleration)
        self.ukf.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # State covariance
        self.ukf.P = np.diag([0.1, 0.1, 0.1, 0.1,  # quaternion
                              0.01, 0.01, 0.01,    # gyro bias
                              0.1, 0.1, 0.1,       # velocity
                              0.1, 0.1, 0.1,       # position
                              0.5, 0.5, 0.5])      # acceleration (higher uncertainty)
        
        # Process noise
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.0001, 0.0001,  # quaternion
                              0.0001, 0.0001, 0.0001,          # gyro bias
                              0.01, 0.01, 0.01,                # velocity
                              0.01, 0.01, 0.01,                # position
                              0.1, 0.1, 0.1])                  # acceleration (higher process noise)
        
        # Measurement noise (increased for acceleration components)
        self.ukf.R = np.diag([0.1, 0.1, 0.1,        # accelerometer
                              0.03, 0.03, 0.03,      # gyroscope
                              20.0, 20.0, 20.0,      # magnetometer
                              0.5, 0.5, 0.5])        # world acceleration
    
    def update_visualization(self):
        """Update visualization based on sensor data"""
        if not self.ser or not self.ser.is_open:
            return
        
        try:
            # Read data from serial port
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                data = parse_raw_data(line)
                
                if data:
                    accel, gyro, mag = data
                    
                    # Transform acceleration to world frame (removing gravity)
                    # Get current quaternion
                    quat = self.ukf.x[0:4]
                    R = quaternion_to_matrix(quat)
                    accel_world = R @ accel - np.array([0, 0, G])
                    
                    # Create augmented measurement vector including world-frame acceleration
                    z = np.concatenate([accel, gyro, mag, accel_world])
                    
                    # Calculate dt
                    current_time = time.time()
                    dt = current_time - self.prev_time
                    print(dt)
                    self.prev_time = current_time
                    
                    # Update UKF time step
                    self.ukf.dt = dt
                    
                    # Update current acceleration in state vector before prediction
                    self.ukf.x[13:16] = accel_world
                    
                    # Run UKF
                    self.ukf.predict()
                    self.ukf.update(z)
                    
                    # Check initialization period
                    elapsed_time = current_time - self.start_time
                    if elapsed_time < self.init_period:
                        self.ukf.x[7:10] = np.zeros(3)  # Force velocity to zero
                        self.ukf.x[10:13] = np.zeros(3)  # Force position to zero
                        self.status_label.setText(f"Initializing... ({elapsed_time:.1f}s)")
                    else:
                        self.initialized = True
                    
                    # Get current state estimates
                    quat = self.ukf.x[0:4]
                    position = self.ukf.x[10:13]
                    velocity = self.ukf.x[7:10]
                    acceleration = self.ukf.x[13:16]
                    
                    # Update PCB position and orientation
                    self.update_pcb(quat, position)
                    
                    # Update trail
                    self.update_trail(position)
                    
                    # Display current state
                    roll, pitch, yaw = quaternion_to_euler(quat)
                    speed = np.linalg.norm(velocity)
                    
                    status_text = f'Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}° | ' \
                                 f'Pos: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] | ' \
                                 f'Vel: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}] | ' \
                                 f'Accel: [{acceleration[0]:.2f}, {acceleration[1]:.2f}, {acceleration[2]:.2f}] | ' \
                                 f'Speed: {speed:.2f} m/s'
                    self.status_label.setText(status_text)
                    
                    # Center view on PCB
                    self.view.opts['center'] = pg.Vector(float(position[0]), float(position[1]), float(position[2]))
        
        except Exception as e:
            print(f"Error in update_visualization: {e}")
            self.status_label.setText(f"Error: {e}")
            # Print stack trace for easier debugging
            import traceback
            traceback.print_exc()
    
    def update_pcb(self, quaternion, position):
        """Update PCB position and orientation"""
        # Get rotation matrix from quaternion
        R = quaternion_to_matrix(quaternion)
        
        # A simpler approach: directly manipulate the mesh vertices
        # PCB dimensions in meters
        l, w, h = 0.015, 0.02, 0.002  # Length, width, height
        
        # Define original box vertices
        vertices = np.array([
            [-l/2, -w/2, -h/2],  # 0: bottom-left-back
            [l/2, -w/2, -h/2],   # 1: bottom-right-back
            [l/2, w/2, -h/2],    # 2: bottom-right-front
            [-l/2, w/2, -h/2],   # 3: bottom-left-front
            [-l/2, -w/2, h/2],   # 4: top-left-back
            [l/2, -w/2, h/2],    # 5: top-right-back
            [l/2, w/2, h/2],     # 6: top-right-front
            [-l/2, w/2, h/2],    # 7: top-left-front
        ])
        
        # Rotate vertices
        rotated_vertices = np.array([R @ v for v in vertices])
        
        # Translate vertices
        transformed_vertices = rotated_vertices + position
        
        # Update mesh with new vertices
        self.pcb.setMeshData(vertexes=transformed_vertices, faces=self.pcb_faces)
        
        # Update PCB axes
        arrow_length = 0.1
        
        # X axis (red)
        x_pos = np.array([[float(position[0]), float(position[1]), float(position[2])],
                          [float(position[0] + R[0, 0]*arrow_length), 
                           float(position[1] + R[1, 0]*arrow_length), 
                           float(position[2] + R[2, 0]*arrow_length)]])
        self.pcb_axes[0].setData(pos=x_pos)
        
        # Y axis (green)
        y_pos = np.array([[float(position[0]), float(position[1]), float(position[2])],
                          [float(position[0] + R[0, 1]*arrow_length), 
                           float(position[1] + R[1, 1]*arrow_length), 
                           float(position[2] + R[2, 1]*arrow_length)]])
        self.pcb_axes[1].setData(pos=y_pos)
        
        # Z axis (blue)
        z_pos = np.array([[float(position[0]), float(position[1]), float(position[2])],
                          [float(position[0] + R[0, 2]*arrow_length), 
                           float(position[1] + R[1, 2]*arrow_length), 
                           float(position[2] + R[2, 2]*arrow_length)]])
        self.pcb_axes[2].setData(pos=z_pos)
        
        # Update velocity vector (yellow)
        velocity = self.ukf.x[7:10]
        velocity_scale = 0.3  # Scale factor to make vector visible
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude > 0.01:  # Only show if velocity is significant
            vel_end = position + velocity * velocity_scale
            vel_pos = np.array([
                [float(position[0]), float(position[1]), float(position[2])],
                [float(vel_end[0]), float(vel_end[1]), float(vel_end[2])]
            ])
        #     self.velocity_arrow.setData(pos=vel_pos)
        # else:
        #     # Hide arrow if velocity is too small
        #     self.velocity_arrow.setData(pos=np.array([[0,0,0], [0,0,0]]))
        
        # Update acceleration vector (purple)
        acceleration = self.ukf.x[13:16]
        accel_scale = 0.5  # Scale factor to make vector visible
        accel_magnitude = np.linalg.norm(acceleration)
        if accel_magnitude > 0.01:  # Only show if acceleration is significant
            accel_end = position + acceleration * accel_scale
            accel_pos = np.array([
                [float(position[0]), float(position[1]), float(position[2])],
                [float(accel_end[0]), float(accel_end[1]), float(accel_end[2])]
            ])
        #     self.accel_arrow.setData(pos=accel_pos)
        # else:
        #     # Hide arrow if acceleration is too small
        #     self.accel_arrow.setData(pos=np.array([[0,0,0], [0,0,0]]))
    
    def update_trail(self, position):
        """Update trail showing movement path"""
        # Ensure position is converted to a regular list before appending
        self.trail_points.append(position.tolist())
        if len(self.trail_points) > 1:
            trail_array = np.array(self.trail_points)
            self.trail.setData(pos=trail_array)
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed")
        event.accept()

def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = PCBVisualizer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
