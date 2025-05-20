import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import serial
from matplotlib.animation import FuncAnimation
import time
from collections import deque  # For storing trail points

# Constants
COM_PORT = 'COM6'
BAUD_RATE = 115200
GYRO_BIAS = np.array([0.04, 0.0, -0.04])  # Given gyro bias
G = 9.81  # Gravity constant
LONDON_MAG_INCLINATION = 66  # Magnetic inclination in London (degrees)

# Accelerometer calibration parameters
ACCEL_OFFSETS = np.array([0.397600, -0.117200, -0.084255])
ACCEL_SCALE_FACTORS = np.array([1.002334, 0.990214, 1.021738])

# State vector representation:
# x[0:4] = quaternion (w,x,y,z)
# x[4:7] = gyro bias
# x[7:10] = velocity (vx, vy, vz) in world frame
# x[10:13] = position (px, py, pz) in world frame

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
    
    # Update position based on velocity
    new_position = position + velocity * dt
    
    # Velocity remains the same in prediction (will be updated in measurement)
    new_velocity = velocity
    
    # Return new state
    return np.concatenate([new_quat, new_bias, new_velocity, new_position])

def measurement_fn(x):
    """Measurement function for UKF - predicts sensor readings from state"""
    quat = x[0:4]
    gyro_bias = x[4:7]
    
    # Get rotation matrix from quaternion
    R = quaternion_to_matrix(quat)
    
    # Expected accelerometer reading (gravity in sensor frame)
    # In world frame, gravity is [0, 0, G]
    accel_expected = R.T @ np.array([0, 0, G])
    
    # Expected gyroscope reading (known bias - estimated bias)
    gyro_expected = GYRO_BIAS - gyro_bias
    
    # Expected magnetometer reading
    # In London, magnetic field points down and north with inclination ~66°
    inclination_rad = np.radians(LONDON_MAG_INCLINATION)
    # Create unit vector in world frame
    mag_field_world = np.array([
        np.cos(inclination_rad), 
        0, 
        -np.sin(inclination_rad)
    ])
    # Scale to match sensor readings (determined from data)
    mag_scaling = 1500  # Approximate magnitude from sample data
    mag_field_world *= mag_scaling
    
    # Transform to body frame
    mag_expected = R.T @ mag_field_world
    
    # Combine all measurements
    return np.concatenate([accel_expected, gyro_expected, mag_expected])

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
        
        # Apply calibration to accelerometer data
        accel = (accel_raw - ACCEL_OFFSETS) / ACCEL_SCALE_FACTORS
        
        return (accel, gyro, mag)
    except:
        return None

def setup_visualization():
    """Setup 3D visualization for the PCB"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # PCB dimensions in meters
    l, w, h = 0.05, 0.1, 0.01  # Length, width, height
    
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
    
    # Initialize box edges
    box_edges = []
    for i in range(6):
        box_edges.append(ax.plot([], [], [], 'b-')[0])
    
    # Add coordinate system axes
    arrow_length = 0.15
    ax.quiver(0, 0, 0, arrow_length, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, arrow_length, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, arrow_length, color='b', label='Z')
    
    # Add sensors axes attached to PCB
    pcb_x = ax.quiver(0, 0, 0, 0, 0, 0, color='r', label='PCB-X')
    pcb_y = ax.quiver(0, 0, 0, 0, 0, 0, color='g', label='PCB-Y')
    pcb_z = ax.quiver(0, 0, 0, 0, 0, 0, color='b', label='PCB-Z')
    
    # Add trail for movement path
    trail, = ax.plot([], [], [], 'r-', linewidth=1.5, label='Path')
    
    # Set axis limits
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('PCB Orientation & Position Tracking with UKF')
    
    plt.legend()
    
    return fig, ax, box_edges, pcb_x, pcb_y, pcb_z, vertices, trail

def update_visualization(box_edges, pcb_x, pcb_y, pcb_z, vertices, quaternion, position, trail, trail_points):
    """Update PCB visualization based on current quaternion and position"""
    # Get rotation matrix from quaternion
    R = quaternion_to_matrix(quaternion)
    
    # Rotate vertices
    rotated_vertices = (R @ vertices.T).T
    
    # Translate vertices to current position
    translated_vertices = rotated_vertices + position
    
    # Update box edges (centered around current position)
    # Bottom face
    pts = translated_vertices[[0, 1, 2, 3, 0]]
    box_edges[0].set_data(pts[:, 0], pts[:, 1])
    box_edges[0].set_3d_properties(pts[:, 2])
    
    # Top face
    pts = translated_vertices[[4, 5, 6, 7, 4]]
    box_edges[1].set_data(pts[:, 0], pts[:, 1])
    box_edges[1].set_3d_properties(pts[:, 2])
    
    # Connecting edges
    box_edges[2].set_data(translated_vertices[[0, 4], 0], translated_vertices[[0, 4], 1])
    box_edges[2].set_3d_properties(translated_vertices[[0, 4], 2])
    
    box_edges[3].set_data(translated_vertices[[1, 5], 0], translated_vertices[[1, 5], 1])
    box_edges[3].set_3d_properties(translated_vertices[[1, 5], 2])
    
    box_edges[4].set_data(translated_vertices[[2, 6], 0], translated_vertices[[2, 6], 1])
    box_edges[4].set_3d_properties(translated_vertices[[2, 6], 2])
    
    box_edges[5].set_data(translated_vertices[[3, 7], 0], translated_vertices[[3, 7], 1])
    box_edges[5].set_3d_properties(translated_vertices[[3, 7], 2])
    
    # Update PCB coordinate system vectors (at current position)
    arrow_length = 0.1
    # X axis (red)
    pcb_x.set_segments([np.array([[position[0], position[1], position[2]], 
                                 [position[0] + R[0, 0]*arrow_length, 
                                  position[1] + R[1, 0]*arrow_length, 
                                  position[2] + R[2, 0]*arrow_length]])])
    # Y axis (green)
    pcb_y.set_segments([np.array([[position[0], position[1], position[2]], 
                                 [position[0] + R[0, 1]*arrow_length, 
                                  position[1] + R[1, 1]*arrow_length, 
                                  position[2] + R[2, 1]*arrow_length]])])
    # Z axis (blue)
    pcb_z.set_segments([np.array([[position[0], position[1], position[2]], 
                                 [position[0] + R[0, 2]*arrow_length, 
                                  position[1] + R[1, 2]*arrow_length, 
                                  position[2] + R[2, 2]*arrow_length]])])
    
    # Update trail
    trail_points.append(position.copy())
    if len(trail_points) > 1:
        trail_array = np.array(trail_points)
        trail.set_data(trail_array[:, 0], trail_array[:, 1])
        trail.set_3d_properties(trail_array[:, 2])
    
    # Update axis limits to keep PCB centered with margin
    margin = 0.3
    ax = box_edges[0].axes
    ax.set_xlim(position[0] - margin, position[0] + margin)
    ax.set_ylim(position[1] - margin, position[1] + margin)
    ax.set_zlim(position[2] - margin, position[2] + margin)

def main():
    # Initialize UKF
    n_state = 13  # 4 quaternion + 3 gyro bias + 3 velocity + 3 position
    n_measurement = 9  # 3 accel + 3 gyro + 3 mag
    
    # Create sigma points
    points = MerweScaledSigmaPoints(n=n_state, alpha=0.1, beta=2., kappa=-1)
    
    # Initialize filter
    ukf = UnscentedKalmanFilter(dim_x=n_state, dim_z=n_measurement, dt=0.01, 
                               fx=state_transition_fn, hx=measurement_fn, points=points)
    
    # Initial state (identity quaternion, zero gyro bias, zero velocity, zero position)
    ukf.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # State covariance
    ukf.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    # Process noise
    ukf.Q = np.diag([0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 
                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    # Measurement noise
    ukf.R = np.diag([0.1, 0.1, 0.1, 0.03, 0.03, 0.03, 50.0, 50.0, 50.0])
    
    # Setup visualization
    fig, ax, box_edges, pcb_x, pcb_y, pcb_z, vertices, trail = setup_visualization()
    
    # Initialize trail points
    trail_points = deque(maxlen=100)  # Store last 100 positions
    
    # Initialize serial connection
    ser = serial.Serial(COM_PORT, BAUD_RATE)
    print(f"Connected to {COM_PORT} at {BAUD_RATE} baud")
    
    # Time tracking for integration
    prev_time = time.time()
    
    # Animation function
    def update(frame):
        nonlocal prev_time
        
        try:
            # Read data from serial port
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                data = parse_raw_data(line)
                
                if data:
                    accel, gyro, mag = data
                    z = np.concatenate([accel, gyro, mag])
                    
                    # Calculate dt
                    current_time = time.time()
                    dt = current_time - prev_time
                    prev_time = current_time
                    
                    # Update UKF time step
                    ukf.dt = dt
                    
                    # Run UKF
                    ukf.predict()
                    ukf.update(z)
                    
                    # Get current state estimates
                    quat = ukf.x[0:4]
                    position = ukf.x[10:13]
                    
                    # Get rotation matrix from quaternion
                    R = quaternion_to_matrix(quat)
                    
                    # Transform acceleration to world frame (removing gravity)
                    accel_body = accel.copy()
                    accel_world = R @ accel_body - np.array([0, 0, G])
                    
                    # Update velocity based on acceleration
                    ukf.x[7:10] += accel_world * dt
                    
                    # Apply velocity damping to prevent drift
                    damping = 0.98
                    ukf.x[7:10] *= damping
                    
                    # Update visualization
                    update_visualization(
                        box_edges, pcb_x, pcb_y, pcb_z, 
                        vertices, quat, position, trail, trail_points
                    )
                    
                    # Display current orientation and position
                    roll, pitch, yaw = quaternion_to_euler(quat)
                    vel = ukf.x[7:10]
                    speed = np.linalg.norm(vel)
                    title_text = f'Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°\n' \
                               f'Pos: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] ' \
                               f'Speed: {speed:.2f} m/s'
                    ax.set_title(title_text)
                    
                    # Force a redraw to update title and axis limits
                    fig.canvas.draw_idle()
                    
        except Exception as e:
            print(f"Error reading from serial port: {e}")
            
        return box_edges + [pcb_x, pcb_y, pcb_z, trail]
    
    # Create animation with lower blit setting
    ani = FuncAnimation(fig, update, frames=None, 
                       interval=10, blit=False)  # Changed blit to False
    
    plt.tight_layout()
    plt.show()
    
    # Close serial port when done
    ser.close()

if __name__ == "__main__":
    main()