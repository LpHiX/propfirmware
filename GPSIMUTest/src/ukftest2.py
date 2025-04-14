import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import serial
from matplotlib.animation import FuncAnimation
import time

# Constants
COM_PORT = 'COM6'
BAUD_RATE = 115200
GYRO_BIAS = np.array([0.04, 0.0, -0.04])  # Given gyro bias
G = 9.81  # Gravity constant
LONDON_MAG_INCLINATION = 66  # Magnetic inclination in London (degrees)

# State vector representation:
# x[0:4] = quaternion (w,x,y,z)
# x[4:7] = gyro bias

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
    
    # Return new state
    return np.concatenate([new_quat, new_bias])

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
        accel = np.array([float(values[0]), float(values[1]), float(values[2])])
        gyro = np.array([float(values[3]), float(values[4]), float(values[5])])
        mag = np.array([float(values[6]), float(values[7]), float(values[8])])
        
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
    
    # Set axis limits
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_zlim(-0.2, 0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('PCB Orientation Tracking with UKF')
    
    plt.legend()
    
    return fig, ax, box_edges, pcb_x, pcb_y, pcb_z, vertices

def update_visualization(box_edges, pcb_x, pcb_y, pcb_z, vertices, quaternion):
    """Update PCB visualization based on current quaternion"""
    # Get rotation matrix from quaternion
    R = quaternion_to_matrix(quaternion)
    
    # Rotate vertices
    rotated_vertices = (R @ vertices.T).T
    
    # Update box edges
    # Bottom face
    box_edges[0].set_data(rotated_vertices[[0, 1, 2, 3, 0], 0], 
                         rotated_vertices[[0, 1, 2, 3, 0], 1])
    box_edges[0].set_3d_properties(rotated_vertices[[0, 1, 2, 3, 0], 2])
    
    # Top face
    box_edges[1].set_data(rotated_vertices[[4, 5, 6, 7, 4], 0], 
                         rotated_vertices[[4, 5, 6, 7, 4], 1])
    box_edges[1].set_3d_properties(rotated_vertices[[4, 5, 6, 7, 4], 2])
    
    # Connecting edges
    box_edges[2].set_data(rotated_vertices[[0, 4], 0], rotated_vertices[[0, 4], 1])
    box_edges[2].set_3d_properties(rotated_vertices[[0, 4], 2])
    
    box_edges[3].set_data(rotated_vertices[[1, 5], 0], rotated_vertices[[1, 5], 1])
    box_edges[3].set_3d_properties(rotated_vertices[[1, 5], 2])
    
    box_edges[4].set_data(rotated_vertices[[2, 6], 0], rotated_vertices[[2, 6], 1])
    box_edges[4].set_3d_properties(rotated_vertices[[2, 6], 2])
    
    box_edges[5].set_data(rotated_vertices[[3, 7], 0], rotated_vertices[[3, 7], 1])
    box_edges[5].set_3d_properties(rotated_vertices[[3, 7], 2])
    
    # Update PCB coordinate system
    arrow_length = 0.1
    # X axis (red)
    pcb_x.set_segments([np.array([[0, 0, 0], [R[0, 0], R[1, 0], R[2, 0]]])*arrow_length])
    # Y axis (green)
    pcb_y.set_segments([np.array([[0, 0, 0], [R[0, 1], R[1, 1], R[2, 1]]])*arrow_length])
    # Z axis (blue)
    pcb_z.set_segments([np.array([[0, 0, 0], [R[0, 2], R[1, 2], R[2, 2]]])*arrow_length])

def main():
    # Initialize UKF
    n_state = 7  # 4 quaternion + 3 gyro bias
    n_measurement = 9  # 3 accel + 3 gyro + 3 mag
    
    # Create sigma points
    points = MerweScaledSigmaPoints(n=n_state, alpha=0.1, beta=2., kappa=-1)
    
    # Initialize filter
    ukf = UnscentedKalmanFilter(dim_x=n_state, dim_z=n_measurement, dt=0.04, 
                               fx=state_transition_fn, hx=measurement_fn, points=points)
    
    # Initial state (identity quaternion, zero gyro bias)
    ukf.x = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # State covariance
    ukf.P = np.diag([0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
    
    # Process noise
    ukf.Q = np.diag([0.0001, 0.01, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
    
    # Measurement noise
    ukf.R = np.diag([0.1, 0.1, 0.1, 0.03, 0.03, 0.03, 50.0, 50.0, 50.0])
    
    # Setup visualization
    fig, ax, box_edges, pcb_x, pcb_y, pcb_z, vertices = setup_visualization()
    
    # Initialize serial connection
    ser = serial.Serial(COM_PORT, BAUD_RATE)
    print(f"Connected to {COM_PORT} at {BAUD_RATE} baud")
    
    # Animation function
    def update(frame):
        try:
            # Read data from serial port
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                data = parse_raw_data(line)
                
                if data:
                    accel, gyro, mag = data
                    z = np.concatenate([accel, gyro, mag])
                    
                    # Run UKF
                    ukf.predict()
                    ukf.update(z)
                    
                    # Get current quaternion estimate
                    quat = ukf.x[0:4]
                    
                    # Update visualization
                    update_visualization(box_edges, pcb_x, pcb_y, pcb_z, vertices, quat)
                    
                    # Display current orientation
                    roll, pitch, yaw = quaternion_to_euler(quat)
                    ax.set_title(f'Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°')
                    
        except Exception as e:
            print(f"Error reading from serial port: {e}")
            
        return box_edges + [pcb_x, pcb_y, pcb_z]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=None, 
                       interval=10, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    # Close serial port when done
    ser.close()

if __name__ == "__main__":
    main()