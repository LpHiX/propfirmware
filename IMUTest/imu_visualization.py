import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
from collections import deque
import math

# Constants
CALIBRATION_TIME = 5  # seconds
CALIBRATION_TIMEOUT = 15  # seconds - fallback timeout if calibration doesn't complete
ANIMATION_INTERVAL = 20  # milliseconds (50Hz = 1000/50 = 20ms)
SERIAL_PORT = 'COM7'
BAUD_RATE = 115200  # Adjust based on your board's configuration

# Create a serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Allow time for connection to establish

# Buffer for collecting multiline serial data
serial_buffer = ""

# Rotation state
class IMUState:
    def __init__(self):
        # Orientation quaternion (w, x, y, z)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Calibration offsets
        self.accel_offset = np.array([0.0, 0.0, 0.0])
        self.gyro_offset = np.array([0.0, 0.0, 0.0])
        self.mag_offset = np.array([0.0, 0.0, 0.0])
        self.mag_scale = np.array([1.0, 1.0, 1.0])
        
        # Initial values
        self.accel = np.array([0.0, 0.0, 0.0])
        self.gyro = np.array([0.0, 0.0, 0.0])
        self.mag = np.array([0.0, 0.0, 0.0])
        
        # Last update time
        self.last_update = time.time()
        
        # Calibration data storage
        self.calibration_data = {
            'accel': [],
            'gyro': [],
            'mag': []
        }
        
        # Calibration status
        self.calibrating = True
        self.calibration_start_time = time.time()

# Initialize state
imu_state = IMUState()

# Data buffers for plotting
buffer_size = 100
time_buffer = deque(maxlen=buffer_size)
roll_buffer = deque(maxlen=buffer_size)
pitch_buffer = deque(maxlen=buffer_size)
yaw_buffer = deque(maxlen=buffer_size)

# Parse sensor data from serial input with enhanced parsing for multiline data
def parse_serial_data(line):
    global serial_buffer
    
    # Append the new line to our buffer
    serial_buffer += line + "\n"
    
    # Only process if we have a complete data block (marked by separator line)
    if "------------------------" in serial_buffer:
        print(f"Processing complete data block: {serial_buffer}")
        data_block = serial_buffer
        serial_buffer = ""  # Clear the buffer
    else:
        # Wait for more data
        return {}
    
    data = {}
    
    # Parse accelerometer data
    accel_match = re.search(r'Acceleration \(m/s\^2\): X=([-\d.]+), Y=([-\d.]+), Z=([-\d.]+)', data_block)
    if accel_match:
        data['accel'] = np.array([float(accel_match.group(1)), float(accel_match.group(2)), float(accel_match.group(3))])
        print(f"Parsed accel: {data['accel']}")
    
    # Parse gyroscope data
    gyro_match = re.search(r'Rotation \(rad/s\): X=([-\d.]+), Y=([-\d.]+), Z=([-\d.]+)', data_block)
    if gyro_match:
        data['gyro'] = np.array([float(gyro_match.group(1)), float(gyro_match.group(2)), float(gyro_match.group(3))])
        print(f"Parsed gyro: {data['gyro']}")
    
    # Parse magnetometer data
    mag_match = re.search(r'Magnetic: X=([-\d.]+), Y=([-\d.]+), Z=([-\d.]+)', data_block)
    if mag_match:
        data['mag'] = np.array([float(mag_match.group(1)), float(mag_match.group(2)), float(mag_match.group(3))])
        print(f"Parsed mag: {data['mag']}")
    
    # Check if we have all sensor data
    if 'accel' in data and 'gyro' in data and 'mag' in data:
        print("Successfully parsed complete sensor data")
    else:
        print(f"Incomplete sensor data: {data.keys()}")
    
    return data

# Quaternion operations
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_to_euler(q):
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

# Sensor fusion using Madgwick filter
def madgwick_update(state, accel, gyro, mag, dt, beta=0.1):
    # Ensure data is normalized
    if np.linalg.norm(accel) > 0:
        accel = accel / np.linalg.norm(accel)
    
    if np.linalg.norm(mag) > 0:
        mag = mag / np.linalg.norm(mag)
    
    # Current orientation
    q = state.q
    
    # Extract quaternion elements
    q0, q1, q2, q3 = q
    
    # Gyroscope in rad/s
    gx, gy, gz = gyro
    
    # Reference direction of Earth's magnetic field
    h = quaternion_multiply(
        quaternion_multiply(q, np.array([0, 0, 0, 1])),
        quaternion_conjugate(q)
    )
    bx = np.sqrt(h[1]**2 + h[2]**2)
    bz = h[3]
    
    # Estimated direction of gravity and magnetic field
    vx = 2 * (q1 * q3 - q0 * q2)
    vy = 2 * (q0 * q1 + q2 * q3)
    vz = q0**2 - q1**2 - q2**2 + q3**2
    
    wx = 2 * bx * (0.5 - q2**2 - q3**2) + 2 * bz * (q1 * q3 - q0 * q2)
    wy = 2 * bx * (q1 * q2 - q0 * q3) + 2 * bz * (q0 * q1 + q2 * q3)
    wz = 2 * bx * (q0 * q2 + q1 * q3) + 2 * bz * (0.5 - q1**2 - q2**2)
    
    # Error is cross product between estimated direction and measured direction of gravity
    ex = (vy * accel[2] - vz * accel[1]) + (wy * mag[2] - wz * mag[1])
    ey = (vz * accel[0] - vx * accel[2]) + (wz * mag[0] - wx * mag[2])
    ez = (vx * accel[1] - vy * accel[0]) + (wx * mag[1] - wy * mag[0])
    
    # Apply proportional feedback
    gx += beta * ex
    gy += beta * ey
    gz += beta * ez
    
    # Integrate rate of change of quaternion
    qa_dot = -0.5 * (q1 * gx + q2 * gy + q3 * gz)
    qb_dot = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
    qc_dot = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
    qd_dot = 0.5 * (q0 * gz + q1 * gy - q2 * gx)
    
    # Integrate to get quaternion
    q0 += qa_dot * dt
    q1 += qb_dot * dt
    q2 += qc_dot * dt
    q3 += qd_dot * dt
    
    # Normalize quaternion
    norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q = np.array([q0, q1, q2, q3]) / norm
    
    return q

# Calibration function
def calibrate(state, accel, gyro, mag):
    # Add data to calibration storage
    state.calibration_data['accel'].append(accel)
    state.calibration_data['gyro'].append(gyro)
    state.calibration_data['mag'].append(mag)
    
    # Check if calibration period is over
    current_time = time.time()
    if current_time - state.calibration_start_time >= CALIBRATION_TIME:
        # Calculate offsets
        accel_data = np.array(state.calibration_data['accel'])
        gyro_data = np.array(state.calibration_data['gyro'])
        mag_data = np.array(state.calibration_data['mag'])
        
        # Accelerometer: at rest, should show [0, 0, 1g]
        # Zeroing out the inclination, not the gravity component
        accel_avg = np.mean(accel_data, axis=0)
        gravity_vector = accel_avg / np.linalg.norm(accel_avg)
        
        # Calculate initial orientation from gravity vector
        pitch = -np.arcsin(gravity_vector[0])
        roll = np.arctan2(gravity_vector[1], gravity_vector[2])
        
        # Create quaternion from roll and pitch
        cy = np.cos(0 / 2)  # Initial yaw = 0
        sy = np.sin(0 / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        
        state.q = np.array([w, x, y, z])
        
        # Gyroscope: at rest, should show [0, 0, 0]
        state.gyro_offset = np.mean(gyro_data, axis=0)
        
        # Magnetometer calibration (hard-iron offsets)
        state.mag_offset = (np.max(mag_data, axis=0) + np.min(mag_data, axis=0)) / 2
        
        # Magnetometer calibration (soft-iron scaling)
        mag_centered = mag_data - state.mag_offset
        avg_delta = (np.max(mag_centered, axis=0) - np.min(mag_centered, axis=0)) / 2
        avg_radius = np.mean(avg_delta)
        state.mag_scale = avg_radius / avg_delta
        
        print("Calibration complete!")
        print(f"Accel gravity vector: {gravity_vector}")
        print(f"Gyro offset: {state.gyro_offset}")
        print(f"Mag offset: {state.mag_offset}")
        print(f"Mag scale: {state.mag_scale}")
        
        state.calibrating = False

# Update function to process new sensor data
def update_orientation(state, data):
    current_time = time.time()
    dt = current_time - state.last_update
    state.last_update = current_time
    
    # Force exit from calibration if taking too long
    if state.calibrating and (current_time - state.calibration_start_time > CALIBRATION_TIMEOUT):
        print(f"Calibration timeout after {CALIBRATION_TIMEOUT} seconds. Forcing exit from calibration.")
        
        # Set default values if we don't have enough data
        if not state.calibration_data['accel'] or not state.calibration_data['gyro'] or not state.calibration_data['mag']:
            print("Warning: Not enough calibration data collected. Using default values.")
            state.gyro_offset = np.array([0.0, 0.0, 0.0])
            state.mag_offset = np.array([0.0, 0.0, 0.0])
            state.mag_scale = np.array([1.0, 1.0, 1.0])
        else:
            # Use what data we have
            if state.calibration_data['accel']:
                accel_data = np.array(state.calibration_data['accel'])
                accel_avg = np.mean(accel_data, axis=0)
                print(f"Using {len(accel_data)} accel samples for calibration")
            
            if state.calibration_data['gyro']:
                gyro_data = np.array(state.calibration_data['gyro'])
                state.gyro_offset = np.mean(gyro_data, axis=0)
                print(f"Using {len(gyro_data)} gyro samples for calibration")
            
            if state.calibration_data['mag']:
                mag_data = np.array(state.calibration_data['mag'])
                state.mag_offset = (np.max(mag_data, axis=0) + np.min(mag_data, axis=0)) / 2
                print(f"Using {len(mag_data)} mag samples for calibration")
        
        state.calibrating = False
        return
    
    # Check if we have complete sensor data
    has_all_data = 'accel' in data and 'gyro' in data and 'mag' in data
    
    # During calibration period, collect whatever data we have
    if state.calibrating:
        if 'accel' in data:
            state.calibration_data['accel'].append(data['accel'])
        if 'gyro' in data:
            state.calibration_data['gyro'].append(data['gyro'])
        if 'mag' in data:
            state.calibration_data['mag'].append(data['mag'])
        
        # Only proceed with calibration check if we have some data
        if has_all_data or (len(state.calibration_data['accel']) > 0 and 
                            len(state.calibration_data['gyro']) > 0 and 
                            len(state.calibration_data['mag']) > 0):
            calibrate(state, data.get('accel', np.array([0.0, 0.0, 0.0])), 
                            data.get('gyro', np.array([0.0, 0.0, 0.0])), 
                            data.get('mag', np.array([0.0, 0.0, 0.0])))
        return
    
    # After calibration, update orientation only if we have all the data
    if has_all_data:
        accel = data['accel']
        gyro = data['gyro'] - state.gyro_offset  # Apply gyro offset
        mag = (data['mag'] - state.mag_offset) * state.mag_scale  # Apply mag calibration
        
        # Update orientation using sensor fusion
        state.q = madgwick_update(state, accel, gyro, mag, dt)
        
        # Convert quaternion to Euler angles
        euler = quaternion_to_euler(state.q)
        
        # Update data buffers for plotting
        time_buffer.append(current_time)
        roll_buffer.append(math.degrees(euler[0]))
        pitch_buffer.append(math.degrees(euler[1]))
        yaw_buffer.append(math.degrees(euler[2]))

# Set up the figure for animation - IMPORTANT: disable blitting for 3D animation
fig = plt.figure(figsize=(15, 10))
plt.suptitle('IMU Orientation Tracking', fontsize=16)

# 3D axis for showing orientation
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_title('Board Orientation')
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])
ax1.set_zlim([-1, 1])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Create a cuboid to represent the board
vertices = np.array([
    [-0.5, -0.3, -0.1],  # bottom-left-back
    [0.5, -0.3, -0.1],   # bottom-right-back
    [0.5, 0.3, -0.1],    # bottom-right-front
    [-0.5, 0.3, -0.1],   # bottom-left-front
    [-0.5, -0.3, 0.1],   # top-left-back
    [0.5, -0.3, 0.1],    # top-right-back
    [0.5, 0.3, 0.1],     # top-right-front
    [-0.5, 0.3, 0.1]     # top-left-front
])

# Define the faces of the cuboid
faces = [
    [0, 1, 2, 3],  # Bottom face
    [4, 5, 6, 7],  # Top face
    [0, 1, 5, 4],  # Back face
    [3, 2, 6, 7],  # Front face
    [0, 3, 7, 4],  # Left face
    [1, 2, 6, 5]   # Right face
]

# Define colors for each face
face_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

# Create a list to store the plotted faces
face_polygons = []
for i, face in enumerate(faces):
    verts = [vertices[j] for j in face]
    polygon = ax1.add_collection3d(Poly3DCollection([verts], alpha=0.8, color=face_colors[i]))
    face_polygons.append(polygon)

# Arrow to show direction
arrow, = ax1.plot([0, 0.8], [0, 0], [0, 0], 'k-', lw=2, markersize=8)
arrow_head = ax1.scatter([0.8], [0], [0], color='black', s=100)

# Plots for Euler angles
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Roll')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Degrees')
roll_line, = ax2.plot([], [], 'r-')

ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('Pitch')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Degrees')
pitch_line, = ax3.plot([], [], 'g-')

ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('Yaw')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Degrees')
yaw_line, = ax4.plot([], [], 'b-')

# Add text to display the angles
text_box = plt.figtext(0.02, 0.02, '', fontsize=10, wrap=True, 
                      bbox=dict(facecolor='white', alpha=0.8))

# Fixed list of all artists that will be returned
# This ensures we don't accidentally return None values
all_artists = []

# Animation update function
def update_plot(frame):
    global all_artists
    
    try:
        # Collect data from serial port
        data = {}
        
        # Make sure we have serial data to read
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line:  # Only process non-empty lines
                    data = parse_serial_data(line)
                    if data:  # Only update if we parsed complete data
                        update_orientation(imu_state, data)
            except Exception as e:
                print(f"Error reading serial data: {e}")
        
        # If this is the first frame, initialize all_artists
        if not all_artists:
            all_artists = face_polygons + [arrow, arrow_head, roll_line, pitch_line, yaw_line, text_box]
        
        # Skip visualization during calibration
        if imu_state.calibrating:
            elapsed = time.time() - imu_state.calibration_start_time
            text_box.set_text(f'Calibrating... {elapsed:.1f}s / {CALIBRATION_TIME}s\n'
                             f'Samples: Accel={len(imu_state.calibration_data["accel"])}, '
                             f'Gyro={len(imu_state.calibration_data["gyro"])}, '
                             f'Mag={len(imu_state.calibration_data["mag"])}')
            return all_artists
        
        # Calculate the rotation matrix from quaternion
        q = imu_state.q
        rot_matrix = np.array([
            [1 - 2*q[2]**2 - 2*q[3]**2, 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
            [2*q[1]*q[2] + 2*q[0]*q[3], 1 - 2*q[1]**2 - 2*q[3]**2, 2*q[2]*q[3] - 2*q[0]*q[1]],
            [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], 1 - 2*q[1]**2 - 2*q[2]**2]
        ])
        
        # Update cuboid orientation
        try:
            rotated_vertices = np.array([np.dot(rot_matrix, vertex) for vertex in vertices])
            
            # Update the 3D polygons
            for i, face in enumerate(faces):
                verts3d = [rotated_vertices[j] for j in face]
                face_polygons[i].set_verts([verts3d])
        except Exception as e:
            print(f"Error updating 3D model: {e}")
        
        # Update direction arrow
        try:
            forward_vector = np.dot(rot_matrix, [0.8, 0, 0])
            arrow.set_data([0, forward_vector[0]], [0, forward_vector[1]])
            arrow.set_3d_properties([0, forward_vector[2]])
            arrow_head._offsets3d = ([forward_vector[0]], [forward_vector[1]], [forward_vector[2]])
        except Exception as e:
            print(f"Error updating arrow: {e}")
        
        # Update Euler angle plots
        try:
            if len(time_buffer) > 1:
                times = np.array(time_buffer) - time_buffer[0]
                
                roll_line.set_data(times, roll_buffer)
                pitch_line.set_data(times, pitch_buffer)
                yaw_line.set_data(times, yaw_buffer)
                
                # Adjust y-limits if needed
                for ax, data in zip([ax2, ax3, ax4], [roll_buffer, pitch_buffer, yaw_buffer]):
                    if len(data) > 0:
                        min_val = min(data)
                        max_val = max(data)
                        margin = (max_val - min_val) * 0.1 + 1  # Add a small margin
                        ax.set_ylim(min_val - margin, max_val + margin)
                    
                    # Adjust x-limits
                    ax.set_xlim(0, max(times))
            
            # Update text display with current angles
            if len(roll_buffer) > 0:
                text_box.set_text(f'Roll: {roll_buffer[-1]:.2f}°\nPitch: {pitch_buffer[-1]:.2f}°\nYaw: {yaw_buffer[-1]:.2f}°')
        except Exception as e:
            print(f"Error updating plots: {e}")
        
        return all_artists
    except Exception as e:
        print(f"Error in animation update: {e}")
        return all_artists  # Always return the same list of artists

# Create animation - DISABLE BLITTING FOR 3D PLOTS
ani = FuncAnimation(fig, update_plot, interval=ANIMATION_INTERVAL, blit=False)

plt.tight_layout()
plt.show()

# Clean up
ser.close()
