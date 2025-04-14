import re
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

class GY87Parser:
    def __init__(self, max_history=1000):
        # Initialize sensor data variables
        self.accel = np.zeros(3)  # X, Y, Z acceleration in m/s^2
        self.gravity_accel = np.zeros([3, 100])
        self.accel_without_gravity = np.zeros(3)
        self.calibrate_number = 0
        self.gyro = np.zeros(3)   # X, Y, Z rotation in rad/s
        self.mag = np.zeros(3)    # X, Y, Z magnetic field
        self.temp_mpu = 0.0       # Temperature from MPU6050
        self.heading = 0.0        # Heading in degrees
        self.direction = ""       # Direction (N, NE, E, etc.)
        self.pressure = 0.0       # Pressure in hPa
        self.temp_bmp = 0.0       # Temperature from BMP180
        self.altitude = 0.0       # Altitude in meters
        
        # Variables for position tracking
        self.velocity = np.zeros(3)  # X, Y, Z velocity in m/s
        self.position = np.zeros(3)  # X, Y, Z position in m

        
        # Variables for plotting
        self.max_history = max_history
        self.position_history = deque(maxlen=max_history)
        self.position_history.append(self.position.copy())
        
        # Setup 3D visualization
        self.fig = plt.figure(figsize=(10, 10))  # Increased figure height
        
        # Create layout with space for text display
        self.ax = self.fig.add_subplot(211, projection='3d')  # Use 2-row layout, plot in top row
        self.position_line, = self.ax.plot([], [], [], 'b-', label='Position Track')
        self.current_point = self.ax.scatter(0, 0, 0, color='red', s=100, label='Current Position')
        
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xlabel('X Position - Current centered (m)')
        self.ax.set_ylabel('Y Position - Current centered (m)')
        self.ax.set_zlabel('Z Position - Current centered (m)')
        self.ax.set_title('3D Position Tracking - Centered View')
        self.ax.legend()
        
        # Create text area for data display
        self.text_ax = self.fig.add_subplot(212)
        self.text_ax.axis('off')  # Hide axes
        self.data_text = self.text_ax.text(0.05, 0.6, "", transform=self.text_ax.transAxes, 
                                          fontsize=12, family='monospace',
                                          verticalalignment='top', wrap=True)
        
        # Animation setup - 10ms interval for 100fps
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=10, blit=False,save_count=100)  # Changed blit to False to allow text updates
    
    def parse_data(self, data_str):
        """Parse sensor data from string format"""
        # MPU6050 data
        accel_match = re.search(r'Accel: X=([-\d.]+), Y=([-\d.]+), Z=([-\d.]+)', data_str)
        if accel_match:
            self.accel = np.array([float(accel_match.group(1)), float(accel_match.group(2)), float(accel_match.group(3))])
        
        # gyro_match = re.search(r'Rotation \(rad/s\): X=([-\d.]+), Y=([-\d.]+), Z=([-\d.]+)', data_str)
        # if gyro_match:
        #     self.gyro = np.array([float(gyro_match.group(1)), float(gyro_match.group(2)), float(gyro_match.group(3))])
        
        # temp_mpu_match = re.search(r'Temperature: ([-\d.]+) °C', data_str)
        # if temp_mpu_match:
        #     self.temp_mpu = float(temp_mpu_match.group(1))
        
        # # QMC5883L data
        # mag_match = re.search(r'Magnetic: X=([-\d.]+), Y=([-\d.]+), Z=([-\d.]+)', data_str)
        # if mag_match:
        #     self.mag = np.array([float(mag_match.group(1)), float(mag_match.group(2)), float(mag_match.group(3))])
        
        # heading_match = re.search(r'Heading: ([-\d.]+) degrees', data_str)
        # if heading_match:
        #     self.heading = float(heading_match.group(1))
        
        # # BMP180 data
        # pressure_match = re.search(r'Pressure: ([-\d.]+) hPa', data_str)
        # if pressure_match:
        #     self.pressure = float(pressure_match.group(1))
        
        # temp_bmp_match = re.search(r'Temperature: ([-\d.]+) °C', data_str.split('BMP180 Data:')[1] if 'BMP180 Data:' in data_str else "")
        # if temp_bmp_match:
        #     self.temp_bmp = float(temp_bmp_match.group(1))
        
        # altitude_match = re.search(r'Altitude: ([-\d.]+) m', data_str)
        # if altitude_match:
        #     self.altitude = float(altitude_match.group(1))
        
        # Update position tracking
        if self.calibrate_number < 100:
            self.calibrate()
        else:
            self.update_position()

    def calibrate(self):
        """Calibrate the sensor data"""
        # Collect calibration data for gravity compensation
        self.gravity_accel[:, self.calibrate_number] = self.accel
        print(f"Calibration data collected: {self.gravity_accel[:, self.calibrate_number]}")
        self.calibrate_number += 1
        
        if self.calibrate_number == 100:
            # Calculate average gravity vector
            avg_gravity = np.mean(self.gravity_accel[:, 1:99], axis=1)
            print(f"Calibration complete. Average Gravity Vector: {avg_gravity}")
            # Store the average gravity vector for future use
            self.gravity_accel = avg_gravity
            self.last_time = time.time()
    def update_position(self):
        """Update velocity and position using acceleration data"""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Remove gravity component from Z acceleration
        self.accel_without_gravity = self.accel.copy()
        # accel_without_gravity[2] -= 9.81  # Subtract gravity (9.81 m/s^2)
        self.accel_without_gravity -= self.gravity_accel
        
        # Apply simple noise filtering
        threshold = 0.05  # m/s^2
        self.accel_without_gravity = np.where(
            np.abs(self.accel_without_gravity) < threshold, 
            0, 
            self.accel_without_gravity
        )
        
        # # First integration: acceleration -> velocity
        self.velocity += self.accel_without_gravity * dt
        
        # Apply velocity decay to counter drift
        decay_factor = 0.995  # Adjust as needed for drift control
        self.velocity *= decay_factor
        
        # Second integration: velocity -> position
        self.position += self.velocity * dt
        
        # Store position history for plotting
        self.position_history.append(self.position.copy())
    
    def update_plot(self, frame):
        """Update the 3D plot with current position"""
        if self.position_history:
            positions = np.array(self.position_history)
            x = positions[:, 0]
            y = positions[:, 1]
            z = positions[:, 2]
            
            # Update position track
            self.position_line.set_data(x, y)
            self.position_line.set_3d_properties(z)
            
            # Update current position point
            self.current_point._offsets3d = ([self.position[0]], [self.position[1]], [self.position[2]])
            
            # Keep the object at the center by adjusting axis limits
            view_range = 0.2  # Size of visible area around current position
            
            # Set axis limits centered around current position
            self.ax.set_xlim(self.position[0] - view_range/2, self.position[0] + view_range/2)
            self.ax.set_ylim(self.position[1] - view_range/2, self.position[1] + view_range/2)
            self.ax.set_zlim(self.position[2] - view_range/2, self.position[2] + view_range/2)
            
            # Update the title with current position information
            self.ax.set_title(f'3D Position Tracking - Current Position: [{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}] m')
            
            # Update the text display with current values
            text_content = (
                f"ACCELERATION (m/s²):\n"
                f"    X: {self.accel[0]:8.3f}    Y: {self.accel[1]:8.3f}    Z: {self.accel[2]:8.3f}\n\n"
                f"ACCELERATION (m/s²) - GRAVITY COMPENSATED:\n"
                f"    X: {self.accel_without_gravity[0]:8.3f}    Y: {self.accel_without_gravity[1]:8.3f}    Z: {self.accel_without_gravity[2]:8.3f}\n\n"
                f"VELOCITY (m/s):\n"
                f"    X: {self.velocity[0]:8.3f}    Y: {self.velocity[1]:8.3f}    Z: {self.velocity[2]:8.3f}\n\n"
                f"POSITION (m):\n"
                f"    X: {self.position[0]:8.3f}    Y: {self.position[1]:8.3f}    Z: {self.position[2]:8.3f}"
            )
            self.data_text.set_text(text_content)
            
        return self.position_line, self.current_point
    
    def start_visualization(self):
        """Start the real-time visualization"""
        plt.show()


# Example usage
if __name__ == "__main__":
    import serial
    import io
    import threading
    import time

    
    # Initialize parser
    parser = GY87Parser()
    
    # Set up serial communication
    try:
        # Adjust port and baudrate as needed
        ser = serial.Serial('COM7', 921600, timeout=1)
        buffer = ""
        
        print("Tracking position in real-time. Close the plot window to exit.")
        
        # Function to read and parse data in a loop
        def read_and_update():
            while True:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line:
                        parser.parse_data(line)
                except Exception as e:
                    print(f"Error reading data: {e}")
        
        # Start data reading in a separate thread
        thread = threading.Thread(target=read_and_update, daemon=True)
        thread.start()
        
        # Start visualization (blocks until the plot window is closed)
        parser.start_visualization()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed.")
