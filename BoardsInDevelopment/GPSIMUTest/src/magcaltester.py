import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse

# Constants
GRAVITY = 9.81  # Standard gravity in m/s²

def parse_args():
    parser = argparse.ArgumentParser(description='MPU6050 Accelerometer Data Reader')
    parser.add_argument('--port', type=str, default='COM6', help='Serial port (default: COM6)')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('--calibration', type=str, help='Path to calibration file')
    return parser.parse_args()

def load_calibration(file_path):
    """Load calibration parameters from file."""
    offsets = np.zeros(3)
    scale_factors = np.ones(3)
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                # Parse offsets
                offset_line = lines[0].strip().replace('Offsets:', '').strip()
                offset_values = offset_line.split(',')
                offsets = np.array([float(val) for val in offset_values])
                
                # Parse scale factors
                scale_line = lines[1].strip().replace('Scale Factors:', '').strip()
                scale_values = scale_line.split(',')
                scale_factors = np.array([float(val) for val in scale_values])
                
        print(f"Loaded calibration parameters:")
        print(f"Offsets: X={offsets[0]:.5f}, Y={offsets[1]:.5f}, Z={offsets[2]:.5f}")
        print(f"Scale Factors: X={scale_factors[0]:.5f}, Y={scale_factors[1]:.5f}, Z={scale_factors[2]:.5f}")
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        print("Using default calibration (no adjustment)")
    
    return offsets, scale_factors

def apply_calibration(raw_data, offsets, scale_factors):
    """Apply calibration to raw accelerometer data."""
    calibrated_data = np.zeros_like(raw_data)
    
    for i in range(3):  # For x, y, z axes
        calibrated_data[i] = (raw_data[i] - offsets[i]) / scale_factors[i]
    
    return calibrated_data

def main():
    args = parse_args()
    
    # Load calibration parameters
    offsets = np.zeros(3)
    scale_factors = np.ones(3)
    
    if args.calibration:
        offsets, scale_factors = load_calibration(args.calibration)
    
    # Setup plots
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Data storage
    data_points = 100
    time_data = np.zeros(data_points)
    raw_data = np.zeros((data_points, 3))
    cal_data = np.zeros((data_points, 3))
    raw_mag = np.zeros(data_points)
    cal_mag = np.zeros(data_points)
    
    # Plot lines
    raw_lines = [
        ax1.plot(time_data, raw_data[:, 0], 'r-', label='X-axis')[0],
        ax1.plot(time_data, raw_data[:, 1], 'g-', label='Y-axis')[0],
        ax1.plot(time_data, raw_data[:, 2], 'b-', label='Z-axis')[0],
        ax1.plot(time_data, raw_mag, 'k-', label='Magnitude')[0]
    ]
    
    cal_lines = [
        ax2.plot(time_data, cal_data[:, 0], 'r-', label='X-axis')[0],
        ax2.plot(time_data, cal_data[:, 1], 'g-', label='Y-axis')[0],
        ax2.plot(time_data, cal_data[:, 2], 'b-', label='Z-axis')[0],
        ax2.plot(time_data, cal_mag, 'k-', label='Magnitude')[0]
    ]
    
    # Plot formatting
    ax1.set_title('Raw Accelerometer Data')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend()
    ax1.grid(True)
    ax1.axhline(y=GRAVITY, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    ax2.set_title('Calibrated Accelerometer Data')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.legend()
    ax2.grid(True)
    ax2.axhline(y=GRAVITY, color='k', linestyle='--', alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    try:
        # Open serial connection
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Connected to {args.port}")
        time.sleep(2)  # Allow time for connection to establish
        
        # Clear buffer
        ser.reset_input_buffer()
        
        start_time = time.time()
        counter = 0
        
        print("Press Ctrl+C to exit")
        
        while True:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith('Raw:'):
                    data = line.replace('Raw:', '').split(',')
                    acc_x, acc_y, acc_z = float(data[0]), float(data[1]), float(data[2])
                    
                    # Shift data arrays
                    raw_data = np.roll(raw_data, -1, axis=0)
                    cal_data = np.roll(cal_data, -1, axis=0)
                    time_data = np.roll(time_data, -1)
                    raw_mag = np.roll(raw_mag, -1)
                    cal_mag = np.roll(cal_mag, -1)
                    
                    # Add new data point
                    current_time = time.time() - start_time
                    time_data[-1] = current_time
                    raw_data[-1] = [acc_x, acc_y, acc_z]
                    
                    # Apply calibration
                    cal_values = apply_calibration(np.array([acc_x, acc_y, acc_z]), offsets, scale_factors)
                    cal_data[-1] = cal_values
                    
                    # Calculate magnitudes
                    raw_mag[-1] = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
                    cal_mag[-1] = np.sqrt(cal_values[0]**2 + cal_values[1]**2 + cal_values[2]**2)
                    
                    # Print data occasionally
                    counter += 1
                    if counter % 10 == 0:
                        print(f"Raw: X={acc_x:.2f}, Y={acc_y:.2f}, Z={acc_z:.2f}, Mag={raw_mag[-1]:.2f}")
                        print(f"Cal: X={cal_values[0]:.2f}, Y={cal_values[1]:.2f}, Z={cal_values[2]:.2f}, Mag={cal_mag[-1]:.2f}")
                        print("---")
                    
                    # Update plots
                    for i in range(3):
                        raw_lines[i].set_data(time_data, raw_data[:, i])
                        cal_lines[i].set_data(time_data, cal_data[:, i])
                    
                    raw_lines[3].set_data(time_data, raw_mag)
                    cal_lines[3].set_data(time_data, cal_mag)
                    
                    # Adjust axes
                    ax1.set_xlim(time_data[0], time_data[-1])
                    ax2.set_xlim(time_data[0], time_data[-1])
                    
                    y_min = min(np.min(raw_data), np.min(cal_data)) - 1
                    y_max = max(np.max(raw_data), np.max(cal_data)) + 1
                    
                    ax1.set_ylim(y_min, y_max)
                    ax2.set_ylim(y_min, y_max)
                    
                    # Update plot
                    plt.pause(0.01)
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
    finally:
        # Close the serial connection
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed")
        
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()