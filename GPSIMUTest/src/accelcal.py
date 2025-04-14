import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Serial port configuration
SERIAL_PORT = 'COM6'  # Your COM port
BAUD_RATE = 115200      # Adjust to match your setup

# Constants
GRAVITY = 9.81        # Standard gravity in m/s²
SAMPLE_SIZE = 500     # Number of samples to collect for calibration

def collect_calibration_data(ser, positions=6, samples_per_position=100):
    """
    Collect calibration data by placing sensor in 6 different orientations.
    
    The 6 positions are:
    1. Z-axis up (flat)
    2. Z-axis down (upside down)
    3. X-axis up (standing on right edge)
    4. X-axis down (standing on left edge)
    5. Y-axis up (standing on bottom edge)
    6. Y-axis down (standing on top edge)
    """
    calibration_data = []
    
    for position in range(positions):
        position_data = []
        
        if position == 0:
            print("\nPlace sensor FLAT with Z-axis UP (circuit board facing up)")
        elif position == 1:
            print("\nFlip sensor UPSIDE DOWN with Z-axis DOWN (circuit board facing down)")
        elif position == 2:
            print("\nStand sensor on its RIGHT EDGE (X-axis UP)")
        elif position == 3:
            print("\nStand sensor on its LEFT EDGE (X-axis DOWN)")
        elif position == 4:
            print("\nStand sensor on its BOTTOM EDGE (Y-axis UP)")
        elif position == 5:
            print("\nStand sensor on its TOP EDGE (Y-axis DOWN)")
            
        input("Press Enter when ready, then keep the sensor STILL...")
        print(f"Collecting data for position {position+1}/6...")
        
        # Clear any buffered data
        ser.reset_input_buffer()
        time.sleep(1)
        
        # Collect samples
        for _ in range(samples_per_position):
            try:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith('Raw:'):
                    data = line.replace('Raw:', '').split(',')
                    acc_x, acc_y, acc_z = float(data[0]), float(data[1]), float(data[2])
                    position_data.append([acc_x, acc_y, acc_z])
            except Exception as e:
                print(f"Error reading data: {e}")
                continue
        
        calibration_data.append(np.mean(position_data, axis=0))
        print(f"Position {position+1} average readings: {calibration_data[-1]}")
    
    return np.array(calibration_data)

def calculate_calibration_parameters(calibration_data):
    """
    Calculate offset and scale factors based on collected data.
    
    Assumes the order of positions is:
    0: Z-axis up   (x≈0, y≈0, z≈+g)
    1: Z-axis down (x≈0, y≈0, z≈-g)
    2: X-axis up   (x≈+g, y≈0, z≈0)
    3: X-axis down (x≈-g, y≈0, z≈0)
    4: Y-axis up   (x≈0, y≈+g, z≈0)
    5: Y-axis down (x≈0, y≈-g, z≈0)
    """
    # Extract readings for each axis
    x_readings = np.array([calibration_data[2][0], calibration_data[3][0]])  # X-axis up/down
    y_readings = np.array([calibration_data[4][1], calibration_data[5][1]])  # Y-axis up/down
    z_readings = np.array([calibration_data[0][2], calibration_data[1][2]])  # Z-axis up/down
    
    # Calculate offsets (bias) - the average of min and max readings
    offset_x = (x_readings[0] + x_readings[1]) / 2
    offset_y = (y_readings[0] + y_readings[1]) / 2
    offset_z = (z_readings[0] + z_readings[1]) / 2
    
    # Calculate scale factors
    # The difference between max and min readings should equal 2g
    scale_x = (x_readings[0] - x_readings[1]) / (2 * GRAVITY)
    scale_y = (y_readings[0] - y_readings[1]) / (2 * GRAVITY)
    scale_z = (z_readings[0] - z_readings[1]) / (2 * GRAVITY)
    
    # Create offset and scale factor arrays
    offsets = np.array([offset_x, offset_y, offset_z])
    scale_factors = np.array([scale_x, scale_y, scale_z])
    
    return offsets, scale_factors

def apply_calibration(raw_data, offsets, scale_factors):
    """Apply calibration to raw accelerometer data."""
    calibrated_data = np.zeros_like(raw_data)
    
    for i in range(3):  # For x, y, z axes
        calibrated_data[:, i] = (raw_data[:, i] - offsets[i]) / scale_factors[i]
    
    return calibrated_data

def main():
    print("===== MPU6050 Accelerometer Calibration =====")
    
    # Option to use quick calibration or advanced 6-position calibration
    mode = input("Choose calibration mode (1: Quick, 2: Advanced 6-position): ")
    
    try:
        # Open serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT}")
        time.sleep(2)  # Allow time for connection to establish
        
        if mode == "1":
            # Quick calibration - assuming sensor is placed flat (Z-axis up)
            print("\nPlace the sensor FLAT on a level surface.")
            input("Press Enter when ready, then keep the sensor STILL...")
            
            # Collect data for calibration
            print(f"Collecting {SAMPLE_SIZE} samples for calibration...")
            accel_data = []
            
            # Clear buffer
            ser.reset_input_buffer()
            time.sleep(1)
            
            # Collect samples
            for _ in range(SAMPLE_SIZE):
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line.startswith('Raw:'):
                        data = line.replace('Raw:', '').split(',')
                        acc_x, acc_y, acc_z = float(data[0]), float(data[1]), float(data[2])
                        accel_data.append([acc_x, acc_y, acc_z])
                except Exception as e:
                    print(f"Error reading data: {e}")
                    continue
            
            accel_data = np.array(accel_data)
            
            # Calculate simple offsets and scale factors
            avg_data = np.mean(accel_data, axis=0)
            print(f"Average readings: X={avg_data[0]:.3f}, Y={avg_data[1]:.3f}, Z={avg_data[2]:.3f}")
            
            # For simple calibration, we assume sensor is flat, so Z should be 9.81 and X,Y should be 0
            offsets = np.array([avg_data[0], avg_data[1], 0])
            
            # For Z-axis, calculate scale factor to normalize to 9.81
            z_scale = avg_data[2] / GRAVITY
            scale_factors = np.array([1.0, 1.0, z_scale])
            
            # Use a default scale for X and Y (can be refined with advanced calibration)
            print("\nQuick Calibration Results:")
            print(f"Offsets: X={offsets[0]:.5f}, Y={offsets[1]:.5f}, Z={offsets[2]:.5f}")
            print(f"Scale Factors: X={scale_factors[0]:.5f}, Y={scale_factors[1]:.5f}, Z={scale_factors[2]:.5f}")
            
        else:
            # Advanced 6-position calibration
            calibration_data = collect_calibration_data(ser)
            offsets, scale_factors = calculate_calibration_parameters(calibration_data)
            
            print("\nAdvanced Calibration Results:")
            print(f"Offsets: X={offsets[0]:.5f}, Y={offsets[1]:.5f}, Z={offsets[2]:.5f}")
            print(f"Scale Factors: X={scale_factors[0]:.5f}, Y={scale_factors[1]:.5f}, Z={scale_factors[2]:.5f}")
        
        # Save calibration parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"accel_calibration_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(f"Offsets: {offsets[0]:.6f}, {offsets[1]:.6f}, {offsets[2]:.6f}\n")
            f.write(f"Scale Factors: {scale_factors[0]:.6f}, {scale_factors[1]:.6f}, {scale_factors[2]:.6f}\n")
        print(f"\nCalibration parameters saved to {filename}")
        
        # Test calibration
        print("\nTesting calibration results...")
        print("Place the sensor flat (Z-axis up) and keep it still")
        input("Press Enter to begin testing...")
        
        # Clear buffer
        ser.reset_input_buffer()
        time.sleep(1)
        
        # Collect and process data
        test_data = []
        calibrated_data = []
        
        for _ in range(100):
            try:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith('Raw:'):
                    data = line.replace('Raw:', '').split(',')
                    acc_x, acc_y, acc_z = float(data[0]), float(data[1]), float(data[2])
                    raw = np.array([[acc_x, acc_y, acc_z]])
                    test_data.append([acc_x, acc_y, acc_z])
                    
                    # Apply calibration
                    cal = apply_calibration(raw, offsets, scale_factors)
                    calibrated_data.append(cal[0])
                    
                    # Print in real-time
                    mag_raw = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
                    mag_cal = np.sqrt(cal[0][0]**2 + cal[0][1]**2 + cal[0][2]**2)
                    print(f"Raw: X={acc_x:.2f}, Y={acc_y:.2f}, Z={acc_z:.2f}, Mag={mag_raw:.2f}")
                    print(f"Cal: X={cal[0][0]:.2f}, Y={cal[0][1]:.2f}, Z={cal[0][2]:.2f}, Mag={mag_cal:.2f}")
                    print("---")
            except Exception as e:
                print(f"Error reading data: {e}")
                continue
        
        # Convert to numpy arrays
        test_data = np.array(test_data)
        calibrated_data = np.array(calibrated_data)
        
        # Calculate magnitudes
        raw_mag = np.sqrt(np.sum(test_data**2, axis=1))
        cal_mag = np.sqrt(np.sum(calibrated_data**2, axis=1))
        
        # Plot results
        plt.figure(figsize=(12, 10))
        
        # Raw data plot
        plt.subplot(2, 1, 1)
        plt.plot(test_data[:, 0], 'r-', label='X-axis')
        plt.plot(test_data[:, 1], 'g-', label='Y-axis')
        plt.plot(test_data[:, 2], 'b-', label='Z-axis')
        plt.plot(raw_mag, 'k-', label='Magnitude')
        plt.axhline(y=GRAVITY, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Raw Accelerometer Data')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        plt.grid(True)
        
        # Calibrated data plot
        plt.subplot(2, 1, 2)
        plt.plot(calibrated_data[:, 0], 'r-', label='X-axis')
        plt.plot(calibrated_data[:, 1], 'g-', label='Y-axis')
        plt.plot(calibrated_data[:, 2], 'b-', label='Z-axis')
        plt.plot(cal_mag, 'k-', label='Magnitude')
        plt.axhline(y=GRAVITY, color='k', linestyle='--', alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.title('Calibrated Accelerometer Data')
        plt.xlabel('Sample Number')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"accel_calibration_results_{timestamp}.png")
        plt.show()
        
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
    finally:
        # Close the serial connection
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed")

if __name__ == "__main__":
    main()