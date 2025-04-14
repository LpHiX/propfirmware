#!/usr/bin/env python3
"""
IMU Sensor Calibration Tool

This script takes raw accelerometer, gyroscope, and magnetometer data 
and performs calibration to provide offset and scale factors.

Usage:
1. Capture raw sensor data in various orientations
2. Either pipe data to this script or save to a file and specify with --file
3. Run the script to generate calibration values

Example: 
    python imu_calibration.py --port COM3 --baudrate 115200
    python imu_calibration.py --file sensor_data.txt
"""

import argparse
import numpy as np
import serial
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys
import re
from datetime import datetime

# Default gravity value in m/s²
GRAVITY = 9.81

class SensorCalibrator:
    def __init__(self):
        # Initialize data storage
        self.accel_data = []
        self.gyro_data = []
        self.mag_data = []
        
        # Initialize calibration parameters
        self.accel_bias = np.zeros(3)
        self.accel_scale = np.ones(3)
        self.gyro_bias = np.zeros(3)
        self.mag_bias = np.zeros(3)
        self.mag_scale = np.ones(3)
        
        # Initialize data collection state
        self.samples_collected = 0
        self.target_samples = 1000
        self.is_collecting = False
        
    def parse_data_line(self, line):
        """Parse a single line of raw sensor data"""
        # Expected format: "Raw:accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z"
        if not line.startswith("Raw:"):
            return False
            
        try:
            # Extract values after "Raw:" prefix
            data_part = line[4:].strip()
            values = [float(x) for x in data_part.split(',')]
            
            if len(values) != 9:
                print(f"Warning: Invalid data format, expected 9 values, got {len(values)}")
                return False
                
            # Add the parsed values to the respective data lists
            self.accel_data.append(values[0:3])
            self.gyro_data.append(values[3:6])
            self.mag_data.append(values[6:9])
            
            self.samples_collected += 1
            return True
        except Exception as e:
            print(f"Error parsing data: {e}")
            return False
    
    def calibrate_accelerometer(self):
        """
        Calibrate the accelerometer using the assumption that the magnitude 
        of the acceleration vector should be equal to gravity (9.81 m/s²) 
        when the device is stationary.
        """
        if len(self.accel_data) < 100:
            print("Insufficient accelerometer data for calibration")
            return False
            
        # Convert to numpy array for easier calculations
        accel_array = np.array(self.accel_data)
        
        # Calculate mean (bias)
        self.accel_bias = np.mean(accel_array, axis=0)
        
        # Remove bias
        accel_centered = accel_array - self.accel_bias
        
        # Calculate average magnitude
        magnitudes = np.sqrt(np.sum(accel_centered**2, axis=1))
        avg_magnitude = np.mean(magnitudes)
        
        # Scale factor to convert to correct gravity value
        self.accel_scale = GRAVITY / avg_magnitude
        
        print(f"Accelerometer Calibration Results:")
        print(f"Bias (Offset): X={self.accel_bias[0]:.4f}, Y={self.accel_bias[1]:.4f}, Z={self.accel_bias[2]:.4f}")
        print(f"Scale Factor: {self.accel_scale:.6f}")
        
        return True
        
    def calibrate_gyroscope(self):
        """
        Calibrate the gyroscope by estimating the zero-rate level (bias)
        when the device is stationary.
        """
        if len(self.gyro_data) < 100:
            print("Insufficient gyroscope data for calibration")
            return False
            
        # Convert to numpy array
        gyro_array = np.array(self.gyro_data)
        
        # Calculate mean (bias) - should be close to zero when stationary
        self.gyro_bias = np.mean(gyro_array, axis=0)
        
        # Calculate standard deviation to see how noisy the measurements are
        gyro_std = np.std(gyro_array, axis=0)
        
        print(f"Gyroscope Calibration Results:")
        print(f"Bias (Offset): X={self.gyro_bias[0]:.6f}, Y={self.gyro_bias[1]:.6f}, Z={self.gyro_bias[2]:.6f} rad/s")
        print(f"Noise (Std Dev): X={gyro_std[0]:.6f}, Y={gyro_std[1]:.6f}, Z={gyro_std[2]:.6f} rad/s")
        
        return True
        
    def calibrate_magnetometer(self):
        """
        Calibrate the magnetometer by finding the hard-iron (bias) and 
        soft-iron (scale) corrections to form a sphere of data.
        
        For accurate calibration, data should be collected while rotating
        the device in a figure-8 pattern to cover all orientations.
        """
        if len(self.mag_data) < 100:
            print("Insufficient magnetometer data for calibration")
            return False
            
        # Convert to numpy array
        mag_array = np.array(self.mag_data)
        
        # Find min and max values for each axis
        mag_min = np.min(mag_array, axis=0)
        mag_max = np.max(mag_array, axis=0)
        
        # Hard-iron correction (bias)
        self.mag_bias = (mag_max + mag_min) / 2
        
        # Soft-iron correction (scale)
        avg_delta = np.mean((mag_max - mag_min) / 2)
        self.mag_scale = avg_delta / ((mag_max - mag_min) / 2)
        
        print(f"Magnetometer Calibration Results:")
        print(f"Hard-Iron Bias: X={self.mag_bias[0]:.2f}, Y={self.mag_bias[1]:.2f}, Z={self.mag_bias[2]:.2f}")
        print(f"Soft-Iron Scale: X={self.mag_scale[0]:.4f}, Y={self.mag_scale[1]:.4f}, Z={self.mag_scale[2]:.4f}")
        
        return True
    
    def plot_data(self):
        """Plot raw and calibrated sensor data"""
        if not self.accel_data or not self.gyro_data or not self.mag_data:
            print("No data to plot")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Convert data to numpy arrays
        accel_array = np.array(self.accel_data)
        gyro_array = np.array(self.gyro_data)
        mag_array = np.array(self.mag_data)
        
        # Calibrated data
        accel_calibrated = (accel_array - self.accel_bias) * self.accel_scale
        gyro_calibrated = gyro_array - self.gyro_bias
        mag_calibrated = (mag_array - self.mag_bias) * self.mag_scale
        
        # === Accelerometer Plots ===
        # Raw accelerometer data 3D
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(accel_array[:, 0], accel_array[:, 1], accel_array[:, 2], c='r', marker='o', alpha=0.5)
        ax1.set_title('Raw Accelerometer Data')
        ax1.set_xlabel('X (m/s²)')
        ax1.set_ylabel('Y (m/s²)')
        ax1.set_zlabel('Z (m/s²)')
        
        # Calibrated accelerometer data 3D
        ax2 = fig.add_subplot(2, 3, 4, projection='3d')
        ax2.scatter(accel_calibrated[:, 0], accel_calibrated[:, 1], accel_calibrated[:, 2], c='g', marker='o', alpha=0.5)
        ax2.set_title('Calibrated Accelerometer Data')
        ax2.set_xlabel('X (m/s²)')
        ax2.set_ylabel('Y (m/s²)')
        ax2.set_zlabel('Z (m/s²)')
        
        # === Gyroscope Plots ===
        # Raw gyroscope data
        ax3 = fig.add_subplot(2, 3, 2)
        t = np.arange(len(gyro_array))
        ax3.plot(t, gyro_array[:, 0], 'r-', label='X')
        ax3.plot(t, gyro_array[:, 1], 'g-', label='Y')
        ax3.plot(t, gyro_array[:, 2], 'b-', label='Z')
        ax3.set_title('Raw Gyroscope Data')
        ax3.set_xlabel('Sample')
        ax3.set_ylabel('Angular Velocity (rad/s)')
        ax3.legend()
        ax3.grid(True)
        
        # Calibrated gyroscope data
        ax4 = fig.add_subplot(2, 3, 5)
        ax4.plot(t, gyro_calibrated[:, 0], 'r-', label='X')
        ax4.plot(t, gyro_calibrated[:, 1], 'g-', label='Y')
        ax4.plot(t, gyro_calibrated[:, 2], 'b-', label='Z')
        ax4.set_title('Calibrated Gyroscope Data')
        ax4.set_xlabel('Sample')
        ax4.set_ylabel('Angular Velocity (rad/s)')
        ax4.legend()
        ax4.grid(True)
        
        # === Magnetometer Plots ===
        # Raw magnetometer data 3D
        ax5 = fig.add_subplot(2, 3, 3, projection='3d')
        ax5.scatter(mag_array[:, 0], mag_array[:, 1], mag_array[:, 2], c='b', marker='o', alpha=0.5)
        ax5.set_title('Raw Magnetometer Data')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        
        # Calibrated magnetometer data 3D
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.scatter(mag_calibrated[:, 0], mag_calibrated[:, 1], mag_calibrated[:, 2], c='m', marker='o', alpha=0.5)
        ax6.set_title('Calibrated Magnetometer Data')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sensor_calibration_{timestamp}.png"
        plt.savefig(filename)
        print(f"Calibration plots saved to {filename}")
        
        # Show plots
        plt.show()
    
    def generate_arduino_code(self):
        """Generate Arduino-compatible code with calibration values"""
        arduino_code = """
// IMU Sensor Calibration Parameters
// Generated by imu_calibration.py on {}

// Accelerometer calibration
float accel_bias[3] = {{{:.6f}f, {:.6f}f, {:.6f}f}};  // X, Y, Z offsets
float accel_scale = {:.6f}f;  // Scale factor

// Gyroscope calibration
float gyro_bias[3] = {{{:.6f}f, {:.6f}f, {:.6f}f}};  // X, Y, Z offsets

// Magnetometer calibration
int16_t mag_bias[3] = {{{:.0f}, {:.0f}, {:.0f}}};  // X, Y, Z hard-iron offsets
float mag_scale[3] = {{{:.6f}f, {:.6f}f, {:.6f}f}};  // X, Y, Z soft-iron scale factors

// Function to apply calibration to accelerometer readings
void calibrateAccelerometer(float rawX, float rawY, float rawZ, float *calibX, float *calibY, float *calibZ) {{
  *calibX = (rawX - accel_bias[0]) * accel_scale;
  *calibY = (rawY - accel_bias[1]) * accel_scale;
  *calibZ = (rawZ - accel_bias[2]) * accel_scale;
}}

// Function to apply calibration to gyroscope readings
void calibrateGyroscope(float rawX, float rawY, float rawZ, float *calibX, float *calibY, float *calibZ) {{
  *calibX = rawX - gyro_bias[0];
  *calibY = rawY - gyro_bias[1];
  *calibZ = rawZ - gyro_bias[2];
}}

// Function to apply calibration to magnetometer readings
void calibrateMagnetometer(int16_t rawX, int16_t rawY, int16_t rawZ, float *calibX, float *calibY, float *calibZ) {{
  *calibX = (rawX - mag_bias[0]) * mag_scale[0];
  *calibY = (rawY - mag_bias[1]) * mag_scale[1];
  *calibZ = (rawZ - mag_bias[2]) * mag_scale[2];
}}
""".format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.accel_bias[0], self.accel_bias[1], self.accel_bias[2],
            self.accel_scale,
            self.gyro_bias[0], self.gyro_bias[1], self.gyro_bias[2],
            self.mag_bias[0], self.mag_bias[1], self.mag_bias[2],
            self.mag_scale[0], self.mag_scale[1], self.mag_scale[2]
        )
        
        # Save to file
        filename = "imu_calibration.h"
        with open(filename, "w") as f:
            f.write(arduino_code)
        
        print(f"Arduino calibration code saved to {filename}")
        return arduino_code
        
    def process_file(self, filename):
        """Process sensor data from a file"""
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            self.parse_data_line(line)
            
        print(f"Processed {self.samples_collected} data points from file")
        return self.samples_collected > 0
        
    def read_from_serial(self, port, baudrate, duration=30):
        """Collect sensor data from serial port for specified duration"""
        try:
            ser = serial.Serial(port, baudrate, timeout=1)
            start_time = time.time()
            
            print(f"Reading from {port} at {baudrate} baud for {duration} seconds...")
            print("Press Ctrl+C to stop early")
            
            while time.time() - start_time < duration:
                # Read a line from serial
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    if self.parse_data_line(line):
                        sys.stdout.write(f"\rCollected {self.samples_collected} samples...")
                        sys.stdout.flush()
            
            ser.close()
            print(f"\nCompleted data collection: {self.samples_collected} samples")
            return True
            
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            return False
        except KeyboardInterrupt:
            print(f"\nData collection stopped. Collected {self.samples_collected} samples")
            if ser and ser.is_open:
                ser.close()
            return self.samples_collected > 0

def main():
    parser = argparse.ArgumentParser(description='IMU Sensor Calibration Tool')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help='Input file with sensor data')
    group.add_argument('--port', help='Serial port for reading real-time data')
    parser.add_argument('--baudrate', type=int, default=115200, help='Serial port baudrate')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds to collect data')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    calibrator = SensorCalibrator()
    
    # Input processing
    if args.file:
        if not calibrator.process_file(args.file):
            print("Failed to process input file")
            return
    elif args.port:
        if not calibrator.read_from_serial(args.port, args.baudrate, args.duration):
            print("Failed to read from serial port")
            return
    
    # Calibration
    print("\n=== Running Calibration ===")
    accel_success = calibrator.calibrate_accelerometer()
    gyro_success = calibrator.calibrate_gyroscope()
    mag_success = calibrator.calibrate_magnetometer()
    
    if not accel_success or not gyro_success or not mag_success:
        print("WARNING: One or more calibrations failed")
    
    # Generate Arduino code
    print("\n=== Generating Arduino Code ===")
    calibrator.generate_arduino_code()
    
    # Plot data if requested
    if not args.no_plot:
        print("\n=== Generating Plots ===")
        calibrator.plot_data()
    
    print("\nCalibration complete!")

if __name__ == "__main__":
    main()