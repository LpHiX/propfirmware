import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from io import StringIO  # Add this import for StringIO

def read_test_data(file_path):
    """
    Read and parse the test data from CSV file, skipping the header comment.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return None
    
    # Read the file content to handle the comment line
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip the comment line if it exists
    data_lines = lines[1:] if lines[0].startswith('#') else lines
    
    # Parse the CSV data using pandas - fixed StringIO usage
    data = pd.read_csv(StringIO(''.join(data_lines)))
    
    return data

def plot_pressures(data):
    """
    Plot all pressure sensor readings on a single plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot sensor data
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pressure Reading (bar)')
    ax.plot(data['timestamp'], 0.028169 * (data['Sensorboard0_pts_Chamber_mv'] - 859), 'b-', linewidth=2, label='Chamber Pressure')
    ax.plot(data['timestamp'], 0.00842697 * (data['Sensorboard0_pts_FuelCALIBRATED_mv'] - 746.5), 'g-', linewidth=2, label='Fuel Pressure')
    ax.plot(data['timestamp'], 0.0287081 * (data['Sensorboard0_pts_OxCALIBRATED_mv'] - 868.16), 'r-', linewidth=2, label='Oxidizer Pressure')
    
    # Add grid, legend and title
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=12)
    plt.title('Pressure Sensor Readings', fontsize=16)
    
    # Better formatting
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('pressure_readings.png')
    plt.show()

def main():
    # File path
    file_path = "testdata.csv"
    
    # Read the data
    data = read_test_data(file_path)
    
    if data is not None:
        # Plot only pressure data
        plot_pressures(data)
        
        print(f"Successfully plotted pressure data from {file_path}")
    else:
        print("Failed to read data file.")

if __name__ == "__main__":
    main()
