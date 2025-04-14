import serial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
from scipy.optimize import least_squares

def read_serial_data(port='COM6', baud_rate=115200, timeout=10, num_samples=1000):
    """Read magnetometer data from serial port."""
    print(f"Opening serial port {port} and collecting {num_samples} samples...")
    
    mag_data = []
    try:
        with serial.Serial(port, baud_rate, timeout=timeout) as ser:
            count = 0
            while count < num_samples:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith('Raw:'):
                    try:
                        # Format: accel x y z, gyro x y z, mag x y z
                        data = line[4:].split(',')
                        if len(data) == 9:  # Make sure we have all 9 values
                            mag_x, mag_y, mag_z = float(data[6]), float(data[7]), float(data[8])
                            mag_data.append([mag_x, mag_y, mag_z])
                            count += 1
                            if count % 100 == 0:
                                print(f"Collected {count} samples...")
                    except ValueError as e:
                        print(f"Error parsing data: {e}")
                        continue
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None
    
    return np.array(mag_data)

def ellipsoid_fit(data):
    """
    Fit an ellipsoid to the magnetometer data.
    Based on the algorithm from: http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
    """
    # Design matrix
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    D = np.column_stack((x*x, y*y, z*z, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z, np.ones_like(x)))
    
    # Objective function to minimize
    S = np.dot(D.T, D)
    C = np.zeros([10, 10])
    C[0:6, 0:6] = np.eye(6)
    C[6:10, 6:10] = 0
    E = np.linalg.inv(S).dot(C)
    
    # Find the eigenvector with positive eigenvalue
    _, eigvecs = np.linalg.eig(E.dot(S))
    v = eigvecs[:, np.argmax(np.array([np.dot(eigvec.T, C).dot(eigvec) for eigvec in eigvecs.T]))]
    
    # If v[0] is negative, negate v
    if v[0] < 0:
        v = -v
    
    # Form the ellipsoid parameters
    a = np.array([
        [v[0], v[3], v[4], v[6]],
        [v[3], v[1], v[5], v[7]],
        [v[4], v[5], v[2], v[8]],
        [v[6], v[7], v[8], v[9]]
    ])
    
    # Center of the ellipsoid
    center = -np.linalg.solve(a[0:3, 0:3], v[6:9])
    
    # Form the rotation and scale matrix
    T = np.eye(4)
    T[0:3, 3] = center
    R = T.dot(a).dot(T.T)
    R = R / -R[3, 3]
    R = R[0:3, 0:3]
    
    # Get the calibration parameters
    eigenvalues, eigenvectors = np.linalg.eig(R)
    radii = 1.0 / np.sqrt(eigenvalues)
    
    # Compute the soft iron correction matrix
    A = eigenvectors.dot(np.diag(radii)).dot(eigenvectors.T)
    
    # Compute the hard iron correction (bias)
    b = center
    
    return A, b

def apply_calibration(data, A, b):
    """Apply calibration to the raw magnetometer data."""
    data_calibrated = np.zeros_like(data)
    for i in range(len(data)):
        data_calibrated[i] = A.dot(data[i] - b)
    return data_calibrated

def plot_magnetometer_data(raw_data, calibrated_data):
    """Plot raw and calibrated magnetometer data."""
    fig = plt.figure(figsize=(12, 6))
    
    # Raw data
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(raw_data[:, 0], raw_data[:, 1], raw_data[:, 2], c='r', marker='o')
    ax1.set_title('Raw Magnetometer Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Calibrated data
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(calibrated_data[:, 0], calibrated_data[:, 1], calibrated_data[:, 2], c='b', marker='o')
    ax2.set_title('Calibrated Magnetometer Data')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def simulate_data():
    """Simulate magnetometer data for testing."""
    # Example data provided by the user
    example_data = [
        [37, 1795, -2402],
        [52, 1813, -2400],
        [45, 1803, -2392],
        [47, 1797, -2382],
        [40, 1792, -2390],
        [40, 1811, -2397],
        [30, 1813, -2402]
    ]
    return np.array(example_data)

def main():
    # Get data source choice from user
    choice = input("Choose data source (1: Serial port, 2: Simulated data): ")
    
    if choice == '1':
        # Read from serial port
        num_samples = int(input("Enter number of samples to collect: "))
        mag_data = read_serial_data(num_samples=num_samples)
        if mag_data is None or len(mag_data) == 0:
            print("Failed to collect magnetometer data. Using simulated data instead.")
            mag_data = simulate_data()
    else:
        # Use simulated data
        print("Using simulated data based on provided examples.")
        mag_data = simulate_data()
    
    # Add noise and more points to simulated data for better fitting
    if len(mag_data) < 100 and choice == '2':
        # Generate more data points by adding noise to the existing ones
        np.random.seed(42)
        expanded_data = []
        for _ in range(50):
            for point in mag_data:
                noise = np.random.normal(0, 5, 3)  # Small Gaussian noise
                expanded_data.append(point + noise)
        mag_data = np.array(expanded_data)
    
    print(f"Collected {len(mag_data)} data points.")
    
    # Fit ellipsoid to get calibration parameters
    print("Fitting ellipsoid to magnetometer data...")
    A, b = ellipsoid_fit(mag_data)
    
    print("\nCalibration Results:")
    print("Soft Iron Correction Matrix (A):")
    print(A)
    print("\nHard Iron Bias Vector (b):")
    print(b)
    
    # Apply calibration
    mag_data_calibrated = apply_calibration(mag_data, A, b)
    
    # Plot results
    plot_magnetometer_data(mag_data, mag_data_calibrated)
    
    # Save calibration parameters
    np.savez('mag_calibration_params.npz', A=A, b=b)
    print("\nCalibration parameters saved to 'mag_calibration_params.npz'")

if __name__ == "__main__":
    main()