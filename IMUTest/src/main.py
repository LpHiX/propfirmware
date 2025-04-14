from gy87parser import GY87Parser, Cube3DVisualizer
import sys
import argparse
import math
import time

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='IMU 3D Visualization')
    parser.add_argument('-p', '--port', default='COM3', help='Serial port name (default: COM3)')
    parser.add_argument('-b', '--baud', type=int, default=115200, help='Baud rate (default: 115200)')
    parser.add_argument('-d', '--demo', action='store_true', help='Start in demo mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('-s', '--scale', type=float, default=5.0, 
                        help='Rotation scale factor - increase for more visible movement (default: 5.0)')
    
    args = parser.parse_args()
    
    print(f"Starting IMU visualization...")
    print(f"Port: {args.port}, Baud rate: {args.baud}")
    
    # Create the parser instance
    parser = GY87Parser(args.port, args.baud)
    
    # Set optional modes
    parser.demo_mode = args.demo or parser.demo_mode
    parser.debug = args.debug
    parser.rotation_scale = args.scale
    
    # Create the visualizer
    viz = Cube3DVisualizer(width=1024, height=768)
    
    # Example: manually set position and orientation
    viz.set_position(0, 0, 0)
    viz.set_orientation(0, 0, 0)
    
    # Start the visualization
    print("3D Visualization started")
    print("Use Escape to quit")
    viz.run()

if __name__ == "__main__":
    main()
