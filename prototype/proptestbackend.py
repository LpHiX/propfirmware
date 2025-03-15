# Two goals:
# Send UART command packets to actuator esp32
# Send UART packet request to DAQ esp32, then receive and process the response

import argparse
import serial

def parse_arguments():
    parser = argparse.ArgumentParser(description="Property Testing Backend")
    parser.add_argument("--device", "-d", help="Serial device path (e.g. COM3)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if args.device:
        print("Attempting connection to device:", args.device)
        try:
            ser = serial.Serial(args.device, baudrate=115200, timeout=1)
            print("Connected to device:", args.device)
        except serial.SerialException as e:
            print(f"Error connecting to device: {e}")
            exit(1)
    else:
        print("No device specified. Starting debug mode.")
        # Debug mode: simulate device interaction