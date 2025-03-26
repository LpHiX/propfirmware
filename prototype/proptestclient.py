import socket
import sys
import json

data = {
    "board_name": "ActuatorBoard",
    "message": {
        "solenoids": [
            {
                "gpio": 40,
                "armed": True,
                "powered": True,
                "name": "LED"
            },
        ]
    }
}

def send_udp_command(host='raspberrypi.local', port=8888):
    message = json.dumps({"command": "send", "data": data})
    
    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # Send data
        sock.sendto(message.encode(), (host, port))
        print(f"Sent: {message}")
        
        # Receive response
        sock.settimeout(5)
        response, server = sock.recvfrom(4096)
        print(f"Received: {response.decode()}")
        
    finally:
        sock.close()

if __name__ == "__main__":
    send_udp_command()