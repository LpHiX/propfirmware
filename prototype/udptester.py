import socket
import sys
import json
import time
import threading

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Socket created")
    sock.settimeout(5)
    print("Timeout set to 5 seconds")
    #host = 'raspberrypi.local'
    #host_ip = socket.gethostbyname(host)
    host_ip = '192.168.137.216'
    #host_ip = "localhost"
    print(f"Host IP: {host_ip}")
    port = 8888
    try:
        # while True:
            with open('testin.json', 'r') as f:
                data = json.load(f)
            message = json.dumps(data)
            
            sock.sendto(message.encode(), (host_ip,port))
            # #print("sent")
            response, server = sock.recvfrom(4096)
            print(f"Received: {response.decode()}")
            time.sleep(.1)
    except KeyboardInterrupt:
        sock.close()