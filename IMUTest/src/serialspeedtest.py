import serial
import time
import sys
import threading
import queue

# Configuration
PORT = 'COM6'  # Change to your serial port
BAUD_RATE = 921600     # High baud rate
PACKET_SIZE = 50
REPORT_EVERY = 1000    # Print stats every 1000 packets
BUFFER_SIZE = 4096     # Larger buffer size for pyserial

# Create queues for thread communication
rx_queue = queue.Queue(maxsize=10000)
stats_queue = queue.Queue()

def receiver_thread(ser):
    """Thread dedicated to just receiving data"""
    packets_received = 0
    buffer = bytearray()
    
    while True:
        try:
            # Read all available data
            if ser.in_waiting > 0:
                new_data = ser.read(ser.in_waiting)
                buffer.extend(new_data)
                
                # Process complete packets
                while len(buffer) >= PACKET_SIZE:
                    packet = buffer[:PACKET_SIZE]
                    buffer = buffer[PACKET_SIZE:]
                    rx_queue.put(packet)
                    packets_received += 1
                    
                    # Update stats periodically
                    if packets_received % REPORT_EVERY == 0:
                        stats_queue.put(packets_received)
            else:
                # Short sleep when no data available
                time.sleep(0.0001)
                
        except Exception as e:
            print(f"Receiver error: {e}")
            time.sleep(0.1)

def main():
    try:
        # Open serial port with hardware flow control if available
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0,           # Non-blocking
            write_timeout=1,     # 1 second timeout for writes
            # Uncomment the next line if your hardware supports it
            # rtscts=True,       # Hardware flow control
            dsrdtr=False,
            inter_byte_timeout=None
        )
        
        ser.set_buffer_size(rx_size=BUFFER_SIZE, tx_size=BUFFER_SIZE)
        
        # Wait for port to be ready
        time.sleep(0.5)
        if not ser.is_open:
            ser.open()
            
        print(f"Connected to {PORT} at {BAUD_RATE} baud")
        
        # Start receiver thread
        rx_thread = threading.Thread(target=receiver_thread, args=(ser,), daemon=True)
        rx_thread.start()
        
        # Initialize counters
        packets_sent = 0
        packets_received = 0
        start_time = time.perf_counter()
        last_report_time = start_time
        last_packet_data = None
        
        # Prepare the static packet (only first byte changes)
        base_packet = bytearray(PACKET_SIZE)
        
        # Pre-allocate packet variants for speed
        packets = [bytearray(base_packet) for _ in range(256)]
        for i in range(256):
            packets[i][0] = i
        
        # Main loop - focus only on sending
        while True:
            try:
                # Send packet: first byte is the counter, rest are zeros
                packet_idx = packets_sent % 256
                ser.write(packets[packet_idx])
                packets_sent += 1
                
                # Process received data (non-blocking)
                while not rx_queue.empty():
                    last_packet_data = rx_queue.get_nowait()
                    packets_received += 1
                
                # Check for stats updates (non-blocking)
                if not stats_queue.empty():
                    stats_queue.get_nowait()  # Just clear the queue
                    current_time = time.perf_counter()
                    elapsed = current_time - last_report_time
                    total_elapsed = current_time - start_time
                    
                    # Calculate rates
                    packets_per_sec = REPORT_EVERY / elapsed if elapsed > 0 else 0
                    bytes_per_sec = packets_per_sec * PACKET_SIZE
                    
                    # Print stats
                    print(f"Received {packets_received} packets ({packets_received/packets_sent*100:.2f}%)")
                    print(f"Rate: {packets_per_sec:.2f} packets/sec ({bytes_per_sec/1024:.2f} KB/s)")
                    if last_packet_data:
                        print(f"Last packet: {' '.join([f'{b:02x}' for b in last_packet_data])}")
                    print(f"Elapsed time: {total_elapsed:.2f} seconds")
                    print("---")
                    
                    # Reset for next report
                    last_report_time = current_time
                
                # Optional tiny sleep to prevent CPU overload but keep high throughput
                # Adjust or remove if needed for optimal performance
                time.sleep(0.0001)  # 0.1ms delay
                
            except serial.SerialTimeoutException:
                print("Write timeout - buffer might be full, waiting...")
                time.sleep(0.01)
                continue
                
            except serial.SerialException as e:
                print(f"Serial error: {e}")
                break
                
            except KeyboardInterrupt:
                raise
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                break
            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
        # Final stats
        total_time = time.perf_counter() - start_time
        print(f"Total packets sent: {packets_sent}")
        print(f"Total packets received: {packets_received}")
        if packets_sent > 0:
            print(f"Packet loss: {(1 - packets_received/packets_sent)*100:.2f}%")
        print(f"Total time: {total_time:.2f} seconds")
        if total_time > 0:
            print(f"Average throughput: {packets_received*PACKET_SIZE/total_time/1024:.2f} KB/s")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    main()