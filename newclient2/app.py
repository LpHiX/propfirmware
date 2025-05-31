import sys
import signal
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from ui.main_window import PropertyTestApp

def main():
    # host = "192.168.137.216"
    host = "127.0.0.1"
    port = 8888
    
    app = QApplication(sys.argv)
    window = PropertyTestApp(host, port)
    window.show()
    
    def signal_handler(sig, frame):
        print("Exiting...")
        app.quit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    timer = QTimer()
    timer.start(100)  # Small interval to check signals
    timer.timeout.connect(lambda: None)
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
