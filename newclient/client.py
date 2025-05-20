from PySide6.QtWidgets import (QApplication)
import sys
import signal

if __name__ == "__main__":
    host = "192.168.137.216"
    port = 8888

    app = QApplication(sys.argv)
    window = PropertyTestApp(host, port)
    window.show()

    def signal_handler(sig, frame):
        print("Exiting...")
        app.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    sys.exit(app.exec())