from PySide6.QtWidgets import (QApplication)
from clientapp import ClientApp
from udpclient import UDPClient
import sys
import signal

if __name__ == "__main__":
    # host = "192.168.137.216"
    host = 'localhost'
    port = 8888

    app = QApplication(sys.argv)

    udpclient = UDPClient(host, port)
    window = ClientApp(udpclient)
    window.show()

    def signal_handler(sig, frame):
        print("Exiting...")
        app.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    sys.exit(app.exec())