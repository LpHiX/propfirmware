import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QTimer
import pyqtgraph as pg

class SineWaveWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Animated Sine Wave")
        self.setGeometry(100, 100, 800, 500)
        
        # Create a central widget and layout
        central_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(central_widget)
        
        # Create a plot
        self.plot = central_widget.addPlot(title="Sine Wave Animation")
        self.plot.setLabel('left', 'Amplitude')
        self.plot.setLabel('bottom', 'X')
        self.plot.setYRange(-1.5, 1.5)
        
        # Create a curve item for the plot
        self.curve = self.plot.plot(pen=pg.mkPen('y', width=2))
        
        # Set up data
        self.x = np.linspace(0, 10*np.pi, 1000)
        self.time = 0
        
        # Set up timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(30)  # Update every 30ms
    
    def update_plot(self):
        # Update time value
        self.time += 0.1
        
        # Calculate y values: sine(x + time)
        y = np.sin(self.x + self.time)
        
        # Update the plot data
        self.curve.setData(self.x, y)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SineWaveWindow()
    window.show()
    sys.exit(app.exec())
