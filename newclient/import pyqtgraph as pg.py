import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

app = QtWidgets.QApplication([])

# Create the plot widget
plot_widget = pg.PlotWidget()
plot_widget.setWindowTitle("PySide6 Safe Plot")
plot_widget.resize(600, 400)

# Disable OpenGL (safe fallback)
pg.setConfigOption('useOpenGL', False)

# Add the plot data explicitly
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]
curve = pg.PlotDataItem(x, y, pen='b')
plot_widget.addItem(curve)

# Show the widget first
plot_widget.show()

# Now set explicit view ranges to stop crazy auto-scaling
plot_widget.setXRange(min(x), max(x))
plot_widget.setYRange(min(y), max(y))
plot_widget.enableAutoRange(False)

app.exec()