from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QLineEdit
import sys
import pyqtgraph as pg
import numpy as np
import time
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import QTimer
start_time = time.time()

def composite_widget(offset):
    container = QWidget()
    v_layout = QVBoxLayout(container)
    plot = pg.PlotWidget()
    curve = plot.plot([], pen='y')
    v_layout.addWidget(plot)
    button = QPushButton(f"Arm {offset}")
    button1 = QPushButton(f"Shutoff {offset}")
    button2 = QPushButton(f"Full Open {offset}")
    button3 = QLineEdit("0")
    button3.maxLength= 3
    v_layout.addWidget(button)
    v_layout.addWidget(button1)
    v_layout.addWidget(button2)
    v_layout.addWidget(button3)

    def update_plot():
        t = time.time() - start_time
        x = np.linspace(0, 10, 500)
        y = np.sin(x + t + offset) + offset
        curve.setData(x, y)

    timer = QTimer(container)
    timer.timeout.connect(update_plot)
    timer.start(50)
    return container

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("3 Buttons in 3 Columns")

layout = QHBoxLayout()

comp0 = composite_widget(0)
comp1 = composite_widget(1)
comp2 = composite_widget(2)

layout.addWidget(comp0)
layout.addWidget(comp1)
layout.addWidget(comp2)

window.setLayout(layout)
window.resize(300, 100)
window.show()

sys.exit(app.exec())