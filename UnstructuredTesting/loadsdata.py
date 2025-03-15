import sys
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg

import pandas
import matplotlib.pyplot as plt
from rocketcea.cea_obj_w_units import CEA_Obj

STARTTIME = 0
ENDTIME = STARTTIME+5

CEA = CEA_Obj(
    oxName='N2O',
    fuelName='Isopropanol',
    isp_units='sec',
    cstar_units='m/s',
    pressure_units='Bar',
    temperature_units='K',
    sonic_velocity_units='m/s',
    enthalpy_units='J/g',
    density_units='kg/m^3',
    specific_heat_units='J/kg-K',
    viscosity_units='centipoise',  # stored value in pa-s
    thermal_cond_units='W/cm-degC',  # stored value in W/m-K
    fac_CR=8.7248,
    make_debug_prints=False)

class DAQView(QtWidgets.QMainWindow):
    def __init__(self, sensor_data):
        super().__init__()
        self.setWindowTitle("DAQ View")
        self.resize(1100, 700)
        
        # Save sensor_data (only individual sensors with tuple values)
        self.sensor_data = sensor_data
        
        # Keep references to the created individual tabs and combined plots
        self.plots = {}          # key: sensor name -> individual pg.PlotWidget
        self.combined_plots = {} # key: combined name -> widget
        
        # Create side panel controls
        self.availableList = QtWidgets.QListWidget()
        self.availableList.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        # Only list individual sensors (tuple values)
        for name, data in sensor_data.items():
            if isinstance(data, tuple):
                self.availableList.addItem(name)
        
        combine_btn = QtWidgets.QPushButton("Combine Selected")
        combine_btn.clicked.connect(self.combine_selected)
        
        self.combinedList = QtWidgets.QListWidget()
        self.combinedList.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        remove_btn = QtWidgets.QPushButton("Remove Selected Combined")
        remove_btn.clicked.connect(self.remove_selected_combined)
        
        sideLayout = QtWidgets.QVBoxLayout()
        sideLayout.addWidget(QtWidgets.QLabel("Available Sensors:"))
        sideLayout.addWidget(self.availableList)
        sideLayout.addWidget(combine_btn)
        sideLayout.addSpacing(20)
        sideLayout.addWidget(QtWidgets.QLabel("Combined Plots:"))
        sideLayout.addWidget(self.combinedList)
        sideLayout.addWidget(remove_btn)
        sideWidget = QtWidgets.QWidget()
        sideWidget.setLayout(sideLayout)
        sideWidget.setMinimumWidth(200)
        
        # Create a tab widget to hold sensor plots (both individual and combined)
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Add the individual sensor tabs initially
        for sensor_name, data in sensor_data.items():
            if isinstance(data, tuple):
                tab = QtWidgets.QWidget()
                tab_layout = QtWidgets.QVBoxLayout()
                tab.setLayout(tab_layout)
                
                plot = pg.PlotWidget(title=sensor_name)
                plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
                time_arr, sensor_arr = data
                plot.plot(time_arr, sensor_arr, pen=pg.mkPen('b', width=2))
                tab_layout.addWidget(plot)
                
                self.tab_widget.addTab(tab, sensor_name)
                self.plots[sensor_name] = plot
        
        # Create a horizontal splitter so that the side panel and tab widget are visible
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(sideWidget)
        splitter.addWidget(self.tab_widget)
        splitter.setStretchFactor(1, 10)
        
        self.setCentralWidget(splitter)
        
        # Add a reset zoom button below the tab widget
        reset_btn = QtWidgets.QPushButton("Reset Zoom")
        reset_btn.clicked.connect(self.reset_zoom)
        bottomLayout = QtWidgets.QVBoxLayout()
        bottomLayout.addWidget(reset_btn)
        bottomWidget = QtWidgets.QWidget()
        bottomWidget.setLayout(bottomLayout)
        # Place bottomWidget at the bottom of the main window
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, QtWidgets.QDockWidget("", self, widget=bottomWidget))
    
    def combine_selected(self):
        # Require at least 2 sensors to combine.
        selected = self.availableList.selectedItems()
        if len(selected) < 2:
            QtWidgets.QMessageBox.warning(self, "Selection Error", "Please select at least two sensors to combine.")
            return
        # Build a combined name by joining sensor names with commas.
        sensor_names = [item.text() for item in selected]
        combined_name = "Combined: " + ", ".join(sensor_names)
        if combined_name in self.combined_plots:
            QtWidgets.QMessageBox.information(self, "Already Combined", "This combination already exists.")
            return

        # Create new combined tab.
        tab = QtWidgets.QWidget()
        tab_layout = QtWidgets.QVBoxLayout()
        tab.setLayout(tab_layout)

        plot = pg.PlotWidget(title=combined_name)
        plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        # Add a legend to identify curves.
        plot.addLegend(offset=(10, 10))

        # Predefined color list; extend with additional colors if needed.
        colors = ["b", "r", "g", "c", "m", "y", "w"]
        num_colors = len(colors)

        for i, sensor in enumerate(sensor_names):
            time_arr, sensor_arr = self.sensor_data[sensor]
            color = colors[i % num_colors]  # cycle through colors if there are more sensors than available colors.
            plot.plot(
                time_arr, sensor_arr,
                pen=pg.mkPen(color, width=2),
                name=sensor
            )

        tab_layout.addWidget(plot)
        self.tab_widget.addTab(tab, combined_name)
        self.combined_plots[combined_name] = tab
        # Add to the combined plots list on the side panel.
        self.combinedList.addItem(combined_name)
        
    def remove_selected_combined(self):
        selected = self.combinedList.selectedItems()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "Selection Error", "Please select a combined plot to remove.")
            return
        combined_name = selected[0].text()
        # Find and remove the tab with this combined plot.
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == combined_name:
                self.tab_widget.removeTab(i)
                break
        # Remove from our dictionary and list widget.
        if combined_name in self.combined_plots:
            del self.combined_plots[combined_name]
        row = self.combinedList.row(selected[0])
        self.combinedList.takeItem(row)
    
    def reset_zoom(self):
        # Reset auto range on current tab.
        current_index = self.tab_widget.currentIndex()
        if current_index < 0:
            return
        sensor_name = self.tab_widget.tabText(current_index)
        # If it's an individual sensor tab.
        if sensor_name in self.plots:
            self.plots[sensor_name].enableAutoRange()
        else:
            # For a combined tab, find the PlotWidget within the tab.
            tab = self.tab_widget.widget(current_index)
            plot = tab.findChild(pg.PlotWidget)
            if plot is not None:
                plot.enableAutoRange()

if __name__ == "__main__":
    # Read data
    kermit_cf = pandas.read_csv("20250223_102653.597438Z_sts_sen0_telem.csv")
    stark_cf = pandas.read_csv("20250223_102653.491824Z_sts_stark_telem.csv")

    min_time = stark_cf['BackendTime'][0] + 5374.5 * 1e3
    kermit_cf = pandas.DataFrame(kermit_cf[kermit_cf['BackendTime'] > min_time])
    stark_cf = pandas.DataFrame(stark_cf[stark_cf['BackendTime'] > min_time])

    # Process time columns (in seconds)
    stark_cfTime = stark_cf['BackendTime']
    kermit_cfTime = kermit_cf['BackendTime']
    kermit_cfTime = (kermit_cfTime - stark_cfTime.iloc[0]) / 1e3
    stark_cfTime = (stark_cfTime - stark_cfTime.iloc[0]) / 1e3

    stark_cfTime = stark_cfTime.to_numpy()
    kermit_cfTime = kermit_cfTime.to_numpy()

    # Prepare sensor data dictionary for individual plots: key -> (time array, sensor data)
    sensor_data = {
        "thrust": (kermit_cfTime, kermit_cf['ch3sens'].to_numpy()),
        "chamberP": (stark_cfTime, stark_cf['ch3sens'].to_numpy()),
        "wfuelinjP": (stark_cfTime, stark_cf['ch1sens'].to_numpy()),
        "cfuelinjP": (stark_cfTime, stark_cf['ch2sens'].to_numpy()),
        "oxinjP": (stark_cfTime, stark_cf['ch0sens'].to_numpy()),
        "flowmeter": (stark_cfTime, stark_cf['flowmeter'].to_numpy()),
        "N2TankP": (kermit_cfTime, kermit_cf['ch1sens'].to_numpy()),
        "oxAngle": (stark_cfTime, stark_cf['oxAngle'].to_numpy()),
        "fuelAngle": (stark_cfTime, stark_cf['fuelAngle'].to_numpy()),
        "engineT": (kermit_cfTime, kermit_cf['temp1'].to_numpy()),
        "oxTankKg": (kermit_cfTime, kermit_cf['ch2sens'].to_numpy())
    }
    # Add individual pressure sensors (they remain available for combining)
    fuelTankP = (stark_cfTime, stark_cf['ch4sens'].to_numpy())
    oxTankP = (kermit_cfTime, kermit_cf['ch0sens'].to_numpy())
    
    # Optionally, you could also add a pre-combined tab by providing a dict;
    # in this example we leave combination to the GUI.
    
    app = QtWidgets.QApplication(sys.argv)
    view = DAQView(sensor_data)
    view.show()
    sys.exit(app.exec_())