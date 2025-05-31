from PySide6.QtCore import QObject, Signal, QDateTime
import json
from typing import Dict, List, Any

class DataManager(QObject):
    hardwareConfigUpdated = Signal(dict)
    boardStatesUpdated = Signal(str, dict)  # board_name, states
    machineStateUpdated = Signal(str)
    timeUpdated = Signal(str, str)  # datetime_str, hotfiretime_str
    
    def __init__(self):
        super().__init__()
        self.hardware_json = None
        self.state_defaults = {}
        self.hardware_types = []
        self.last_update_time = None
        self.actuator_widgets = {}
        self.sensor_widgets = {}
        
    def set_hardware_json(self, hardware_json_str):
        try:
            hardware_json = json.loads(hardware_json_str)
            self.hardware_json = hardware_json
            
            if "state_defaults" not in hardware_json:
                print("No state defaults in hardware json")
                return False
                
            self.state_defaults = hardware_json['state_defaults']
            self.hardware_types = list(self.state_defaults.keys())
            self.hardwareConfigUpdated.emit(hardware_json)
            return True
        except json.JSONDecodeError:
            print("Hardware json is not valid JSON")
            return False
            
    def update_board_states(self, states_json_str):
        try:
            states = json.loads(states_json_str)
            for board_name, board_states in states.items():
                if self.hardware_json and board_name in self.hardware_json.get("boards", {}):
                    self.boardStatesUpdated.emit(board_name, board_states)
            self.last_update_time = QDateTime.currentDateTime()
            return True
        except json.JSONDecodeError:
            print("Board states json is not valid JSON")
            return False
            
    def update_machine_state(self, state_str):
        try:
            state = json.loads(state_str)
            self.machineStateUpdated.emit(state)
            return True
        except json.JSONDecodeError:
            print("Machine state is not valid JSON")
            return False
            
    def update_system_time(self, time_json_str):
        try:
            time_data = json.loads(time_json_str)
            datetime_str = time_data.get("date_time", "No date_time")
            hotfiretime_str = time_data.get("hotfire_time", "No hotfire_time")
            self.timeUpdated.emit(datetime_str, hotfiretime_str)
            return True
        except json.JSONDecodeError:
            print("Time data is not valid JSON")
            return False
            
    def get_last_update_elapsed_time(self):
        if self.last_update_time is None:
            return None
        current_time = QDateTime.currentDateTime()
        return self.last_update_time.msecsTo(current_time)
        
    def register_actuator_widget(self, board_name, actuator_type, actuator_name, widget):
        key = f"{board_name}_{actuator_type}_{actuator_name}"
        self.actuator_widgets[key] = widget
        
    def register_sensor_widget(self, board_name, sensor_type, sensor_name, widget):
        key = f"{board_name}_{sensor_type}_{sensor_name}"
        self.sensor_widgets[key] = widget
        
    def clear_all_widgets(self):
        self.actuator_widgets = {}
        self.sensor_widgets = {}
