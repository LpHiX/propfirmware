from PySide6.QtWidgets import QLabel, QPushButton
from ui.widgets.actuators.base import ActuatorControllerWidget

class SolenoidControllerWidget(ActuatorControllerWidget):
    def __init__(self, config, board_name, actuator_name, udpmanager):
        super().__init__(config, board_name, actuator_name, "solenoids", udpmanager)
    
    def get_actuator_type_name(self):
        return "Solenoid"
    
    def setup_specific_ui(self):
        # Solenoid-specific UI elements
        self.powered_label = QLabel("Powered: No data")
        self.powered_label.setFixedWidth(self.LABEL_WIDTH)
        self.layout.addWidget(self.powered_label)
        
        self.poweron_button = QPushButton("Power On")
        self.poweron_button.setFixedWidth(self.BUTTON_WIDTH)
        self.poweron_button.clicked.connect(lambda: self.send_command("powered", True))
        
        self.poweroff_button = QPushButton("Power Off")
        self.poweroff_button.setFixedWidth(self.BUTTON_WIDTH)
        self.poweroff_button.clicked.connect(lambda: self.send_command("powered", False))
        
        self.layout.addWidget(self.poweron_button)
        self.layout.addWidget(self.poweroff_button)
    
    def update_specific_states(self, states):
        try:
            powered = states[self.actuator_type][self.actuator_name]["powered"]
            self.powered_label.setText(f"Powered: {powered}")
        except KeyError:
            self.powered_label.setText("Powered: No data")
