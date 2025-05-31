from PySide6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout
from PySide6.QtCore import Qt

class ActuatorControllerWidget(QWidget):
    """Base class for all actuator controller widgets"""
    def __init__(self, config, board_name, actuator_name, actuator_type, udpmanager):
        super().__init__()
        self.config = config
        self.board_name = board_name
        self.actuator_name = actuator_name
        self.actuator_type = actuator_type  # "servos", "solenoids", etc.
        self.udpmanager = udpmanager
        
        # Define standard widths
        self.LABEL_WIDTH = 150
        self.BUTTON_WIDTH = 100
        self.INPUT_WIDTH = 100
        
        # Create common layout
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        
        # Create common UI elements
        self.name_label = QLabel(f"{self.get_actuator_type_name()}: {self.actuator_name}")
        self.name_label.setFixedWidth(self.LABEL_WIDTH + 150)
        self.layout.addWidget(self.name_label)
        
        # Armed state is common to most actuators
        self.armed_label = QLabel(f"Armed: No data")
        self.armed_label.setFixedWidth(self.LABEL_WIDTH)
        self.layout.addWidget(self.armed_label)
        
        # Common arm/disarm buttons
        self.arm_button = QPushButton("Arm")
        self.arm_button.setFixedWidth(self.BUTTON_WIDTH)
        self.arm_button.clicked.connect(lambda: self.send_command("armed", True))
        
        self.disarm_button = QPushButton("Disarm")
        self.disarm_button.setFixedWidth(self.BUTTON_WIDTH)
        self.disarm_button.clicked.connect(lambda: self.send_command("armed", False))
        
        self.layout.addWidget(self.arm_button)
        self.layout.addWidget(self.disarm_button)
        
        # Add specific UI elements
        self.setup_specific_ui()
        self.layout.addStretch()
    
    def send_command(self, command, value):
        """Send command to the actuator"""
        self.udpmanager.send_actuator_command(
            self.board_name, 
            self.actuator_type, 
            self.actuator_name, 
            command, 
            value
        )
    
    def update_common_states(self, states):
        """Update common states like armed status"""
        try:
            armed = states[self.actuator_type][self.actuator_name]["armed"]
            self.armed_label.setText(f"Armed: {armed}")
        except KeyError:
            self.armed_label.setText("Armed: No data")
    
    def update_states(self, states):
        """Update widget with current states"""
        self.update_common_states(states)
        self.update_specific_states(states)
    
    def setup_specific_ui(self):
        """Set up actuator-specific UI elements. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement setup_specific_ui()")
    
    def update_specific_states(self, states):
        """Update actuator-specific states. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement update_specific_states()")
    
    def get_actuator_type_name(self):
        """Return human-readable actuator type name. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement get_actuator_type_name()")
