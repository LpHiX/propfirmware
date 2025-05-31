from PySide6.QtWidgets import QLabel, QPushButton, QLineEdit
from ui.widgets.actuators.base import ActuatorControllerWidget

class ServoControllerWidget(ActuatorControllerWidget):
    def __init__(self, config, board_name, servo_name, udpmanager):
        super().__init__(config, board_name, servo_name, "servos", udpmanager)
    
    def get_actuator_type_name(self):
        return "Servo"
    
    def setup_specific_ui(self):
        # Servo-specific UI elements
        self.angle_label = QLabel("Angle: No data")
        self.angle_label.setFixedWidth(self.LABEL_WIDTH)
        self.layout.addWidget(self.angle_label)
        
        self.manual_angle = QLineEdit(self)
        self.manual_angle.setFixedWidth(self.INPUT_WIDTH)
        self.manual_angle.setPlaceholderText("0")
        
        self.manual_angle_button = QPushButton("Set Angle")
        self.manual_angle_button.setFixedWidth(self.BUTTON_WIDTH)
        self.manual_angle_button.clicked.connect(self.set_manual_angle)
        
        self.layout.addWidget(self.manual_angle)
        self.layout.addWidget(self.manual_angle_button)
    
    def update_specific_states(self, states):
        try:
            angle = states[self.actuator_type][self.actuator_name]["angle"]
            self.angle_label.setText(f"Angle: {angle}")
        except KeyError:
            self.angle_label.setText("Angle: No data")
    
    def set_manual_angle(self):
        try:
            angle = int(self.manual_angle.text())
            if angle < 0 or angle > 180:
                raise ValueError("Angle must be between 0 and 180")
            self.send_command("angle", angle)
        except ValueError as e:
            print(f"Invalid angle: {e}")
