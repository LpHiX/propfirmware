

class PropertyTestApp(QMainWindow):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.hardware_json = None
        self.state_defaults: dict = {}
        self.hardware_types: list[str] = []
        self.actuator_list = []
        self.sensor_list = []


        self.setWindowTitle("Property Test App")
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)  

        self.main_tab = QWidget()
        self.tab_widget.addTab(self.main_tab, "Main")
        self.main_layout = QHBoxLayout(self.main_tab)
        
        self.pressure_tab = QWidget()
        self.tab_widget.addTab(self.pressure_tab, "Pressure")
        self.pressure_layout = QVBoxLayout(self.pressure_tab)


        #container = QWidget()
        #container.setLayout(self.main_layout)


        self.data_waiting_label = QLabel("Waiting for data...")

        self.last_update_time = None
        self.last_state_update_label = QLabel("Last state update: Never")
        self.last_state_update_label.setFixedWidth(200)
        self.backend_state_timer = QTimer(self)
        self.backend_state_timer.timeout.connect(self.backend_state_coroutine)
        self.backend_state_timer.start(10)
        self.manual_command = QTextEdit(self)
        self.manual_response = QTextEdit(self)
        self.manual_response.setReadOnly(True)
        self.manual_command_button = QPushButton("Send Command")
        self.manual_command_button.clicked.connect(self.send_manual_command)
        self.manual_command_layout = QVBoxLayout()
        self.manual_command_layout.addWidget(self.last_state_update_label)
        self.manual_command_layout.addWidget(self.manual_command)
        self.manual_command_layout.addWidget(self.manual_command_button)
        self.manual_command_layout.addWidget(self.manual_response)

        self.preset_commands_layout = QVBoxLayout()
        
        self.datetime_str = "Request timer didn't load"
        self.datetime_label = QLabel(f"Backend Time: {self.datetime_str}")
        self.preset_commands_layout.addWidget(self.datetime_label)

        self.hotfiretime_str = "Request timer didn't load"
        self.hotfiretime_label = QLabel(self.hotfiretime_str)
        self.hotfiretime_label.setStyleSheet("font-size: 60px;")
        self.preset_commands_layout.addWidget(self.hotfiretime_label)

        self.statemachine_str = "Request timer didn't load"
        self.statemachine_label = QLabel("Machine state: {self.statemachine_str}")
        self.preset_commands_layout.addWidget(self.statemachine_label)

        self.abort_button = QPushButton("Abort")
        self.abort_button.setStyleSheet("font-size: 60px;")
        self.abort_button.setFixedHeight(300)
        self.abort_button.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"abort engine", "data":{}})))
        self.preset_commands_layout.addWidget(self.abort_button)

        self.hotfire_button = QPushButton("Hotfire")
        self.hotfire_button.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"start hotfire sequence", "data":{}})))
        self.preset_commands_layout.addWidget(self.hotfire_button)

        self.return_from_idle_button = QPushButton("Return to Idle")
        self.return_from_idle_button.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"return to idle", "data":{}})))
        self.preset_commands_layout.addWidget(self.return_from_idle_button)

        self.reload_hardware_json = QPushButton("Reload Hardware JSON")
        self.reload_hardware_json.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"reload hardware json", "data":{}})))
        self.preset_commands_layout.addWidget(self.reload_hardware_json)

        self.get_new_hardware_json = QPushButton("Get New Hardware JSON (DOESNT WORK)")
        self.get_new_hardware_json.clicked.connect(lambda: self.udp_manager.send(json.dumps({"command":"get hardware json", "data":{}})))
        self.preset_commands_layout.addWidget(self.get_new_hardware_json)

        self.preset_commands_layout.addStretch()

        self.commands_layout = QHBoxLayout()
        self.commands_layout.addLayout(self.manual_command_layout)
        self.commands_layout.addLayout(self.preset_commands_layout)

        self.control_area = QVBoxLayout()
        self.sensor_area = QVBoxLayout()

        self.actuators_sensors_area = QVBoxLayout()
        self.actuators_sensors_area.addLayout(self.control_area)
        self.actuators_sensors_area.addLayout(self.sensor_area)


        self.main_layout.addWidget(self.data_waiting_label)
        self.main_layout.addLayout(self.actuators_sensors_area)
        self.main_layout.addLayout(self.commands_layout)

        


        self.pressureplot = pg.PlotWidget(title="Pressure")
        self.pressureplot.addLegend()
        self.pressure_layout.addWidget(self.pressureplot)


        self.udp_manager = UDPManager(self.host, self.port)
        self.udp_manager.dataReceived.connect(self.handle_data_received)

        self.commands_responses = {
            "get hardware json": self.hardware_json_received,
            "get boards states": self.boards_states_received,
            "get state": self.machinestate_received,
            "get time": self.time_received
        }
    def time_received(self, responsestr):
        try:
            response = json.loads(responsestr)["response"]
        except json.JSONDecodeError:
            print("Time is not valid JSON")
            return
        if "date_time" in response:
            self.datetime_str = response["date_time"]
            self.datetime_label.setText(f"Backend Time: {self.datetime_str}")
        else:
            print("No date_time in response")
        if "hotfire_time" in response:
            self.hotfiretime_str = response["hotfire_time"]
            self.hotfiretime_label.setText(f"{self.hotfiretime_str}")
        else:
            print("No hotfire_time in response")

    def machinestate_received(self, response):
        try:
            self.statemachine_str = json.loads(response)["response"]
            self.statemachine_label.setText(f"Machine state: {self.statemachine_str}")
        except json.JSONDecodeError:
            print("Machine state is not valid JSON")
            return
    def handle_data_received(self, data):
        try:
            data = json.loads(data)
            #print(f"Decoded data: {data}")
            if "command" not in data:
                print("No command in data json, backend is being weird")
                return
            if "response" not in data:
                print("No response in data json, backend is being weird")
                return
            
            command = data["command"]
            response = data["response"]
            #print(type(response))
            if command in self.commands_responses:
                self.commands_responses[command](response)
            else:
                self.manual_response.append(json.dumps(data, indent=4))
        except json.JSONDecodeError:
            print("Failed to decode JSON data")
    def hardware_json_received(self, response):
        self.udp_manager.set_hardware_json_received(True)
        self.main_layout.removeWidget(self.data_waiting_label)
        self.data_waiting_label.setParent(None)
        self.data_waiting_label.deleteLater()
        try :
            response = json.loads(response)
        except json.JSONDecodeError:
            print("Hardware json is not valid JSON")
            return
        
        self.hardware_json = response
        if "boards" not in self.hardware_json:
            print("No boards in hardware json")
            return
        
        if "state_defaults" not in self.hardware_json:
            print("No state defaults in hardware json")
            return
        
        self.state_defaults: dict = self.hardware_json['state_defaults']
        self.hardware_types = list(self.state_defaults.keys())
        
        actuator_group = QGroupBox("Actuators")
        actuator_layout = QVBoxLayout()
        self.actuator_list.clear()

        default_actuator_types = {
            "servos": ServoControllerWidget,
            "solenoids": SolenoidControllerWidget,
            "pyros": PyroControllerWidget
        }

        for board_name, board_config in self.hardware_json["boards"].items():
            if board_config.get("is_actuator", False):
                for actuator_type, actuator_widget_class in default_actuator_types.items():
                    if actuator_type in board_config:
                        for actuator_name, _ in board_config[actuator_type].items():
                            actuator_controller_widget = actuator_widget_class(board_config, board_name, actuator_name, self.udp_manager)
                            actuator_layout.addWidget(actuator_controller_widget)
                            self.actuator_list.append(actuator_controller_widget)
        actuator_layout.addStretch()
        actuator_group.setLayout(actuator_layout)
        self.control_area.addWidget(actuator_group)
        
        sensor_group = QGroupBox("Sensors")
        sensor_mainlayout = QVBoxLayout()
        sensors_layout = []
        self.sensor_list.clear()
        for board_name, board_config in self.hardware_json["boards"].items():
            if not board_config.get("is_actuator", False):
                for sensor_type, sensor_default_data in self.state_defaults.items():
                    if sensor_type in board_config and isinstance(board_config[sensor_type], dict):
                        sensor_layout = QHBoxLayout()
                        sensor_mainlayout.addLayout(sensor_layout)
                        sensors_layout.append(sensor_layout)
                        for sensor_name, _ in board_config[sensor_type].items():
                            sensor_controller_widget = SensorControllerWidget(board_config, sensor_default_data, board_name, sensor_name, sensor_type, self.udp_manager, self)
                            sensor_layout.addWidget(sensor_controller_widget)
                            self.sensor_list.append(sensor_controller_widget)
        sensor_mainlayout.addStretch()
        sensor_group.setLayout(sensor_mainlayout)
        self.sensor_area.addWidget(sensor_group)

    def boards_states_received(self, response):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            print("Boards states is not valid JSON")
            return
        
        for board_name, states in response.items():
            if board_name in self.hardware_json["boards"]:
                for actuator in self.actuator_list:
                    if actuator.board_name == board_name:
                        actuator.update_states(states)
                for sensor in self.sensor_list:
                    if sensor.board_name == board_name:
                        sensor.update_states(states)
            else:
                print(f"Board {board_name} not found in hardware json")
        self.last_update_time = QDateTime.currentDateTime()
    def send_manual_command(self):
        command = self.manual_command.toPlainText()
        if not command:
            return
        #self.manual_response.clear()
        try:
            command_json = json.loads(command)
            if "command" not in command_json:
                self.manual_response.append("No command in JSON")
                return
            if "data" not in command_json:
                self.manual_response.append("No data in JSON")
                return
            self.udp_manager.send(command)
        except json.JSONDecodeError:
            self.manual_response.append("Invalid JSON format")
    def backend_state_coroutine(self):
        self.udp_manager.send(json.dumps({"command":"get state", "data":{}}))
        self.udp_manager.send(json.dumps({"command":"get time", "data":{}}))
        if self.last_update_time is not None:
            current_time = QDateTime.currentDateTime()
            elapsed_time = self.last_update_time.msecsTo(current_time)
            self.last_state_update_label.setText(f"Last state update: {elapsed_time/1000:.1f} seconds ago")
            if elapsed_time > 5000:
                self.last_state_update_label.setStyleSheet("color: red")
            else:
                self.last_state_update_label.setStyleSheet("color: white")
        else:
            self.last_state_update_label.setStyleSheet("color: red")