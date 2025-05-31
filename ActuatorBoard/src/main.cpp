#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <ArduinoJson.h>  // Include ArduinoJson library
#include <Adafruit_NeoPixel.h>
#include "Adafruit_PWMServoDriver.h"


// Define the RGB LED pin
#define PCA9685_I2C_ADDRESS 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA9685_I2C_ADDRESS);
#define RGB_LED_PIN 48
#define NUM_PIXELS 1  // Just the one integrated LED

#define SERVOMIN 1000  // Minimum pulse length count
#define SERVOMID 1550  // Mid pulse length count
#define SERVOMAX 2000  // Maximum pulse xlength count
#define SERVO_FREQ 50 // Servo frequency (50 Hz)

#define I2C_SDA 5
#define I2C_SCL 6

// Non-blocking LED state handling
unsigned long lastLedChangeTime = 0;
unsigned long lastHeartbeatTime = 0;
unsigned long heartbeatInterval = 5000;  // 5 seconds interval for heartbeat
bool ledState = false;  // false = red, true = green
bool pwmInitialized = false;
float servos_angles[16] = {0};
bool servos_armed[16] = {false};
bool solenoids_armed[8] = {false};
bool solenoids_powered[8] = {false};
int solenoids_GPIOS[8] = {1, 2, 42, 41, 40, 39, 38, 13}; // 42, 41, 39 IS BROKEN ?!?!?!?!
bool pyros_armed[2] = {false};
bool pyros_powered[2] = {false};
int pyros_GPIOS[2] = {47, 21};//fstudent

// Initialize NeoPixel object
Adafruit_NeoPixel pixel(NUM_PIXELS, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
    // Initialize the NeoPixel
  pixel.begin();
  pixel.setPixelColor(0, pixel.Color(255, 255, 0));
  pixel.setBrightness(5);
  pixel.show();
  // Initialize USB Serial for computer communication
  Serial.begin(921600);

  Wire.begin(I2C_SDA, I2C_SCL); // Initialize I2C with custom SDA and SCL pins
  Serial.println("Initializing PCA9685...");
  Wire.beginTransmission(PCA9685_I2C_ADDRESS);
  byte error = Wire.endTransmission();
  
  if (error == 0) {
    // I2C device found at address 0x40
    Serial.println("PCA9685 found at address 0x40");
    
    // Initialize PWM
    if (pwm.begin()) {
      pwm.setPWMFreq(SERVO_FREQ);  // Set PWM frequency to 50 Hz
      pwmInitialized = true;
      Serial.println("PCA9685 initialized successfully");
    } else {
      Serial.println("Failed to initialize PCA9685");
    }
  } else {
    Serial.print("I2C error: Device not found at address 0x40, error code: ");
    Serial.println(error);
  }


  Serial2.setRxBufferSize(4096);
  Serial2.begin(921600, SERIAL_8N1, 18, 17); // RX=18, TX=17 for Raspberry Pi communication
  while (Serial2.available()) { Serial2.read(); }
}

void loop() {
  unsigned long currentTime = millis();
  
  // Handle UART data - no delays in this section
  if (Serial2.available()) {
    // Create buffer with room for incoming data
    const int bufferSize = 4096;
    char buffer[bufferSize];
    int bytesRead = 0;
    
    // Read all available bytes at once
    while (Serial2.available() && bytesRead < bufferSize-1) {
      buffer[bytesRead++] = Serial2.read();
    }
    
    // Null-terminate the string
    buffer[bytesRead] = '\0';



    Serial.println("Received: ");
    Serial.println(buffer);

    if (bytesRead > 0) {
      JsonDocument doc; // Size depends on your JSON complexity
      DeserializationError error = deserializeJson(doc, buffer);

      JsonDocument responseDoc;
      responseDoc["servos"] = JsonObject();
      responseDoc["solenoids"] = JsonObject();
      responseDoc["pyros"] = JsonObject();
      if (error) {
        Serial.print("failed");
        Serial.println(error.f_str());
      } else {
        // If doc has "servos", process it:
        JsonObject servos = doc["servos"].as<JsonObject>();
        for (JsonPair servo : servos) {
          const char* servoName = servo.key().c_str();
          JsonObject servoData = servo.value().as<JsonObject>();
          int channel = servoData["channel"];
          if (channel < 0 || channel > 15) {
            continue; // Skip this iteration if the channel is invalid
          }
          if (pwmInitialized) {
            if (servoData.containsKey("armed")) {
              bool armed_desired = servoData["armed"];
              servos_armed[channel] = armed_desired;
            }

            if (servos_armed[channel]) {
              if (servoData.containsKey("angle")) {
                float angle_desired = servoData["angle"];
                int microseconds = map(angle_desired, 0, 180, SERVOMIN, SERVOMAX);
                pwm.writeMicroseconds(channel, microseconds);
                servos_angles[channel] = angle_desired;
              }
            }
          }
          responseDoc["servos"][servoName]["armed"] = servos_armed[channel];
          responseDoc["servos"][servoName]["angle"] = servos_angles[channel];
        }
  
        JsonObject solenoids = doc["solenoids"].as<JsonObject>();
        for (JsonPair solenoid : solenoids) {
          const char* solenoidName = solenoid.key().c_str();
          JsonObject solenoidData = solenoid.value().as<JsonObject>();
          int channel = solenoidData["channel"];

          if (channel < 0 || channel > 7) {
            continue;
          }
          if (solenoidData.containsKey("armed")){
            bool armed_desired = solenoidData["armed"];
            solenoids_armed[channel] = armed_desired;
          }
          if (solenoids_armed[channel]) {
            if (solenoidData.containsKey("powered")){
              bool powered_desired = solenoidData["powered"];
              pinMode(solenoids_GPIOS[channel], OUTPUT);
              digitalWrite(solenoids_GPIOS[channel], powered_desired ? HIGH : LOW);
              solenoids_powered[channel] = powered_desired;
            } 
          } else {
            pinMode(solenoids_GPIOS[channel], INPUT); 
            solenoids_powered[channel] = false;
          }

          responseDoc["solenoids"][solenoidName]["armed"] = solenoids_armed[channel];
          responseDoc["solenoids"][solenoidName]["powered"] = solenoids_powered[channel];
        }

        JsonObject pyros = doc["pyros"].as<JsonObject>();
        for (JsonPair pyro : pyros) {
          const char* pyroName = pyro.key().c_str();
          JsonObject pyroData = pyro.value().as<JsonObject>();
          int channel = pyroData["channel"];

          if (channel < 0 || channel > 2) {
            continue;
          }
          pinMode(pyros_GPIOS[channel], OUTPUT);

          if (pyroData.containsKey("armed")){
            bool armed_desired = pyroData["armed"];
            pyros_armed[channel] = armed_desired;
          }
          if (pyros_armed[channel]) {
            if (pyroData.containsKey("powered")){
              bool powered_desired = pyroData["powered"];
              // pinMode(pyros_GPIOS[channel], OUTPUT);
              digitalWrite(pyros_GPIOS[channel], powered_desired ? HIGH : LOW);
              pyros_powered[channel] = powered_desired;
            } 
          } else {
            // pinMode(pyros_GPIOS[channel], INPUT); 
            pyros_powered[channel] = false;
            digitalWrite(pyros_GPIOS[channel], LOW);
          }

          responseDoc["pyros"][pyroName]["armed"] = pyros_armed[channel];
          responseDoc["pyros"][pyroName]["powered"] = pyros_powered[channel];
        }

        responseDoc["send_id"] = doc["send_id"];
        serializeJson(responseDoc, Serial2);
        Serial2.println(); // Send a newline after the JSON response
        //Serial.println("Sending:");
        //serializeJson(responseDoc, Serial);
      }
    }

    // Set the LED green and remember when we changed it
    pixel.setPixelColor(0, pixel.Color(0, 255, 0));
    ledState = true;
    lastLedChangeTime = currentTime;
  }
  if (currentTime - lastHeartbeatTime > heartbeatInterval) {
    // Send heartbeat message
    Serial.println("heartbeat");
    lastHeartbeatTime = currentTime;
  }
  // Handle LED state in a non-blocking way
  if (ledState && currentTime - lastLedChangeTime > 100) {
    // Change back to red after 500ms
    pixel.setPixelColor(0, pixel.Color(255, 0, 0));
    ledState = false;
  }
  
  
  // Update LED - happens every loop iteration
  pixel.show();
}