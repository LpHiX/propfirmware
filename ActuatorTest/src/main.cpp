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
#define SERVOMAX 2000  // Maximum pulse length count
#define SERVO_FREQ 50 // Servo frequency (50 Hz)

#define I2C_SDA 21
#define I2C_SCL 20

// Non-blocking LED state handling
unsigned long lastLedChangeTime = 0;
unsigned long lastHeartbeatTime = 0;
unsigned long heartbeatInterval = 5000;  // 5 seconds interval for heartbeat
bool ledState = false;  // false = red, true = green
bool pwmInitialized = false;

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



    Serial.print("Received: ");
    Serial2.println(buffer);
    Serial.println(buffer);

    if (bytesRead > 0) {
      JsonDocument doc; // Size depends on your JSON complexity
  
      // Attempt to parse JSON
      DeserializationError error = deserializeJson(doc, buffer);
      
      if (error) {
        Serial.print("failed");
        Serial.println(error.f_str());
      } else {
        // Successfully parsed JSON
        //Serial.println("parsed");
        
        // Access JSON values
  
        JsonArray servos = doc["servos"];
        
        for (JsonObject servo : servos) {
          int channel = servo["channel"];
          bool armed = servo["armed"];
          int microseconds = servo["pwm"];
          
          if (pwmInitialized) {
            if (armed){
              pwm.writeMicroseconds(channel, microseconds);
            } else {
              pwm.writeMicroseconds(channel, SERVOMID);
            }
          }
        }
  
        JsonArray solenoids = doc["solenoids"];
        
        for (JsonObject solenoid : solenoids) {
          int gpiopin = solenoid["gpio"];
          bool armed = solenoid["armed"];
          bool powered = solenoid["powered"];
  
          if (armed) {
            pinMode(gpiopin, OUTPUT);
            digitalWrite(gpiopin, powered ? HIGH : LOW);
          } else {
            pinMode(gpiopin, INPUT);  // Set to input to disable the pin
          }
        }
      }
    }
    
    // Print received data
    // Serial.print("Received: ");
    // Serial.println(buffer);


    
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