#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <ArduinoJson.h>  // Include ArduinoJson library
#include <Adafruit_NeoPixel.h>

// Define the RGB LED pin
#define RGB_LED_PIN 48
#define NUM_PIXELS 1  // Just the one integrated LED

// Initialize NeoPixel object
Adafruit_NeoPixel pixel(NUM_PIXELS, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);

// uint32_t Wheel(byte WheelPos) {
//   WheelPos = 255 - WheelPos;
//   if(WheelPos < 85) {
//     return pixel.Color(255 - WheelPos * 3, 0, WheelPos * 3);
//   }
//   if(WheelPos < 170) {
//     WheelPos -= 85;
//     return pixel.Color(0, WheelPos * 3, 255 - WheelPos * 3);
//   }
//   WheelPos -= 170;
//   return pixel.Color(WheelPos * 3, 255 - WheelPos * 3, 0);
// }

void setup() {
  // Initialize the NeoPixel
  pixel.begin();
  pixel.setPixelColor(0, pixel.Color(255, 0, 0));
  pixel.setBrightness(5);
  
  // Initialize USB Serial for computer communication
  Serial.begin(115200);
  Serial2.setRxBufferSize(4096);
  Serial2.begin(921600, SERIAL_8N1, 18, 17); // RX=18, TX=17 for Raspberry Pi communication
  while (Serial2.available()) { Serial2.read(); }
}

// Non-blocking LED state handling
unsigned long lastLedChangeTime = 0;
unsigned long lastHeartbeatTime = 0;
unsigned long heartbeatInterval = 5000;  // 5 seconds interval for heartbeat
bool ledState = false;  // false = red, true = green

void loop() {
  unsigned long currentTime = millis();
  
  // Handle UART data - no delays in this section
  if (Serial2.available()) {
    // Create buffer with room for incoming data
    const int bufferSize = 512;
    char buffer[bufferSize];
    int bytesRead = 0;
    
    // Read all available bytes at once
    while (Serial2.available() && bytesRead < bufferSize-1) {
      buffer[bytesRead++] = Serial2.read();
    }
    
    // Null-terminate the string
    buffer[bytesRead] = 0;
    
    // Print received data
    Serial.print("Received: ");
    Serial.println(buffer);
    
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
  if (ledState && currentTime - lastLedChangeTime > 500) {
    // Change back to red after 500ms
    pixel.setPixelColor(0, pixel.Color(255, 0, 0));
    ledState = false;
  }
  
  
  // Update LED - happens every loop iteration
  pixel.show();
}