#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <ArduinoJson.h>  // Include ArduinoJson library
#include <Adafruit_NeoPixel.h>
#include <Adafruit_ADS1X15.h>

Adafruit_ADS1115 ads;

// Define the RGB LED pin
#define RGB_LED_PIN 48
#define NUM_PIXELS 1  // Just the one integrated LED

#define SDA 2
#define SCL 1

int i2c_speed = 3400000;

const uint16_t adc_channels[] = {(0x4000), (0x5000), (0x6000), (0x7000)};

// Non-blocking LED state handling
unsigned long lastLedChangeTime = 0;
unsigned long lastHeartbeatTime = 0;
unsigned long heartbeatInterval = 5000;  // 5 seconds interval for heartbeat
bool ledState = false;  // false = red, true = green
float adc_mvs[16] = {0};
// Initialize NeoPixel object
Adafruit_NeoPixel pixel(NUM_PIXELS, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);

float readChannel(int channel) {
  if (channel < 0 || channel > 3) {
    return 0; // Invalid channel
  }
  
  // Configure the mux for this channel
  ads.startADCReading(adc_channels[channel], /*continuous=*/false);
  
  // Allow sufficient time for the ADC to switch channels and complete a conversion
  delayMicroseconds(1200); // 1.2ms should be sufficient (slightly longer than the minimum required)
  
  // Read the conversion results
  int16_t adc = ads.getLastConversionResults();
  return ads.computeVolts(adc) * 1000; // Convert to millivolts
}

void setup() {
    // Initialize the NeoPixel
  pixel.begin();
  pixel.setPixelColor(0, pixel.Color(255, 255, 0));
  pixel.setBrightness(5);
  pixel.show();

  Wire.begin(SDA, SCL);
  Wire.setClock(i2c_speed);
  
  if (!ads.begin(0x48, &Wire)) {
    Serial.println("Failed to initialize ADS1115!");
  }

  ads.setDataRate(RATE_ADS1115_860SPS); // Set to maximum sample rate
  ads.setGain(GAIN_TWOTHIRDS);          // +/-6.144V range
  // Initialize USB Serial for computer communication
  Serial.begin(921600);

  Serial2.setRxBufferSize(4096);
  Serial2.begin(921600, SERIAL_8N1, 18, 17); // RX=18, TX=17 for Raspberry Pi communication
  while (Serial2.available()) { Serial2.read(); } // Clear the buffer
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
    Serial.println(buffer);

    if (bytesRead > 0) {
      JsonDocument doc; // Size depends on your JSON complexity
      DeserializationError error = deserializeJson(doc, buffer);

      JsonDocument responseDoc;
      
      if (error) {
        Serial.print("failed");
        Serial.println(error.f_str());
      } else {
        //If doc has "pts", process it:
        JsonObject pts_response = responseDoc["pts"].to<JsonObject>();
        JsonObject pts = doc["pts"].as<JsonObject>();
        for (JsonPair pt : pts) {
          const char* ptName = pt.key().c_str();
          JsonObject ptData = pt.value().as<JsonObject>();
          int channel = ptData["channel"];
          responseDoc["pts"][ptName]["mv"] = adc_mvs[channel];
        }

        responseDoc["send_id"] = doc["send_id"];

        serializeJson(responseDoc, Serial2);
        Serial2.println(); // Send a newline after the JSON response
        serializeJson(responseDoc, Serial);
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


  adc_mvs[0] = readChannel(0);


  if (currentTime - lastHeartbeatTime > heartbeatInterval) {
    Serial.println("heartbeat");
    lastHeartbeatTime = currentTime;
  }
  if (ledState && currentTime - lastLedChangeTime > 100) {
    // Change back to red after 500ms
    pixel.setPixelColor(0, pixel.Color(255, 0, 0));
    ledState = false;
  }
  
  
  // Update LED - happens every loop iteration
  pixel.show();
}