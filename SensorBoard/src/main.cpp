#include <Adafruit_NeoPixel.h>
#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <Mcp320x.h>
#include <Adafruit_NeoPixel.h>

#define RGB_LED_PIN 48
#define NUM_PIXELS 1

// ADC Pins
#define ADC_CS   12
#define ADC_MOSI 11
#define ADC_MISO 10
#define ADC_CLK  9
#define ADC_VREF 5000 // 5.0V for the LT3042 reference!

Adafruit_NeoPixel pixel(NUM_PIXELS, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);
MCP3208 adc(ADC_VREF, ADC_CS);

unsigned long lastAdcTime = 0;
unsigned long lastHeartbeatTime = 0;
bool ledState = false;

void setup() {
  pixel.begin();
  pixel.setPixelColor(0, pixel.Color(255, 0, 0));
  pixel.setBrightness(5);
  pixel.show();
  Serial.begin(115200);

  // Initialize Hardware SPI for the ADC pins
  pinMode(ADC_CS, OUTPUT);
  digitalWrite(ADC_CS, HIGH); // default HIGH state
  SPI.begin(ADC_CLK, ADC_MISO, ADC_MOSI, ADC_CS);
  
  // CRITICAL TXB/TXS0108 FIX: Disable ESP32 internal pull-ups on SPI pins
  // The weak pull-ups confuse the auto-direction sensing logic!
  pinMode(ADC_MISO, INPUT);
  pinMode(ADC_MOSI, OUTPUT);
  pinMode(ADC_CLK, OUTPUT);
}

void loop() {
  unsigned long currentMillis = millis();

  // Print ADC every 0.1s (100ms)
  if (currentMillis - lastAdcTime >= 100) {
    lastAdcTime = currentMillis;
    
    Serial.print("ADC Data: ");
    for (int i = 0; i < 8; i++) {
        // Cast integer loop counter directly to Patrick Rogalla's Channel enum
        uint16_t val = adc.read((MCP3208::Channel)i);
        Serial.print("CH");
        Serial.print(i);
        Serial.print(":");
        Serial.print(val);
        if (i < 7) Serial.print("  ");
    }
    Serial.println();
  }

  // Heartbeat every 1s (1000ms)
  if (currentMillis - lastHeartbeatTime >= 1000) {
    lastHeartbeatTime = currentMillis;
    ledState = !ledState;
    if (ledState) {
        pixel.setPixelColor(0, pixel.Color(0, 255, 0)); // Green
    } else {
        pixel.setPixelColor(0, pixel.Color(255, 0, 0)); // Red
    }
    pixel.show();
    Serial.println("Heartbeat");
  }
}
