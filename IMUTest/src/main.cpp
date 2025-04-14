#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#define BUFFER_SIZE 50
uint8_t rxBuffer[BUFFER_SIZE];

// NeoPixel configuration
#define NEOPIXEL_PIN      48
#define NEOPIXEL_COUNT    1
#define BLINK_INTERVAL    1000  // 1 second between blinks
Adafruit_NeoPixel pixels(NEOPIXEL_COUNT, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);
unsigned long lastrecieveddata = 0;
unsigned long lastBlinkTime = 0;
bool pixelStateGreen = false;
bool pixelStateYellow = false;

// Optimization: Track when we last updated the pixel to avoid too frequent updates
unsigned long lastPixelUpdate = 0;
#define MIN_PIXEL_UPDATE_INTERVAL 50 // Minimum ms between pixel updates

void setup() {
  // Set CPU frequency to maximum for best performance
  setCpuFrequencyMhz(240);

  // Initialize serial for communication
  Serial.begin(921600);
  
  // Increase buffer sizes if possible
  // Serial.setRxBufferSize(2048); // Uncomment if available in your Arduino core
  // Serial.setTxBufferSize(2048); // Uncomment if available in your Arduino core
  
  // Initialize NeoPixel
  pixels.begin();
  pixels.setBrightness(5);  // Keep this low to reduce overhead
  pixels.clear();
  pixels.show();
  
  // Quick double blue flash to indicate boot completed
  pixels.setPixelColor(0, pixels.Color(0, 0, 255));
  pixels.show();
  delay(100);
  pixels.clear();
  pixels.show();
  delay(100);
  pixels.setPixelColor(0, pixels.Color(0, 0, 255));
  pixels.show();
  delay(100);
  pixels.clear();
  pixels.show();
}

void loop() {
  unsigned long currentMillis = millis();
  
  // Highest priority: Handle UART data
  if (Serial.available() >= BUFFER_SIZE) {
    lastrecieveddata = currentMillis;  // Update last received data time
    
    // Optimization: use readBytes() instead of individual reads
    Serial.readBytes(rxBuffer, BUFFER_SIZE);
    
    // Echo back immediately - highest priority
    Serial.write(rxBuffer, BUFFER_SIZE);
    
    // Only update NeoPixel occasionally during high traffic to avoid slowing down UART
    if (!pixelStateGreen && currentMillis - lastPixelUpdate >= MIN_PIXEL_UPDATE_INTERVAL) {
      pixelStateGreen = true;
      pixels.setPixelColor(0, pixels.Color(0, 255, 0));
      pixels.show();
      lastPixelUpdate = currentMillis;
    }
    
    // Return immediately to handle more data if available
    return;
  }

  // Lower priority: Handle NeoPixel updates when no data is being processed
  if (currentMillis - lastPixelUpdate >= MIN_PIXEL_UPDATE_INTERVAL) {
    // Check if it's time to turn off green pixel
    if (pixelStateGreen && currentMillis - lastrecieveddata >= 10) {
      pixelStateGreen = false;
      pixels.clear();
      pixels.show();
      lastPixelUpdate = currentMillis;
    }
    
    // Yellow blinking when idle
    if (!pixelStateGreen && 
        currentMillis - lastrecieveddata >= BLINK_INTERVAL && 
        currentMillis - lastBlinkTime >= BLINK_INTERVAL) {
      lastBlinkTime = currentMillis;
      pixelStateYellow = !pixelStateYellow;
      
      if (pixelStateYellow) {
        pixels.setPixelColor(0, pixels.Color(255, 255, 0));
      } else {
        pixels.clear();
      }
      pixels.show();
      lastPixelUpdate = currentMillis;
    }
  }
}