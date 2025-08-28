#include <Arduino.h>
#include <ArduinoJson.h>
#include <Adafruit_NeoPixel.h>

// Define the RGB LED pin
#define RGB_LED_PIN 48
#define NUM_PIXELS 1 // Just the one integrated LED

#define TACHO_PIN 4
#define BLADES 0.5

volatile int64_t lastPrintTime = 0;
volatile int64_t lastHeartbeatTime = 0;
unsigned long lastLedChangeTime = 0;
bool ledState = false; // false = red, true = green

volatile int64_t lastPulseTime = 0;
volatile int64_t pulseInterval = 0;
volatile int64_t twoIntervals = 0;
volatile int pulseCount = 0;
volatile float rpm = 0;
volatile bool newPulse = false;

const int N = 9;
int64_t intervals[N];
int idx = 0;
bool bufferFilled = false;

int64_t medianOfArray(int64_t *a, int n)
{
  // simple insertion-sort copy then pick middle (n small)
  int64_t b[N];
  for (int i = 0; i < n; i++)
    b[i] = a[i];
  for (int i = 1; i < n; i++)
  {
    int64_t v = b[i];
    int j = i - 1;
    while (j >= 0 && b[j] > v)
    {
      b[j + 1] = b[j];
      j--;
    }
    b[j + 1] = v;
  }
  return b[n / 2];
}

void IRAM_ATTR onPulse()
{
  int64_t now = esp_timer_get_time(); // µs since boot
  int64_t dt = now - lastPulseTime;
  if (dt > 2000)
  {
    pulseInterval += dt;
    pulseCount++;

    if (pulseCount == 2)
    {
      twoIntervals = pulseInterval;
      pulseInterval = 0;
      pulseCount = 0;
      newPulse = true;
      lastPulseTime = now; // µs

    }
  }

}

Adafruit_NeoPixel pixel(NUM_PIXELS, RGB_LED_PIN, NEO_GRB + NEO_KHZ800);

void setup()
{
  pixel.begin();
  pixel.setPixelColor(0, pixel.Color(255, 255, 0));
  pixel.setBrightness(5);
  pixel.show();
  Serial.begin(921600);

  Serial2.setRxBufferSize(4096);
  Serial2.begin(921600, SERIAL_8N1, 14, 13); // RX=14, TX=13 for Raspberry Pi communication
  while (Serial2.available())
  {
    Serial2.read();
  } // Clear the buffer

  pinMode(TACHO_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(TACHO_PIN), onPulse, FALLING);
}

void processPulseData()
{
  if (newPulse)
  {
    noInterrupts();
    int64_t interval = twoIntervals; // µs
    newPulse = false;
    interrupts();

    int count = bufferFilled ? N : idx;

    if (interval > 0)
    {

      intervals[idx] = interval;
      idx = (idx + 1) % N;
      if (idx == 0)
        bufferFilled = true;
      // compute median only when buffer has some data
      int count = bufferFilled ? N : idx;
      if (count >= 3)
      { // wait at least 3 samples before median
        int64_t med = medianOfArray(intervals, count);

        // Optionally: reject very different sample vs median (extra guard)
        if (twoIntervals < med * 7 / 10)
        {
          // sample suspiciously small -> ignore (double-count). Keep med instead.
          twoIntervals = med;
        }
        else if (twoIntervals > med * 2)
        {
          // sample suspiciously large -> maybe missed pulse; use med (or accept depending on use).
          twoIntervals = med;
        }

        // compute RPM from median (use med rather than the raw noisy sample)
        if (med > 0)
          rpm = 60000000.0 / (med * BLADES);

        Serial.print("median interval(us): ");
        Serial.print(med);
        Serial.print("  RPM: ");
        Serial.println(rpm);
      }
      else
      {
        // not enough samples yet — show raw estimate
        if (twoIntervals > 0)
          rpm = 60000000.0 / (twoIntervals * BLADES);
        Serial.print("raw interval(us): ");
        Serial.print(twoIntervals);
        Serial.print("  RPM: ");
        Serial.println(rpm);
      }
    }
  }
}

void loop()
{
  processPulseData();
  int64_t currentTime = esp_timer_get_time();

  // timeout: e.g. 2x the period corresponding to 60 RPM
  // 60 RPM = 1 rev/s, 9 blades -> 9 pulses/s, ~111 ms per pulse
  // Two intervals = ~222 ms. Let's allow 500 ms margin.
  if (currentTime - lastPulseTime > 500000) { // 500,000 µs = 0.5s
    rpm = 0;
  }

  if (Serial2.available())
  {
    // Create buffer with room for incoming data
    const int bufferSize = 4096;
    char buffer[bufferSize];
    int bytesRead = 0;

    // Read all available bytes at once
    while (Serial2.available() && bytesRead < bufferSize - 1)
    {
      buffer[bytesRead++] = Serial2.read();
    }

    // Null-terminate the string
    buffer[bytesRead] = '\0';

    // Serial.print("Received: ");
    // Serial.println(buffer);

    if (bytesRead > 0)
    {
      // Create a JsonDocument to hold the incoming JSON data
      JsonDocument requestDoc; // Size depends on your JSON complexity

      // convert the received raw JSON string into a JsonDocument object
      DeserializationError error = deserializeJson(requestDoc, buffer);

      // initialize a response JsonDocument
      JsonDocument responseDoc;

      if (error)
      { // verify successful deserialization of incoming JSON string
        Serial.print("JSON error: ");
        Serial.println(error.f_str());
      }
      else
      {

        // FLOWMETER DATA SENDING =================================================================================

        responseDoc["tachos"].to<JsonObject>();                 // create dictionary "tachos" in responseDoc
        JsonObject tachos = requestDoc["tachos"].as<JsonObject>(); // cast dictionary "tachos" from requestDoc to JsonObject fms

        // loop through each key-value pair in dictionary "fms" in requestDoc
        // fm is temporary JsonPair object (e.g. {"fm1": {"channel": 4}})
        for (JsonPair tacho : tachos)
        {
          const char *tachoName = tacho.key().c_str(); // get name of sensor (key) as a C-string (e.g. fm1)
          // JsonObject fmChannel = fm.value().as<JsonObject>();     // get sensor channel on ESP (value (key-value pair)) as a JsonObject (e.g. {"channel": 4})
          responseDoc["tachos"][tachoName]["rpm"] = rpm; // store rpm in responseDoc (e.g. {"fm1": {"rpm": 1234}})
        }
        // ========================================================================================================

        responseDoc["send_id"] = requestDoc["send_id"]; // copy send_id from requestDoc to responseDoc

        serializeJson(responseDoc, Serial2);
        Serial2.println(); // Send a newline after the JSON response
        // serializeJson(responseDoc, Serial);
        // Serial.println(); // Send a newline after the JSON response to Serial for debugging
      }
    }
    pixel.setPixelColor(0, pixel.Color(0, 255, 0));
    ledState = true;
    lastLedChangeTime = currentTime;
  }

  // Update LED - happens every loop iteration
  if (ledState && currentTime - lastLedChangeTime > 100000)
  { // 1 second
    pixel.setPixelColor(0, pixel.Color(255, 0, 0));
    ledState = false;
  }

  if (currentTime - lastHeartbeatTime > 5000000)
  {
    Serial.println("Heartbeat");
    lastHeartbeatTime = currentTime;
  }
  pixel.show();
}