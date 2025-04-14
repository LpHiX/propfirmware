#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_BMP085_U.h>
#include <QMC5883LCompass.h>
#include <TinyGPS++.h>
#include <Adafruit_NeoPixel.h>

// Pin Definitions
#define I2C_SDA_PIN 7
#define I2C_SCL_PIN 16
#define GPS_TX_PIN 17
#define GPS_RX_PIN 18
#define LED_PIN 48

// LED Status Colors
#define COLOR_ERROR     0xFF0000  // Red
#define COLOR_WAITING   0xFFFF00  // Yellow
#define COLOR_SUCCESS   0x00FF00  // Green
#define COLOR_INIT      0x0000FF  // Blue
#define COLOR_GPS_DATA  0xFF00FF  // Purple - receiving NMEA but no fix
#define COLOR_HEARTBEAT 0x00FFFF  // Cyan - "heartbeat" pulse

// Sensor Instances
Adafruit_MPU6050 mpu;
Adafruit_BMP085_Unified bmp = Adafruit_BMP085_Unified(10085);
QMC5883LCompass compass;
TinyGPSPlus gps;
Adafruit_NeoPixel pixel(1, LED_PIN, NEO_GRB + NEO_KHZ800);

// UART for GPS
HardwareSerial gpsSerial(1);  // Use UART1

// Global status variables
bool mpuStatus = false;
bool bmpStatus = false;
bool compassStatus = false;
bool gpsStatus = false;
bool gpsDataReceived = false;
unsigned long lastGpsCharTime = 0;
unsigned long lastBlinkTime = 0;
unsigned long lastDiagnosticsTime = 0;
int maxSatellitesEverSeen = 0;
bool ledState = false;

// String to hold raw NMEA data for debugging
String nmeaBuffer = "";
bool nmeaLineComplete = false;

// Function prototypes
void updateLedStatus();
void readGY87Data();
void readGPSData();
void printSensorData();
void configureGPSForRocket();
void blinkStatusLED();
void addGPSDiagnostics();
void scanI2CDevices(); // New function prototype
bool tryAlternativeCompassApproach(); // New function prototype
void printRawSensorData(); // New function prototype

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) delay(10);  // Wait for serial console to open
  // delay(2000);
  Serial.println("ESP32 GY87 and GPS Integration for Rocket Control System");
  Serial.println("Current time: 2025-04-11 02:01:41 UTC");
  Serial.println("User: LpHiX");
  
  // Initialize NeoPixel
  pixel.begin();
  pixel.setPixelColor(0, COLOR_INIT);  // Blue during initialization
  pixel.setBrightness(5);  // Set brightness to 50%
  pixel.show();
  
  // Initialize I2C
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  
  // Scan for I2C devices to help with debugging
  Serial.println("Scanning for I2C devices...");
  scanI2CDevices();
  
  // Initialize MPU6050
  Serial.println("Initializing MPU6050...");
  if (mpu.begin()) {
    mpuStatus = true;
    Serial.println("MPU6050 found!");
    
    // Setup MPU6050
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    
    // Enable bypass mode to access magnetometer directly
    Serial.println("Enabling I2C bypass mode for magnetometer access...");
    Wire.beginTransmission(0x68); // MPU6050 address
    Wire.write(0x6A); // USER_CTRL register
    Wire.write(0x00); // Disable master mode
    Wire.endTransmission();
    delay(50);
    
    Wire.beginTransmission(0x68); // MPU6050 address
    Wire.write(0x37); // INT_PIN_CFG register
    Wire.write(0x02); // Enable bypass mode
    Wire.endTransmission();
    delay(100);
    Serial.println("I2C bypass mode enabled");
  } else {
    Serial.println("Failed to find MPU6050 chip");
  }
  
  // Initialize BMP085
  Serial.println("Initializing BMP085...");
  if (bmp.begin()) {
    bmpStatus = true;
    Serial.println("BMP085 found!");
    
    sensor_t sensor;
    bmp.getSensor(&sensor);
    Serial.print("Sensor: "); Serial.println(sensor.name);
  } else {
    Serial.println("Failed to find BMP085 sensor");
  }
  
  // Initialize QMC5883L Compass
  Serial.println("Initializing QMC5883L Compass...");
  
  // Try multiple possible addresses for QMC5883L
  byte compassAddresses[] = {0x0D, 0x1E}; // Common addresses for QMC5883L and HMC5883L
  bool compassFound = false;
  byte foundAddress = 0;
  
  for (byte addr : compassAddresses) {
    Wire.beginTransmission(addr);
    int error = Wire.endTransmission();
    
    if (error == 0) {
      Serial.print("Potential compass found at address 0x");
      Serial.println(addr, HEX);
      foundAddress = addr;
      compassFound = true;
      break;
    }
  }
  
  if (!compassFound) {
    Serial.println("Failed to find QMC5883L Compass at any expected address.");
    Serial.println("Please check your wiring and power connections.");
    compassStatus = false;
    return;
  }
  
  // Manual reset of the QMC5883L
  Wire.beginTransmission(foundAddress);
  Wire.write(0x0B);  // Mode register
  Wire.write(0x01);  // Soft reset
  int resetResult = Wire.endTransmission();
  
  if (resetResult != 0) {
    Serial.println("Failed to reset compass");
    compassStatus = false;
    return;
  }
  
  delay(50);  // Wait for reset to complete
  
  // Manual initialization of QMC5883L
  Wire.beginTransmission(foundAddress);
  Wire.write(0x09);  // Register 9 (control register 1)
  Wire.write(0x1D);  // Set continuous mode (0x01), 8 samples/sec (0x00), and +/-2G range (0x00)
  int writeResult1 = Wire.endTransmission();
  
  Wire.beginTransmission(foundAddress);
  Wire.write(0x0A);  // Register 10 (control register 2)
  Wire.write(0x40);  // Set 512 gain
  int writeResult2 = Wire.endTransmission();
  
  Wire.beginTransmission(foundAddress);
  Wire.write(0x0B);  // Register 11 (set/reset period)
  Wire.write(0x01);  // Recommended value
  int writeResult3 = Wire.endTransmission();
  
  if (writeResult1 != 0 || writeResult2 != 0 || writeResult3 != 0) {
    Serial.println("Failed to configure compass registers");
    compassStatus = false;
    return;
  }
  
  Serial.println("Manual initialization complete, now trying library init...");
  
  // Initialize using the library
  compass.init();
  // compass.setCalibration(-1640.0,1165.0,-566.0,2048.0,-2460.0,178.0);
  // compass.setCalibration(-1593.0,1403.0,-3006.0,-331.0,-2773.0,-20.0);
  // compass.setCalibration(-1393.0,1453.0,-2460.0,252.0,-2106.0,613.0);
  // compass.setCalibration(456.0,3188.0,-3301.0,-491.0,4177.0,6846.0);
  // compass.setCalibration(530.0,3206.0,-3066.0,-475.0,4117.0,6712.0);
  // compass.setCalibration(436.0,3371.0,-3431.0,-273.0,4083.0,6871.0);
  // compass.setCalibration(-1464.0,1486.0,-2231.0,475.0,-474.0,2424.0);
  // compass.setCalibration(-3307.0,-724.0,-174.0,2225.0,-7072.0,-4384.0);
  // compass.setCalibration(666.0,3210.0,-3129.0,-563.0,4351.0,6711.0);
  // compass.setCalibration(-1362.0,1288.0,-2082.0,563.0,-1188.0,1322.0);
  
  // compass.setCalibration(-352.0,2182.0,-2116.0,498.0,-2037.0,510.0);
  compass.setCalibration(-628.0,2071.0,-2095.0,576.0,-1997.0,606.0);
  // Give it some time to stabilize
  delay(200);
  
  // Read values to check if initialization worked
  compass.read();
  int16_t x = compass.getX();
  int16_t y = compass.getY();
  int16_t z = compass.getZ();
  
  Serial.print("Initial compass readings: X=");
  Serial.print(x);
  Serial.print(", Y=");
  Serial.print(y);
  Serial.print(", Z=");
  Serial.println(z);
  
  // Add some diagnostic information
  if (x == 0 && y == 0 && z == 0) {
    Serial.println("Warning: Compass returning all zeros.");
    Serial.println("Trying compass recovery procedures...");
    
    // Attempt recovery with slower I2C speed
    Wire.setClock(100000); // Use slower I2C speed (100kHz instead of 400kHz)
    
    // Try direct register reads to verify communication
    Wire.beginTransmission(foundAddress);
    Wire.write(0x00); // X-axis data register
    Wire.endTransmission(false);
    
    uint8_t buffer[6];
    Wire.requestFrom(foundAddress, (uint8_t)6);
    
    if (Wire.available() >= 6) {
      for (int i = 0; i < 6; i++) {
        buffer[i] = Wire.read();
      }
      
      // Process raw data (LSB first for QMC5883L)
      int16_t rawX = (buffer[1] << 8) | buffer[0];
      int16_t rawY = (buffer[3] << 8) | buffer[2];
      int16_t rawZ = (buffer[5] << 8) | buffer[4];
      
      Serial.print("Raw register reads: X=");
      Serial.print(rawX);
      Serial.print(", Y=");
      Serial.print(rawY);
      Serial.print(", Z=");
      Serial.println(rawZ);
      
      // Try the library read again
      compass.read();
      x = compass.getX();
      y = compass.getY();
      z = compass.getZ();
      
      Serial.print("After recovery attempt: X=");
      Serial.print(x);
      Serial.print(", Y=");
      Serial.print(y);
      Serial.print(", Z=");
      Serial.println(z);
    } else {
      Serial.println("Failed to read compass registers directly");
    }
    
    // Try alternative approach for compass
    if (!tryAlternativeCompassApproach()) {
      Serial.println("Alternative approach for compass failed.");
    }
  }
  
  // Final status check
  compassStatus = (x != 0 || y != 0 || z != 0);
  
  if (compassStatus) {
    Serial.println("QMC5883L Compass initialized successfully!");
  } else {
    Serial.println("QMC5883L Compass initialization failed - check hardware connections");
    Serial.println("TROUBLESHOOTING TIPS:");
    Serial.println("1. Check power connections (3.3V required)");
    Serial.println("2. Verify I2C connections (SDA, SCL)");
    Serial.println("3. Try adding pull-up resistors (4.7kΩ) to SDA and SCL");
    Serial.println("4. Move compass away from magnetic interference");
    Serial.println("5. Check for damaged components");
  }
  
  // Initialize GPS module
  Serial.println("Initializing GPS module...");
  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  Serial.println("GPS serial initialized");
  
  // Configure GPS module for rocket state estimation
  Serial.println("Configuring GPS for optimal rocket state estimation...");
  // configureGPSForRocket();
  
  // Update LED status based on initialization results
  updateLedStatus();
  
  // delay(1000);  // Short delay to stabilize sensors
  
  Serial.println("\nGPS TROUBLESHOOTING GUIDE:");
  Serial.println("1. Make sure the GPS antenna has a clear view of the sky");
  Serial.println("2. First fix may take several minutes (cold start)");
  Serial.println("3. Check that GPS antenna is properly connected");
  Serial.println("4. Verify power supply is stable (GPS needs ~50mA)");
  Serial.println("5. Try moving to a different location");
  Serial.println("\nSystem is now running. LED will blink to indicate activity.");
}

void loop() {
  // Read data from all sensors
  readGY87Data();
  // Skipping GPS data reads since we're ignoring GPS
  
  // Print only raw sensor data
  printRawSensorData();
  
  // Update LED status
  updateLedStatus();
  
  // Blink the LED to show the script is running
  blinkStatusLED();
  
  delay(1);  // Update 10 times per second for more responsive operation
}

void blinkStatusLED() {
  // Create a "heartbeat" effect while maintaining the status color
  unsigned long currentTime = millis();
  
  // Every 1 second, do a quick blink
  if (currentTime - lastBlinkTime > 1000) {
    lastBlinkTime = currentTime;
    
    // Store the current color
    uint32_t currentColor = pixel.getPixelColor(0);
    
    // Flash with the heartbeat color
    pixel.setPixelColor(0, COLOR_HEARTBEAT);
    pixel.show();
    if (currentTime - lastBlinkTime > 50) {
      // Restore the status color
      pixel.setPixelColor(0, currentColor);
      pixel.show();
    }
  }
}

void configureGPSForRocket() {
  // Step 1: Reset the module to ensure clean configuration
  gpsSerial.println("$PMTK104*37");
  delay(1000);
  
  // Step 2: Set baud rate to maximum supported (115200 bps)
  gpsSerial.println("$PMTK251,115200*1F");
  delay(100);
  
  // Need to reinitialize serial communication at new baud rate
  gpsSerial.flush();
  gpsSerial.begin(115200, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);
  delay(100);
  
  // Step 3: Set update rate to maximum (10Hz)
  gpsSerial.println("$PMTK220,100*2F");
  delay(100);
  
  // Step 4: Select only the essential NMEA sentences
  // Enable RMC (position, velocity, time) and GGA (fix data)
  // Disable all others to maximize bandwidth efficiency
  gpsSerial.println("$PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0*28");
  delay(100);
  
  // Step 5: Enable SBAS (Satellite-Based Augmentation System) for improved accuracy
  gpsSerial.println("$PMTK313,1*2E");
  delay(100);
  
  // Step 6: Enable SBAS satellites for ranging (use in position calculation)
  gpsSerial.println("$PMTK301,2*2E");
  delay(100);
  
  // Step 7: Set navigation mode to Aviation (less filtering, more responsive)
  // Note: This is a u-blox specific UBX command
  // CFG-NAV5 message with Aviation dynamics model (8)
  uint8_t cfgNav5[] = {
    0xB5, 0x62, 0x06, 0x24, 0x24, 0x00, 0xFF, 0xFF, 0x08, 0x03, 0x00, 0x00, 0x00, 0x00, 
    0x10, 0x27, 0x00, 0x00, 0x05, 0x00, 0xFA, 0x00, 0xFA, 0x00, 0x64, 0x00, 0x2C, 0x01, 
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0xDC
  };
  
  for (int i = 0; i < sizeof(cfgNav5); i++) {
    gpsSerial.write(cfgNav5[i]);
  }
  delay(250);
  
  Serial.println("GPS configured for optimal rocket state estimation");
  Serial.println("- Update rate: 10Hz");
  Serial.println("- Baud rate: 115200");
  Serial.println("- NMEA sentences: RMC, GGA only");
  Serial.println("- SBAS enabled for improved accuracy");
  Serial.println("- Aviation dynamic mode for better response");
}

void updateLedStatus() {
  if (!mpuStatus || !bmpStatus || !compassStatus) {
    // Error state - one or more sensors failed to initialize
    pixel.setPixelColor(0, COLOR_ERROR);
  } else if (gpsStatus) {
    // Success state - all sensors working and GPS has fix
    pixel.setPixelColor(0, COLOR_SUCCESS);
  } else if (gpsDataReceived) {
    // Receiving NMEA data but no fix yet
    pixel.setPixelColor(0, COLOR_GPS_DATA);
  } else {
    // Waiting state - waiting for GPS data
    pixel.setPixelColor(0, COLOR_WAITING);
  }
  pixel.show();
}

void readGY87Data() {
  // Read MPU6050 data
  if (mpuStatus) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    // Data is stored in the sensors_event_t structures
  }
  
  // Read BMP085 data
  if (bmpStatus) {
    sensors_event_t event;
    bmp.getEvent(&event);
  }
  
  // Read Compass data
  if (compassStatus) {
    compass.read();
  }
}

void readGPSData() {
  nmeaLineComplete = false;
  nmeaBuffer = "";
  
  // Read GPS data
  while (gpsSerial.available() > 0) {
    char c = gpsSerial.read();
    
    // Update timestamp for last character received
    lastGpsCharTime = millis();
    gpsDataReceived = true;
    
    // Add character to NMEA buffer for debugging
    if (c == '\r') {
      // Ignore carriage returns
    } else if (c == '\n') {
      // End of NMEA sentence
      nmeaLineComplete = true;
      
      // Print raw NMEA data for debugging
      Serial.print("NMEA: ");
      Serial.println(nmeaBuffer);
      nmeaBuffer = "";
    } else {
      nmeaBuffer += c;
    }
    
    // Process with TinyGPS++
    if (gps.encode(c)) {
      // Check if we have a valid GPS fix
      if (gps.location.isValid() && gps.date.isValid() && gps.time.isValid()) {
        gpsStatus = true;
      } else {
        gpsStatus = false;
      }
    }
  }
}

void addGPSDiagnostics() {
  // Update satellite statistics
  if (gps.satellites.isValid()) {
    if (gps.satellites.value() > maxSatellitesEverSeen) {
      maxSatellitesEverSeen = gps.satellites.value();
    }
  }
  
  // Print diagnostic info every 30 seconds
  if (millis() - lastDiagnosticsTime > 30000) {
    lastDiagnosticsTime = millis();
    
    Serial.println("\n--- GPS DIAGNOSTICS ---");
    Serial.print("Maximum satellites ever seen: ");
    Serial.println(maxSatellitesEverSeen);
    
    if (maxSatellitesEverSeen == 0) {
      Serial.println("DIAGNOSIS: No satellites detected at all. Likely causes:");
      Serial.println("1. Antenna issue (disconnected, damaged)");
      Serial.println("2. Very poor location with no sky visibility");
      Serial.println("3. Module hardware failure");
      Serial.println("ACTION: Try a different antenna or location");
    } else if (maxSatellitesEverSeen < 3) {
      Serial.println("DIAGNOSIS: Some satellites detected but not enough for a fix");
      Serial.println("1. Antenna is working but reception is very poor");
      Serial.println("2. Location has limited sky visibility");
      Serial.println("ACTION: Try improving antenna position or move to better location");
    } else if (maxSatellitesEverSeen >= 3 && !gps.location.isValid()) {
      Serial.println("DIAGNOSIS: Enough satellites detected but no fix acquired");
      Serial.println("1. Poor signal quality or satellite geometry");
      Serial.println("2. Need more time to acquire fix");
      Serial.println("ACTION: Wait longer or try slightly different antenna position");
    }
    
    Serial.println("------------------------\n");
  }
}

void printSensorData() {
  static unsigned long lastPrintTime = 0;
  
  // Only print every second to avoid flooding serial monitor
  if (millis() - lastPrintTime < 1000) {
    return;
  }
  
  lastPrintTime = millis();
  
  Serial.println("------------------------------------");
  Serial.println("Sensor Readings:");
  
  // Print MPU6050 data
  if (mpuStatus) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    Serial.println("MPU6050:");
    Serial.print("  Acceleration (m/s²): X=");
    Serial.print(a.acceleration.x);
    Serial.print(", Y=");
    Serial.print(a.acceleration.y);
    Serial.print(", Z=");
    Serial.println(a.acceleration.z);
    
    Serial.print("  Rotation (rad/s): X=");
    Serial.print(g.gyro.x);
    Serial.print(", Y=");
    Serial.print(g.gyro.y);
    Serial.print(", Z=");
    Serial.println(g.gyro.z);
    
    Serial.print("  Temperature: ");
    Serial.print(temp.temperature);
    Serial.println(" °C");
  } else {
    Serial.println("MPU6050: Not available");
  }
  
  // Print BMP085 data
  if (bmpStatus) {
    sensors_event_t event;
    bmp.getEvent(&event);
    
    Serial.println("BMP085:");
    if (event.pressure) {
      Serial.print("  Pressure: ");
      Serial.print(event.pressure);
      Serial.println(" hPa");
      
      float seaLevelPressure = SENSORS_PRESSURE_SEALEVELHPA;
      float altitude = bmp.pressureToAltitude(seaLevelPressure, event.pressure);
      Serial.print("  Altitude: ");
      Serial.print(altitude);
      Serial.println(" m");
    } else {
      Serial.println("  Pressure sensor error");
    }
    
    float temperature;
    bmp.getTemperature(&temperature);
    Serial.print("  Temperature: ");
    Serial.print(temperature);
    Serial.println(" °C");
  } else {
    Serial.println("BMP085: Not available");
  }
  
  // Print Compass data
  if (compassStatus) {
    Serial.println("QMC5883L Compass:");
    Serial.print("  X: ");
    Serial.print(compass.getX());
    Serial.print("  Y: ");
    Serial.print(compass.getY());
    Serial.print("  Z: ");
    Serial.println(compass.getZ());
    
    Serial.print("  Azimuth: ");
    Serial.print(compass.getAzimuth());
    Serial.println(" degrees");
    
    const char* directions[] = {"North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"};
    Serial.print("  Direction: ");
    Serial.println(directions[compass.getBearing(compass.getAzimuth())]);
  } else {
    Serial.println("QMC5883L Compass: Not available");
  }
  
  // Print GPS data
  Serial.println("GPS Module:");
  if (gpsStatus) {
    Serial.print("  Location: ");
    Serial.print(gps.location.lat(), 6);
    Serial.print(", ");
    Serial.println(gps.location.lng(), 6);
    
    Serial.print("  Altitude: ");
    Serial.print(gps.altitude.meters());
    Serial.println(" m");
    
    Serial.print("  Date/Time: ");
    Serial.print(gps.date.year());
    Serial.print("-");
    Serial.print(gps.date.month());
    Serial.print("-");
    Serial.print(gps.date.day());
    Serial.print(" ");
    Serial.print(gps.time.hour());
    Serial.print(":");
    Serial.print(gps.time.minute());
    Serial.print(":");
    Serial.println(gps.time.second());
    
    Serial.print("  Speed: ");
    Serial.print(gps.speed.kmph());
    Serial.println(" km/h");
    
    Serial.print("  Course: ");
    Serial.print(gps.course.deg());
    Serial.println(" degrees");
    
    Serial.print("  Satellites: ");
    Serial.println(gps.satellites.value());
  } else if (gpsDataReceived) {
    Serial.println("  Receiving NMEA data but no GPS fix yet");
    Serial.println("  GPS satellite signal acquisition in progress...");
    
    // Print satellites in view if available
    if (gps.satellites.isValid()) {
      Serial.print("  Satellites in view: ");
      Serial.println(gps.satellites.value());
    }
  } else {
    Serial.println("  No GPS data received!");
    Serial.println("  Check wiring connections to GPS module");
  }
  
  Serial.println("------------------------------------");
}

void scanI2CDevices() {
  byte error, address;
  int nDevices = 0;

  Serial.println("Scanning for I2C devices...");

  for (address = 1; address < 128; address++) {
    // The i2c_scanner uses the return value of
    // the Write.endTransmission to see if
    // a device did acknowledge to the address.
    Wire.beginTransmission(address);
    error = Wire.endTransmission();

    if (error == 0) {
      Serial.print("I2C device found at address 0x");
      if (address < 16) {
        Serial.print("0");
      }
      Serial.print(address, HEX);
      
      // Try to identify known devices
      String deviceName = "unknown device";
      if (address == 0x0D) deviceName = "QMC5883L Compass";
      else if (address == 0x1E) deviceName = "HMC5883L/QMC5883L Compass";
      else if (address == 0x68) deviceName = "MPU6050 Gyro/Accel";
      else if (address == 0x76 || address == 0x77) deviceName = "BMP085/BMP180 Barometer";
      else if (address == 0x48) deviceName = "ADS1115 ADC";
      
      Serial.print(" (");
      Serial.print(deviceName);
      Serial.println(")");
      
      nDevices++;
    } else if (error == 4) {
      Serial.print("Unknown error at address 0x");
      if (address < 16) {
        Serial.print("0");
      }
      Serial.println(address, HEX);
    }
  }
  
  if (nDevices == 0) {
    Serial.println("No I2C devices found! Check your wiring.");
    Serial.println("Common issues:");
    Serial.println("1. Power not connected to sensors");
    Serial.println("2. Wrong I2C pins used");
    Serial.println("3. Pull-up resistors missing on SDA/SCL");
  } else {
    Serial.print("Found ");
    Serial.print(nDevices);
    Serial.println(" I2C device(s)");
  }
}

bool tryAlternativeCompassApproach() {
  Serial.println("Trying alternative approach for QMC5883L compass...");
  
  // QMC5883L registers and values
  const byte QMC_ADDRESS = 0x0D;
  const byte REG_CONFIG_1 = 0x09;
  const byte REG_CONFIG_2 = 0x0A;
  const byte REG_PERIOD = 0x0B;
  const byte REG_DATA_X_LSB = 0x00;
  
  // Try to communicate directly with the device
  Wire.beginTransmission(QMC_ADDRESS);
  Wire.write(REG_CONFIG_1);
  // Set continuous mode (0x01), 8 samples/sec (0x10), and +/-2G range (0x08)
  Wire.write(0x19); 
  if (Wire.endTransmission() != 0) {
    Serial.println("Failed to write to REG_CONFIG_1");
    return false;
  }
  delay(50);
  
  // Try to read data
  Wire.beginTransmission(QMC_ADDRESS);
  Wire.write(REG_DATA_X_LSB);
  if (Wire.endTransmission(false) != 0) {
    Serial.println("Failed to set read address");
    return false;
  }
  
  // Request 6 bytes (X, Y, Z data)
  Wire.requestFrom(QMC_ADDRESS, (uint8_t)6);
  if (Wire.available() != 6) {
    Serial.println("Couldn't read 6 bytes from QMC5883L");
    return false;
  }
  
  // Read the data
  int16_t x = Wire.read() | (Wire.read() << 8);
  int16_t y = Wire.read() | (Wire.read() << 8);
  int16_t z = Wire.read() | (Wire.read() << 8);
  
  Serial.print("Raw direct readings - X: ");
  Serial.print(x);
  Serial.print(", Y: ");
  Serial.print(y);
  Serial.print(", Z: ");
  Serial.println(z);
  
  return (x != 0 || y != 0 || z != 0);
}

// Add this new function for raw data output
void printRawSensorData() {
  static unsigned long lastRawPrintTime = 0;
  
  // Output at a higher rate (10 Hz)
  if (millis() - lastRawPrintTime < 10) {
    return;
  }
  
  lastRawPrintTime = millis();
  
  // Variables to store sensor data
  float accel_x = 0, accel_y = 0, accel_z = 0;
  float gyro_x = 0, gyro_y = 0, gyro_z = 0;
  int16_t mag_x = 0, mag_y = 0, mag_z = 0;
  
  // Get MPU6050 data
  if (mpuStatus) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    accel_x = a.acceleration.x;
    accel_y = a.acceleration.y;
    accel_z = a.acceleration.z;
    gyro_x = g.gyro.x;
    gyro_y = g.gyro.y;
    gyro_z = g.gyro.z;
  }
  
  // Get Compass data
  if (compassStatus) {
    compass.read();
    mag_x = compass.getX();
    mag_y = compass.getY();
    mag_z = compass.getZ();
  }
  
  // Print in the requested format
  Serial.print("Raw:");
  Serial.print(accel_x); Serial.print(",");
  Serial.print(accel_y); Serial.print(",");
  Serial.print(accel_z); Serial.print(",");
  Serial.print(gyro_x); Serial.print(",");
  Serial.print(gyro_y); Serial.print(",");
  Serial.print(gyro_z); Serial.print(",");
  // Serial.print("Heading:" );
  // Serial.print(compass.getAzimuth()); Serial.print(",");
  Serial.print(mag_x); Serial.print(",");
  Serial.print(mag_y); Serial.print(",");
  Serial.println(mag_z);
}