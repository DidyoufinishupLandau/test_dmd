const int FROM_DMD_OUT_pin = 2; // Example pin for DMD output, set as input
const int PD_pin = A0;          // Analog input pin for photodiode
const int PD_pin_two = A1;      // Another analog input pin for a second photodiode
String x;
bool dataAcquisitionEnabled = false;
void setup() {
    Serial.begin(115200); // Start serial communication at 115200 baud rate
    pinMode(FROM_DMD_OUT_pin, INPUT);

    // Attach interrupt to the falling edge of the pin
    attachInterrupt(digitalPinToInterrupt(FROM_DMD_OUT_pin), handleInterrupt, FALLING);
}

void  loop() {
  while (!Serial.available());
  x = Serial.readStringUntil('\n');
  x.trim(); // Remove any whitespace
  if (x=="START"){
    attachInterrupt(digitalPinToInterrupt(FROM_DMD_OUT_pin), handleInterrupt, FALLING);
    Serial.println("Data acquisition started");
  }
  else if (x == "STOP") {
            dataAcquisitionEnabled = false;
            detachInterrupt(digitalPinToInterrupt(FROM_DMD_OUT_pin));
            Serial.println("Data acquisition stopped");
        }
  Serial.print(x);
}
void handleInterrupt() {
      int data1 = analogRead(PD_pin);      // Read from first photodiode
      int data2 = analogRead(PD_pin_two);  // Read from second photodiode

      // Send the data directly
      Serial.println(data1);
      Serial.println(data2);
}
