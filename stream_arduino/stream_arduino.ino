const int FROM_DMD_OUT_pin = 2;
const int PD_pin = A0;
const int PD_pin_two = A1;
String x;

void setup() {
    Serial.begin(500000); // Start serial communication at 115200 baud rate
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
  }
  else if (x == "STOP") {
            detachInterrupt(digitalPinToInterrupt(FROM_DMD_OUT_pin));
        }
}
void handleInterrupt() {
      int data1 = analogRead(PD_pin);      // Read from first photodiode
      int data2 = analogRead(PD_pin_two);  // Read from second photodiode

      // Send the data directly
      Serial.println(data1);
      Serial.println(data2);
}
