#include <stdio.h>
#include "driver/pcnt.h"

pcnt_config_t pcnt_config_l;
const int FROM_DMD_OUT_pin = 12;
String x;
int16_t count = 0;

static void pcnt_example_init(void){
    pcnt_config_l.pulse_gpio_num = 13; 
    pcnt_config_l.ctrl_gpio_num = PCNT_PIN_NOT_USED; 
    pcnt_config_l.channel = PCNT_CHANNEL_0; 
    pcnt_config_l.unit = PCNT_UNIT_0; 
    pcnt_config_l.pos_mode = PCNT_COUNT_INC; 
    pcnt_config_l.neg_mode = PCNT_COUNT_DIS; 
    pcnt_config_l.lctrl_mode = PCNT_MODE_KEEP; 
    pcnt_config_l.hctrl_mode = PCNT_MODE_KEEP; 
    pcnt_config_l.counter_h_lim = INT16_MAX; 
    pcnt_config_l.counter_l_lim = INT16_MIN; 
    pcnt_config_l.pos_mode = PCNT_COUNT_INC; 
    pcnt_config_l.neg_mode = PCNT_COUNT_DIS; 

    pcnt_unit_config(&pcnt_config_l);
    pcnt_set_filter_value(PCNT_UNIT_0, 1023);
    pcnt_counter_pause(PCNT_UNIT_0);
    pcnt_counter_clear(PCNT_UNIT_0);
}

void setup() {
    pcnt_example_init();
    Serial.begin(250000); // Start serial communication at 250000 baud rate
    pinMode(FROM_DMD_OUT_pin, INPUT);
    // Attach interrupt to the falling edge of the pin
    attachInterrupt(digitalPinToInterrupt(FROM_DMD_OUT_pin), handleInterrupt, RISING);
    setCpuFrequencyMhz(240);
}

void loop() {
    while (!Serial.available());
    x = Serial.readStringUntil('\n');
    x.trim(); // Remove any whitespace
    if (x=="START") {
        attachInterrupt(digitalPinToInterrupt(FROM_DMD_OUT_pin), handleInterrupt, RISING);
    } else if (x == "STOP") {
        detachInterrupt(digitalPinToInterrupt(FROM_DMD_OUT_pin));
    } else if (x == "TEST") {
        Serial.println("1");
    }
}

void handleInterrupt() {
  int16_t temp = 0;
  pcnt_get_counter_value(PCNT_UNIT_0, &temp); // Get pulse count from pulse counter
  ets_delay_us(100);
  Serial.print("Pulse Count: ");
  Serial.println(count);
}
#include <stdio.h>
#include "driver/pcnt.h"
pcnt_config_t pcnt_config_l;

static void pcnt_example_init(void){
    pcnt_config_l.pulse_gpio_num = 12; 
    pcnt_config_l.ctrl_gpio_num = PCNT_PIN_NOT_USED; 
    pcnt_config_l.channel = PCNT_CHANNEL_0; 
    pcnt_config_l.unit = PCNT_UNIT_0; 
    pcnt_config_l.pos_mode = PCNT_COUNT_INC; 
    pcnt_config_l.neg_mode = PCNT_COUNT_DIS; 
    pcnt_config_l.lctrl_mode = PCNT_MODE_KEEP; 
    pcnt_config_l.hctrl_mode = PCNT_MODE_KEEP; 
    pcnt_config_l.counter_h_lim = INT16_MAX; 
    pcnt_config_l.counter_l_lim = INT16_MIN; 
    pcnt_config_l.pos_mode = PCNT_COUNT_INC; 
    pcnt_config_l.neg_mode = PCNT_COUNT_DIS; 

    pcnt_unit_config(&pcnt_config_l);
    pcnt_set_filter_value(PCNT_UNIT_0, 1023);
    pcnt_counter_pause(PCNT_UNIT_0);
    pcnt_counter_clear(PCNT_UNIT_0);
    pcnt_counter_resume(PCNT_UNIT_0);
}

void setup() {
    pcnt_example_init();
    Serial.begin(250000); // Start serial communication at 250000 baud rate

    setCpuFrequencyMhz(240);
}
void loop() {
  int16_t temp = 0;
  pcnt_get_counter_value(PCNT_UNIT_0, &temp); // Get pulse count from pulse counter
  Serial.println(temp);
}
