#include <stdio.h>
#include "driver/pcnt.h"

pcnt_config_t pcnt_config_l;
const int FROM_DMD_OUT_pin = 12;
int periods = 0;
//int frame = 0;
volatile bool valve = false;
int16_t old_count = 0;

static void pcnt_example_init(void) {
    // Set up pulse counting on GPIO4
    pcnt_config_l.pulse_gpio_num = 4; 
    pcnt_config_l.ctrl_gpio_num = PCNT_PIN_NOT_USED;
    // Attach to pulse counter channel1, pulse 1
    pcnt_config_l.channel = PCNT_CHANNEL_0;
    pcnt_config_l.unit = PCNT_UNIT_0;
    // Increment on arrival, ignore negative mode
    pcnt_config_l.pos_mode = PCNT_COUNT_INC;
    pcnt_config_l.neg_mode = PCNT_COUNT_DIS;
    // Keep (don't reset)
    pcnt_config_l.lctrl_mode = PCNT_MODE_KEEP;
    pcnt_config_l.hctrl_mode = PCNT_MODE_KEEP;
    // Maximum is 16bit integer
    pcnt_config_l.counter_h_lim = INT16_MAX;
    pcnt_config_l.counter_l_lim = INT16_MIN;

    // Send config
    pcnt_unit_config(&pcnt_config_l);
    // Debounce connection
    pcnt_set_filter_value(PCNT_UNIT_0, 2);
    // Reset counter to zero
    pcnt_counter_pause(PCNT_UNIT_0);
    pcnt_counter_clear(PCNT_UNIT_0);
    pcnt_counter_resume(PCNT_UNIT_0);
}

void setup() {
  // Calls the initialization
    pcnt_example_init();
    // Set up serial port communiation to 250kbaud
    Serial.begin(500000);
    // Overclock CPU (better speed)
    setCpuFrequencyMhz(240);
    // Attach a trigger from DMD 'new frame' output to "handleInterrupt" command
    attachInterrupt(digitalPinToInterrupt(FROM_DMD_OUT_pin), handleInterrupt, RISING);
    //
}

void loop() {
  // Set up a counter
    // Get the count value into count
    int16_t count = 0;
    pcnt_get_counter_value(PCNT_UNIT_0, &count); 
    // If counts is less than previous count and the difference is larger than 15000 (ie 2^14)
    // then we increment a periods counter
    if (count < old_count && (old_count - count) > 25000) {
        periods++;
    }
    // Store old count
    old_count = count;
    // Check if new frame has arrived in the meantime
    if (valve) {
      // If so, send the total photons arrived - here periods * 2^15 plus this count
        //Serial.println(frame);
        //Serial.println(periods);
        //Serial.println(old_count);
        Serial.println(periods*32768 + old_count);
        // reset periods
        periods = 0;
        old_count = 0;
        // reset value
        valve = false;
        pcnt_counter_pause(PCNT_UNIT_0);
        pcnt_counter_clear(PCNT_UNIT_0);
        pcnt_counter_resume(PCNT_UNIT_0);
    }
}

void handleInterrupt() {
  // If new frame arrives, set valve to positive
    valve = true;
}
