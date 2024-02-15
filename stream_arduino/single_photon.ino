#include <stdio.h>
#include "driver/pcnt.h"

pcnt_config_t pcnt_config_l;
const int FROM_DMD_OUT_pin = 13;
int periods = 0;
bool valve = false;
int16_t old_count = 0;

static void pcnt_example_init(void) {
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

    pcnt_unit_config(&pcnt_config_l);
    pcnt_set_filter_value(PCNT_UNIT_0, 2);
    pcnt_counter_pause(PCNT_UNIT_0);
    pcnt_counter_clear(PCNT_UNIT_0);
    pcnt_counter_resume(PCNT_UNIT_0);
}

void setup() {
    pcnt_example_init();
    Serial.begin(500000);
    setCpuFrequencyMhz(240);
    attachInterrupt(digitalPinToInterrupt(FROM_DMD_OUT_pin), handleInterrupt, FALLING);
}

void loop() {
    int16_t count;
    pcnt_get_counter_value(PCNT_UNIT_0, &count); 
    
    
    if (count < old_count && (old_count - count) > 15000) {
        periods++;
    }
    old_count = count;
    if (valve) {
        Serial.println(periods*32768 + old_count);
        periods = 0;
    }
    valve = false; 
}

void handleInterrupt() {
    valve = true;
}
