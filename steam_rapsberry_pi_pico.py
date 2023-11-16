import sys
from machine import Pin, ADC
import utime

# Constants
NO_OF_IMAGES_DEFAULT = 4096
PIN_FROM_DMD_OUT = 0
PIN_TO_DMD_IN = 1
PIN_PD = 0
PIN_PD_TWO = 1
PIN_LED = 25

# Initialize connections
from_dmd_out_pin = Pin(PIN_FROM_DMD_OUT, Pin.IN, Pin.PULL_DOWN)
to_dmd_in_pin = Pin(PIN_TO_DMD_IN, Pin.OUT)
pd_pin = ADC(PIN_PD)
pd_pin_two = ADC(PIN_PD_TWO)
led = Pin(PIN_LED, Pin.OUT)

# Class for handling device state
class DeviceState:
    def __init__(self):
        self.no_of_images = NO_OF_IMAGES_DEFAULT
        self.delay = 0
        self.start = False
        self.data = []
        self.data_two = []
        self.acq_counter = 0

    def reset(self):
        self.no_of_images = NO_OF_IMAGES_DEFAULT
        self.delay = 0
        self.start = False
        self.data = []
        self.data_two = []
        self.acq_counter = 0

state = DeviceState()

def handle_interrupt(interrupt_pin):
    state.acq_counter += 1
    state.data.append(pd_pin.read_u16())
    state.data_two.append(pd_pin_two.read_u16())

def send_trigger(sleep_time_us=10):
    to_dmd_in_pin.value(1)
    if sleep_time_us > 0:
        utime.sleep_us(sleep_time_us)
    to_dmd_in_pin.value(0)

def acquire():
    from_dmd_out_pin.irq(trigger=Pin.IRQ_FALLING, handler=handle_interrupt)
    # Rest of the acquisition logic here

def read_input():
    return sys.stdin.readline().encode('UTF-8')

def commands_parser(comm):
    if "N_" in comm:
        state.no_of_images = int(comm.replace("N_", ""))
    elif "D_" in comm:
        state.delay = int(comm.replace("D_", ""))
    elif "S_TRUE" in comm:
        state.start = True
    elif "S_FALSE" in comm:
        state.start = False
    elif "LED_ON" in comm:
        led.high()
    elif "LED_OFF" in comm:
        led.low()
    elif "INFO" in comm:
        print_info()
    elif "one_length" in comm:
        print(len(state.data))
    elif "two_length" in comm:
        print(len(state.data_two))
    elif "ShowDataOne" in comm:
        print_data(state.data)
    elif "ShowDataTwo" in comm:
        print_data(state.data_two)
    elif "C" in comm:
        state.reset()
    elif "*IDN" in comm:
        print("PicoADC v1.0")

def print_info():
    print("Number of images: ", state.no_of_images)
    print("Delay after a single data acquisition (us): ", state.delay)
    print("Data size: ", len(state.data))
    print("Data size two: ", len(state.data_two))

def print_data(data):
    for line in data:
        print(line)
    data.clear()

def main():
    while True:
        text = read_input()
        if text:
            commands_parser(text)
        if state.start:
            acquire()
            state.start = False

main()
