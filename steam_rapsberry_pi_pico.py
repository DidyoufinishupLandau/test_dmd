import os
import time
from machine import Pin, ADC
import utime
import sys

# Initialize connections
# Trigger 0 is connected to DMD output. Set as in, pull down
FROM_DMD_OUT_pin: Pin = Pin(0, Pin.IN, Pin.PULL_DOWN)
# Trigger 1 is connected to DMD input
TO_DMD_IN_pin: Pin = Pin(1, Pin.OUT)
# PD_pin is connected to ADC0
PD_pin: ADC = ADC(0)
PD_pin_two: ADC = ADC(1)

# Global variables
_NO_OF_IMAGES: int = 4096
image_size = 128
_DELAY: int = 0
_START: bool = False
_READY_FOR_ACQ: bool = False
_DATA: list = []
_DATA_two: list = []
_ACQ_COUNTER: int = 0
_Data_counter = 0


def handle_interrupt(interrupt_pin: Pin) -> None:
    """Define interrupt  handler - must be fast and inner loop"""
    global _ACQ_COUNTER
    global _DATA
    # Increment Acquisition handler
    _ACQ_COUNTER += 1
    # print(sys.stdin.buffer.read(read_PD()))
    _DATA.append(read_PD())
    _DATA_two.append(read_PD_two())


def read_PD() -> float:
    """Read from the photodiode pin, return as u16"""

    return PD_pin.read_u16()


def read_PD_two():
    return PD_pin_two.read_u16()


def send_trigger(sleep_time_us: int = 10):
    """Send the output trigger high and low"""
    TO_DMD_IN_pin.value(1)
    # Define sleep-time
    if sleep_time_us > 0:
        utime.sleep_us(sleep_time_us)
    TO_DMD_IN_pin.value(0)


def acquire(no_of_images: int, delay: int = 0) -> list:
    """Primary acquire loop"""
    global _READY_FOR_ACQ
    global _DATA
    global _ACQ_COUNTER
    global _DATA_two
    global _Data_counter
    # Attach handle_interrupt to the falling edge of Pin
    FROM_DMD_OUT_pin.irq(trigger=Pin.IRQ_FALLING, handler=handle_interrupt)
    return None


def Read() -> str:
    return sys.stdin.readline().encode('UTF-8')


def Write(data: list):
    if len(_DATA) > _NO_OF_IMAGES:
        for i in range(_NO_OF_IMAGES):
            print(_DATA[i + 1])
    print("END")


def restart():
    global _NO_OF_IMAGES
    global _DELAY
    global _START
    global _DATA
    global _DATA_two
    global _ACQ_COUNTER
    _NO_OF_IMAGES = 4096
    _DELAY = 0
    _START = False
    _DATA = []
    _DATA_two = []
    _ACQ_COUNTER = 0


def commands_parser(comm: str) -> None:
    global _NO_OF_IMAGES
    global _DELAY
    global _START
    global _DATA
    global _DATA_two

    led = Pin(25, Pin.OUT)
    if "N_" in comm:
        # Set the number of images
        _NO_OF_IMAGES = int(comm.replace("N_", ""))
    if "D_" in comm and not ("LED_" in comm):
        # Set the internal delay
        _DELAY = int(comm.replace("D_", ""))
    if "S_" in comm:
        # Set the start flag
        if "TRUE" in comm:
            _START = True
        elif "FALSE" in comm:
            _START = False
    # Helper function
    if "LED_ON" in comm:
        led.high()
    if "LED_OFF" in comm:
        led.low()

    if "INFO" in comm:
        print("Number of images: ", _NO_OF_IMAGES)
        print("Delay after a single data acquisition (us): ", _DELAY)
        print("Data size: ", len(_DATA))
        print("Data size two: ", len(_DATA_two))

    # GD = Get Data
    if "GD" in comm:
        Write(_DATA)
    # Reset Data

    if "RD" in comm:
        _DATA = []
    if "one_length" in comm:
        print(len(_DATA))
    if "two_legnth" in comm:
        print(len(_DATA_two))
    if "ShowDataOne" in comm:
        for line in _DATA:
            print(line)
    if "ShowDataTwo" in comm:
        for line in _DATA_two:
            print(line)
    if "C" in comm:
        restart()
    if "*IDN" in comm:
        print("PicoADC v1.0")


def main():
    """Main loop"""
    global _START
    _START = False
    # Continuously read
    while True:
        text = Read()
        if len(text) != 0:
            # Parse
            commands_parser(text)
        if _START:
            acquire(_NO_OF_IMAGES, _DELAY)
            _START = False


main()