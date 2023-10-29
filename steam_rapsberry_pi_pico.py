import time
from machine import Pin, ADC
import utime
import sys
"""
remember put this file in thonny, not in pycharm
"""

# Initialize connections
FROM_DMD_OUT_pin = Pin(0, Pin.IN, Pin.PULL_DOWN)
TO_DMD_IN_pin = Pin(1, Pin.OUT)
PD_pin = ADC(0)
PD_pin_two = ADC(1)

def handle_interrupt(interrupt_pin: Pin) -> None:
    """Define interrupt handler"""
    adc_data_one = read_PD()
    adc_data_two = read_PD_two()

    # Directly send data to the computer
    print(f"ONE:{adc_data_one},TWO:{adc_data_two}")

def read_PD() -> int:
    """Read from the first photodiode pin and return as u16"""
    return PD_pin.read_u16()

def read_PD_two() -> int:
    """Read from the second photodiode pin and return as u16"""
    return PD_pin_two.read_u16()

def send_trigger(sleep_time_us: int = 500):
    """Send the output trigger high and low"""
    TO_DMD_IN_pin.value(1)
    if sleep_time_us > 0:
        utime.sleep_us(sleep_time_us)
    TO_DMD_IN_pin.value(0)

def acquire(no_of_images: int, delay: int = 0) -> None:
    """Primary acquire loop"""
    FROM_DMD_OUT_pin.irq(trigger=Pin.IRQ_FALLING, handler=handle_interrupt)

def main():
    """Main loop"""
    while True:
        text = sys.stdin.readline().strip()
        if text == "START":
            acquire(_NO_OF_IMAGES)
        elif text == "STOP":
            break

main()
