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
led = Pin(25, Pin.OUT, Pin.PULL_DOWN)
led.toggle()
# Global variables
_START: bool = False
_BREAK: bool = False
_DATA: list = []
_DATA_two: list = []
i = 0


def handle_interrupt(interrupt_pin: Pin) -> None:
    """Define interrupt  handler - must be fast and inner loop"""
    global _DATA
    global _DATA_two
    _DATA.append(read_PD())
    _DATA_two.append(read_PD_two())

def test_interrupt(pin):
    global i
    global _DATA
    global _DATA_two
    _DATA.append(i)
    _DATA_two.append(i)
    i+=1

def test_acquire():
    led.irq(trigger=Pin.IRQ_RISING, handler=test_interrupt)

def test_disable():
    led.irq(handler=None)
    
def read_PD() -> float:
    """Read from the photodiode pin, return as u16"""

    return PD_pin.read_u16()
def simu_signal():
    for i in range(4096):
        led.value(1)
        led.toggle()
        utime.sleep_us(10)
        led.value(0)
        led.toggle()
def read_PD_two():
    return PD_pin_two.read_u16()

def acquire() -> list:
    """Primary acquire loop"""
    # Attach handle_interrupt to the falling edge of Pin
    FROM_DMD_OUT_pin.irq(trigger=Pin.IRQ_FALLING, handler=handle_interrupt)
    
    return None


def disable_acquire():
    FROM_DMD_OUT_pin.irq(handler=None)

def Read() -> str:
    return sys.stdin.readline().encode('UTF-8')


def restart():
    global _START
    global _DATA
    global _DATA_two
    _START = False
    _DATA = []
    _DATA_two = []
    
def commands_parser(comm: str) -> None:
    global _START
    global _DATA
    global _DATA_two
    global i
    global _BREAK

    led = Pin(25, Pin.OUT)
    if "S_" in comm:
        # Set the start flag
        if "TRUE" in comm:
            _START = True
        elif "FALSE" in comm:
            _START = False

    
    if "LED_ON" in comm:
        led.high()
    if "LED_OFF" in comm:
        led.low()

    if "ShowData" in comm:
        for i in range(4096):
            print(_DATA[i])
            print(_DATA_two[i])
    if "RS" in comm:
        _DATA = []
        _DATA_two = []
        i = 0
    if "SM" in comm:
        simu_signal()
    if "STOP" in comm:
        test_disable()
    if "BREAK" in comm:
        _BREAK = True
def main():
    """Main loop"""
    global _START
    global _BREAK
    _START = False
    _BREAK = False
    # Continuously read
    while True:
        text = Read()
        if len(text) != 0:
            # Parse
            commands_parser(text)
        if _START:
            test_acquire()
            _START = False
        if _BREAK:
            machine.reset()
if __name__ == "__main__":
    main()
