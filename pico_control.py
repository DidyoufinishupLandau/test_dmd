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
_DELAY: int = 0
_START: bool = False
_READY_FOR_ACQ: bool = False
_DATA: list = []
_DATA_two: list = []
_ACQ_COUNTER: int = 0
_Data_counter = 0
num_repeat = 1

def handle_interrupt(interrupt_pin: Pin) -> None:
    """Define interrupt  handler - must be fast and inner loop"""
    global _ACQ_COUNTER
    global _DATA
    # Increment Acquisition handler
    _ACQ_COUNTER += 1
    #print(sys.stdin.buffer.read(read_PD()))
    _DATA.append(read_PD())
    _DATA_two.append(read_PD_two())


def read_PD() -> float:
    """Read from the photodiode pin, return as u16"""

    return PD_pin.read_u16()
def read_PD_two():
    return PD_pin_two.read_u16()


def send_trigger(sleep_time_us: int = 1):
    """Send the output trigger high and low"""
    TO_DMD_IN_pin.value(1)
    # Define sleep-time
    if sleep_time_us > 0:
        utime.sleep_us(sleep_time_us)
    TO_DMD_IN_pin.value(0)


def disable_input_trigger():
    FROM_DMD_OUT_pin.remove_program()  # I cannot find a method that would disable the trigge, irq_clear() doesn't work


def acquire(no_of_images: int, delay: int = 0) -> list:
    """Primary acquire loop"""
    global _READY_FOR_ACQ
    global _DATA
    global _ACQ_COUNTER
    global _DATA_two
    global _Data_counter

    # Attach handle_interrupt to the falling edge of Pin
    FROM_DMD_OUT_pin.irq(trigger=Pin.IRQ_FALLING, handler=handle_interrupt)

    _DATA = []
    _ACQ_COUNTER = 0
    if delay > 0:
        while _ACQ_COUNTER < no_of_images + 1:
            send_trigger()
            
            utime.sleep_us(delay)

    else:
        print("no delay acquire")
        while _ACQ_COUNTER < no_of_images:
            send_trigger()
            #handle_interrupt(0)
        if _ACQ_COUNTER == no_of_images:
            print("saving_image")
            _Data_counter+=1
            name_one = f"one_data_{_Data_counter}.csv"
            name_two = f"two_data_{_Data_counter}.csv"
            str_data = str(_DATA)
            str_data = str_data[1:]
            str_data = str_data[:len(str_data)-1]
            str_data_two = str(_DATA_two)
            str_data_two = str_data_two[1:]
            str_data_two = str_data_two[:len(str_data_two)-1] 
            with open("lock.file", 'w') as lock:
                pass
            with open(name_one, 'w') as file_one:
                file_one.write(str_data)
            file_one.close()
            with open(name_two, 'w') as file_two:
                file_two.write(str_data_two)
            file_two.close()
        print("acquire_complete")
        _DATA = []
        _DATA_two = []
        _ACQ_COUNTER = 0
        print(_Data_counter)
        
    return None


def Read() -> str:
    return sys.stdin.readline()


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
    global _Data_counter
    _NO_OF_IMAGES = 4000
    _DELAY = 0
    _START = False
    _DATA = []
    _DATA_two = []
    _ACQ_COUNTER = 0
    _Data_counter = 0
    
def commands_parser(comm: str) -> None:
    global _NO_OF_IMAGES
    global _DELAY
    global _START
    global _DATA

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
    if "ShowDataOne" in comm:
        print(_DATA)
    if "ShowDataTwo" in comm:
        print(_DATA_two)
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
            for _ in range(num_repeat):
                acquire(_NO_OF_IMAGES, _DELAY)
            _START = False
main()
