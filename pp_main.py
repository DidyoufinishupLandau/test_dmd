import ajile_mock_driver as aj
import numpy as np
from DMD_driver import DMD_driver
import time
import pattern_generator
import generate_pattern # more efficient
import serial
# Connect to the DMD
"""
This file is used for calibration
"""
pattern = generate_pattern.DmdPattern("hadamard", 32, 32)
pattern= pattern.execute()
dmd = DMD_driver()
# Create a default project
dmd.create_project(project_name='test_project')
dmd.create_main_sequence(seq_rep_count=1)
# Image
count = 0
for i in range(0, 500):
    count+=1
    dmd.add_sequence_item(image=pattern_generator.one_side(), seq_id=1, frame_time=10)

print(count)

ser = serial.Serial('COM4', 115200, timeout=3)
ser.write(b'S_TRUE\r\n')
# Create the main sequence
dmd.my_trigger()
dmd.start_projecting()
# Stop the sequence
time.sleep(5)
dmd.stop_projecting()
ser.write(b'LED_ON\r\n')
line = ser.write(b'ShowDataOne\r\n')
for _ in range(500):
    line = ser.readline()
    print(int(line[0:len(line)-2].decode("utf-8")))
line = ser.write(b"ShowDataTwo\r\n")
for _ in range(500):
    line = ser.readline()
    print(int(line[0:len(line)-2].decode("utf-8")))