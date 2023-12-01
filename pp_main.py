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

dmd = DMD_driver()
# Create a default project
dmd.create_project(project_name='test_project')
dmd.create_main_sequence(seq_rep_count=1)
# Image
count = 0
for i in range(0, 4096):
    count+=1
    dmd.add_sequence_item(image=pattern_generator.one_side(), seq_id=1, frame_time=3)

print(count)

# Create the main sequence
dmd.my_trigger()
dmd.start_projecting()
# Stop the sequence
time.sleep(50)
dmd.stop_projecting()