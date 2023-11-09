import ajile_mock_driver as aj
import numpy as np
from DMD_driver import DMD_driver
import time
import pattern_generator
import generate_pattern # more efficient
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
    dmd.add_sequence_item(image=pattern_generator.one_side(), seq_id=1, frame_time=1000)

print(count)


# Create the main sequence
loop = True
while loop:
    # Start the sequence
    dmd.my_trigger()
    dmd.start_projecting()
    # Stop the sequence
    time.sleep(5)
    INPUT = input()
    if INPUT == "N":
        loop = False
dmd.stop_projecting()