import numpy as np
from DMD_main import control_DMD
import time
import generate_pattern

call_pattern = generate_pattern.DmdPattern('hadamard', 32, 32, gray_scale=255)
hadamard_pattern = call_pattern.execute()
inverse_patter = []
for i in range(len(hadamard_pattern)):
    inverse_patter.append((hadamard_pattern[i]==0).astype(int) *255)
# define a 3D array with form [[mask], [mask], [mask]...]
cd = control_DMD(hadamard_pattern, project_name = "my_project", main_sequence_itr=1, frame_time=40)
cd.execute(0,32*32)
cd = control_DMD(hadamard_pattern, project_name = "my_project", main_sequence_itr=1, frame_time=40)
cd.execute(0, 32*32)
"""for i in range(1, 3):
    cd.execute(1000*(i-1), 1000*i)"""