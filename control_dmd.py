import numpy as np
from DMD_main import control_DMD
import time
import generate_pattern


call_pattern = generate_pattern.DmdPattern('hadamard', 128,128, gray_scale=255)
hadamard_pattern, _ = call_pattern.execute(length=1)
print(len(hadamard_pattern))
print("here")
#hadamard_pattern = generate_pattern.embed(hadamard_pattern)

# define a 3D array with form [[mask], [mask], [mask]...]
cd = control_DMD(hadamard_pattern, project_name = "my_project", group = 44, main_sequence_itr=1, frame_time=1)
for i in range(1):
    cd.execute()

"""cd = control_DMD(inverse_patter, project_name = "my_project", main_sequence_itr=1, frame_time=10)
for i in range(1, 2):
    cd.execute(1024*(i-1), 4096*i)"""