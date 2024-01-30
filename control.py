from DMD_main import control_DMD
import generate_pattern

call_pattern = generate_pattern.DmdPattern('hadamard', 128,128, gray_scale=255)
hadamard_pattern, _ = call_pattern.execute(length=0.5)
print(len(hadamard_pattern))
print("here")
#hadamard_pattern = generate_pattern.embed(hadamard_pattern)

# define a 3D array with form [[mask], [mask], [mask]...]
cd = control_DMD(hadamard_pattern, project_name = "my_project", group =  17, main_sequence_itr=1, frame_time=5)
cd.execute()