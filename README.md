This file forks from DMD_driver created by Dr.Patrick Parkinson. The main goal is to create the software control system to drive the DMD belong to our quantum microscopy system.
The demo code has not presented yet as the functions porpocal might altered during the active development in the next 3 month.
The general routine to use this code is:

#create hadamard pattern

call_pattern = generate_pattern.DmdPattern('hadamard', 128,128, gray_scale=255)

hadamard_pattern, conjugate_hadamard_pattern = call_pattern.execute(length=1)

#call function control DMD to read data from photodiode which is drived by a microcontroller.

cd = control_DMD(hadamard_pattern, project_name = "my_project", group =  2, main_sequence_itr=1, frame_time=1)

#start project and save data into csv file sent from photodiode.

cd.execute(measurement_size=int(4096))
