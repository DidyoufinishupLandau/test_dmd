import serial
import numpy as np
import sys

class StreamRaspberry:
    def __init__(self, PORT: str = 'COM3', BAUD_RATE: int = 9600, timeout: int = 3):
        self.ser = serial.Serial(PORT, BAUD_RATE, timeout=timeout)
        if self.ser.isOpen():
            print('Serial port opened:', self.ser.portstr)

    def get_data(self, num_frames, save_data=True, file_name='output'):
        data_one = []
        data_two = []

        try:
            print("Ready for commands (type 'start' to begin, 'break' to exit):")
            while True:
                command = sys.stdin.readline().strip()  # Use .strip() to remove whitespace
                if command == 'break':
                    break
                elif command == 'start':
                    print('Starting data acquisition...')
                    self.ser.write(b"START\n")
                    while len(data_one) < num_frames:
                        # Read a line of data from the Pico
                        line = self.ser.readline().decode('utf-8').strip()
                        print("Received:", line)  # Debug print

                        # Check if the line contains the expected data
                        if "ONE:" in line and "TWO:" in line:
                            try:
                                adc_data_one = int(line.split(",")[0].split(":")[1])
                                adc_data_two = int(line.split(",")[1].split(":")[1])

                                data_one.append(adc_data_one)
                                data_two.append(adc_data_two)
                            except ValueError:
                                print("Malformed data line received:", line)

                    self.ser.write(b"STOP\n")
                    print('Data acquisition stopped.')

                    if save_data:
                        data_one = np.array(data_one).reshape(len(data_one), 1)
                        data_two = np.array(data_two).reshape(len(data_two), 1)
                        data = np.hstack((data_one, data_two))
                        file_name += ".csv"
                        np.savetxt(file_name, data, delimiter=',')
                        return data

        except serial.SerialException as e:
            print("Serial error:", e)
        except serial.SerialTimeoutException as e:
            print("Serial timeout:", e)

        return data_one, data_two

# Usage
SR = StreamRaspberry()
data_one, data_two = SR.get_data(3)
print(data_one)
