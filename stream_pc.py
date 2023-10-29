import serial
import numpy as np

class StreamRaspberry:
    """
    Example usage:
    from stream_raspberry import StreamRaspberry
    sr = StreamRaspberry(port_name, baud_rate)
    data_one, data_two = sr.get_data(num_of_patterns, save_data = False)
    """
    def __init__(self, PORT: str='COM3', BAUD_RATE: int=115200):
        self.ser = serial.Serial(PORT, BAUD_RATE)
        self.ser.flush()

    def get_data(self, num_frames, save_data=True, file_name='output'):
        data_one = []
        data_two = []
        self.ser.flushInput()
        self.ser.write(b"START\n")
        while len(data_one) < num_frames:
            # Read a line of data from the Pico
            line = self.ser.readline().decode('utf-8').strip()

            # Get data from rapsberry pi pico.
            # I wrote this code to help the dynamic acquisation and adaptive sampling
            # I didn't consider if there are enough memory for your computer to do this.
            # if not, please reduce the num frames.
            if "ONE:" in line and "TWO:" in line:
                adc_data_one = int(line.split(",")[0].split(":")[1])
                adc_data_two = int(line.split(",")[1].split(":")[1])

                data_one.append(adc_data_one)
                data_two.append(adc_data_two)

        self.ser.write(b"STOP\n")

        if save_data:
            data_one = np.array(data_one).reshape(len(data_one), 1)
            data_two = np.array(data_two).reshape(len(data_two), 1)
            data = np.hstack((data_one, data_two))
            file_name +=".csv"
            np.savetxt(file_name, data, delimiter=',')
            return data

        return data_one, data_two
