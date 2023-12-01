import serial
import numpy as np
import sys

class StreamRaspberry:
    def __init__(self, PORT: str = 'COM8', BAUD_RATE: int = 115200, timeout: int = 3):
        self.ser = serial.Serial(PORT, BAUD_RATE, timeout=timeout)
        if self.ser.isOpen():
            print('Serial port opened:', self.ser.portstr)
    def start_acquisation(self):
        self.ser.write(b"STOP\r\n")
        self.ser.flush()
        line = self.ser.readline().strip()
        print("reset", line)
        self.ser.write(b"RS\r\n")
        self.ser.flush()
        self.ser.write(b"S_TRUE\r\n")
        self.ser.flush()
    def get_data(self, length):
        data_one = []
        data_two = []

        self.ser.write(b"ShowData\r\n")
        self.ser.flush()
        print("getting data")
        for _ in range(length):
            line = self.ser.readline().strip()
            data_one.append(int(line))
            line = self.ser.readline().strip()
            data_two.append(int(line))
        print("finish")
        self.ser.write(b"STOP\r\n")
        self.ser.flush()
        line = self.ser.readline().strip()
        print(line)
        self.ser.write(b"RS\r\n")
        self.ser.flush()
        self.ser.close()

        return data_one, data_two

def save_data(data_one, data_two, image_size, group, _Data_counter):
    name_one = f"{image_size}_{group}_one_data_{_Data_counter}.csv"
    name_two = f"{image_size}_{group}_two_data_{_Data_counter}.csv"
    str_data = str(data_one)
    str_data = str_data[1:]
    str_data = str_data[:len(str_data) - 1]
    str_data_two = str(data_two)
    str_data_two = str_data_two[1:]
    str_data_two = str_data_two[:len(str_data_two) - 1]

    with open(name_one, 'w') as file_one:
        file_one.write(str_data)
    file_one.close()
    with open(name_two, 'w') as file_two:
        file_two.write(str_data_two)
    file_two.close()
