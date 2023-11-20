import serial
import numpy as np
import sys

class StreamRaspberry:
    def __init__(self, PORT: str = 'COM3', BAUD_RATE: int = 128000, timeout: int = 3):
        self.ser = serial.Serial(PORT, BAUD_RATE, timeout=timeout)
        if self.ser.isOpen():
            print('Serial port opened:', self.ser.portstr)
    def start_acquisation(self):
        self.ser.write(b"S_TRUE\r\n")
    def get_data_one(self):
        self.ser.write(b"one_length")
        one_length = self.ser.readline()
        one_length = int(one_length[0:len(one_length)-2].decode("utf-8"))
        data_one = []
        for _ in range(one_length):
            line = self.ser.readline()
            line = int(line[0:len(line)-2].decode("utf-8"))
            data_one.append(line)
        return data_one
    def get_data_two(self):
        self.ser.write(b"two_length")
        two_length = self.ser.readline()
        two_length = int(two_length[0:len(two_length)-2].decode("utf-8"))
        data_two = []
        for _ in range(two_length):
            line = self.ser.readline()
            line = int(line[0:len(line)-2].decode("utf-8"))
            data_two.append(line)
        return data_two

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