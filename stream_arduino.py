import serial
import time
import numpy as np
class StreamArduino:
    def __init__(self, port='COM11', baudrate=115200,timeout=100):
        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
    def start(self):
        self.arduino.write(bytes("START", 'utf-8'))
        print("pycharm call start")

    def get_data(self, data_length=4096, iter=1):
        data_one = []
        data_two = []
        print("collecting data")
        for i in range(data_length*iter):
            data_one.append(self.arduino.readline().strip())
            data_two.append(self.arduino.readline().strip())
        print("endloop")
        self.arduino.write(bytes("STOP", 'utf-8'))
        data_one = str(np.array(data_one).astype(int).tolist())
        data_two = str(np.array(data_two).astype(int).tolist())
        data_one = data_one[1:len(data_one)-1]
        data_two = data_two[1:len(data_two)-1]
        return data_one, data_two
    def close(self):
        self.arduino.close()
def save_data(data_one, data_two, image_size, group, _Data_counter):
    name_one = f"{image_size}_{group}_one_data_{_Data_counter}.csv"
    name_two = f"{image_size}_{group}_two_data_{_Data_counter}.csv"

    with open(name_one, 'w') as file_one:
        file_one.write(data_one)
    file_one.close()
    with open(name_two, 'w') as file_two:
        file_two.write(data_two)
    file_two.close()
"""arduino = serial.Serial(port='COM11',   baudrate=115200, timeout=.1)
def save_data(data_one, data_two, image_size, group, _Data_counter):
    name_one = f"{image_size}_{group}_one_data_{_Data_counter}.csv"
    name_two = f"{image_size}_{group}_two_data_{_Data_counter}.csv"

    with open(name_one, 'w') as file_one:
        file_one.write(data_one)
    file_one.close()
    with open(name_two, 'w') as file_two:
        file_two.write(data_two)
    file_two.close()

def write_read(x):
    arduino.write(bytes(x,   'utf-8'))
    time.sleep(1)
    data = arduino.readline()
    return   data

def send_command(x):
    arduino.write(bytes(x,   'utf-8'))
def read():
    i = 0
    while i<1000:
        data = arduino.readline()
        print(data)
        time.sleep(0.3)
        i+=1"""
"""while True:
    num = input("Enter a number: ")
    send_command(num)
    read()"""
