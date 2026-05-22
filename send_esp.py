import serial
import time

ser = serial.Serial(
    port="/dev/ttyUSB0",
    baudrate=115200,
    timeout=1
)

time.sleep(2)


def send_command_esp(text: str):

    ser.write((text + "\n").encode())
