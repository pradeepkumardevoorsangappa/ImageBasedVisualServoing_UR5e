import serial
import time

# Configure the serial port. Adjust the port and baud rate as needed.
ser = serial.Serial('COMX', 9600)  # Replace 'COMX' with the appropriate COM port
time.sleep(2)  # Allow time for the Arduino to reset after establishing serial connection

def send_command(command):
    ser.write(command.encode())

def main():
    while True:
        command = input("Enter command (w, s, a, d, f, z, k, l): ")

        if command in ['w', 's', 'a', 'd', 'f', 'z', 'k', 'l']:
            send_command(command)
        else:
            print("Invalid command. Please try again.")

        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        ser.close()
