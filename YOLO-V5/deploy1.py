import rtde_receive
import Pid_test_XYZ as action
import serial
import time
ser = serial.Serial('COM9', 9600)  # Replace 'COMX' with the appropriate COM port
time.sleep(2)  # Allow time for the Arduino to reset after establishing serial connection
# Initialize RTDE Control and Receive interfaces
def send_command(command):
    ser.write(command.encode())

def main():
    command = 'w'#input("Enter command (w, s, a, d, f, z, k, l): ")
    if command in ['w', 's', 'a', 'd', 'f', 'z', 'k', 'l']:
        send_command(command)
    else:
        print("Invalid command. Please try again.")
    time.sleep(0.1)
main()


rtde_r1 = rtde_receive.RTDEReceiveInterface("169.254.37.99")
t1=time.time()
while True:
    t2=time.time()
    if t2-t1 >= 300:
        break
    #######Turn di_2 OFF for stopping
    io_status=rtde_r1.getActualDigitalInputBits()
    if io_status == 4:
        print("Move Robot")
        rtde_r1.disconnect()
        action.start()
        rtde_r1.reconnect()
        retur = 'z'#input("Enter command (w, s, a, d, f, z, k, l): ")
        if retur in ['w', 's', 'a', 'd', 'f', 'z', 'k', 'l']:
            send_command(retur)
    else:
        print('Waiting for starting action')
rtde_r1.disconnect()

# Configure the serial port. Adjust the port and baud rate as needed.


