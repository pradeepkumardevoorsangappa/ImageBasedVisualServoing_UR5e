import rtde_receive
import Pid_test_XYZ as action
import time
# Initialize RTDE Control and Receive interfaces
rtde_r1 = rtde_receive.RTDEReceiveInterface("169.254.37.99")
t1=time.time()
while True:
    t2=time.time()
    if t2-t1 >= 300:
        break
    io_status=rtde_r1.getActualDigitalInputBits()
    if io_status == 4:
        print("Move Robot")
        rtde_r1.disconnect()
        action.start()
        rtde_r1.reconnect()
    else:
        print('Waiting for starting action')
rtde_r1.disconnect()