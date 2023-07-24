import rtde_control
import rtde_receive

rtde_c = rtde_control.RTDEControlInterface("172.17.0.2")
rtde_r = rtde_receive.RTDEReceiveInterface("172.17.0.2")

# Move to initial joint position with a regular moveJ
rtde_c.moveJ([0.0,0.0,0.0,0.0,0.0,0.0])
# Execute 500Hz control loop for 2 seconds, each cycle is 2ms
for i in range(1000):
    t_start = rtde_c.initPeriod()
    rtde_c.speedL(xd=[0.0,0.0,0.0,0.1,0.0,0.0])
    rtde_c.waitPeriod(t_start)
rtde_c.speedStop()
rtde_c.stopScript()