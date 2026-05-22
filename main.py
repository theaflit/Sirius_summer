from pydexarm.pydexarm import *
from send_esp import send_command_esp
import time

dexarm = Dexarm(port="/dev/ttyACM0")

dexarm.go_home()
send_command_esp('open')
dexarm.conveyor_belt_forward(1500)
time.sleep(8)
dexarm.conveyor_belt_stop()
dexarm.move_to(0, 300, -67)
time.sleep(2)
send_command_esp('close')
time.sleep(0.5)
dexarm.move_to(0, 300, 0)
print(dexarm.get_current_position())

dexarm.conveyor_belt_backward(1500)
time.sleep(4)
dexarm.conveyor_belt_stop()
dexarm.move_to(0, 300, -60)
time.sleep(2)

send_command_esp('open')
dexarm.move_to(0, 300, 0)