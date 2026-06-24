import threading
import logging
from queue import Queue
import time

from pydexarm.pydexarm import *
# from send_esp import send_command_esp
from detect.neyro_main import launch_yolo
# from dexarm_module import dexarm_worker 

# TODO: доделать библиотеку для контроллирования манипулятора 
# в зависимости от положения эклера

target_queue = Queue(maxsize=1)

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def dexarm_control():
    # dex = Dexarm("COM3")

    while True:
        distance, dx, dy = target_queue.get()

        logging.info(f"{distance} {dx} {dy}")

        # if abs(dx) > 10:
        #     dex.move_to(x=dx, y=dy, feedrate=5000)

if __name__ == "__main__":

    yolo_thread = threading.Thread(target=launch_yolo, 
                                   args=("best.pt", "cuda", True, target_queue, "test_eclair_video.mp4"))
    dexarm_thread = threading.Thread(target=dexarm_control)

    yolo_thread.start()
    dexarm_thread.start()

    yolo_thread.join()
    dexarm_thread.join()