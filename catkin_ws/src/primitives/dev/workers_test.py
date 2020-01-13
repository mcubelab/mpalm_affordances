from airobot import Robot
from multiprocessing import Process, Pipe
import time
import pybullet as p
from IPython import embed

def worker_yumi(child_conn):
    while True:
        # print("here!")
        try:
            if not child_conn.poll(0.0001):
                continue
            msg = child_conn.recv()
        except (EOFError, KeyboardInterrupt):
            break
        if msg == "RESET":
            yumi = Robot('yumi', pb=True, arm_cfg={'render': True, 'self_collision': False})
            # client_id = p.connect(p.DIRECT)
            print("\n\nfinished worker construction\n\n")
            continue
        if msg == "HOME":
            yumi.arm.go_home()
            continue
        if msg == "END":
            break
        print("before sleep!")
        time.sleep(0.01)
    print("breaking")
    child_conn.close()


parent1, child1 = Pipe()
parent2, child2 = Pipe()
p1 = Process(target=worker_yumi, args=(child1,))
p2 = Process(target=worker_yumi, args=(child2,))
p1.start()
p2.start()

parent1.send("RESET")
parent2.send("RESET")

print("started workers")
time.sleep(5.0)
embed()

parent1.send("END")
parent2.send("END")

print("ended workers")
print("Done!")
