import time
import pygame
from pygame.locals import *
import queue
import math

"""
axis:
    0, 左方向摇杆-左右, 范围(-1, 1), cmd_vel_vy
    1, 左方向摇杆-前后, 范围(-1, 1), cmd_vel_vx
    2, 右方向摇杆-前后, 锁定
    3, 右方向摇杆-左右, 范围(-1, 1), cmd_vel_wz

buttons:
    0, A
    1, B
    2, X
    3, Y
    4, LB
    5, RB
"""

class JoyStickController:
    def __init__(self):
        self.joy_freq = 200

        pygame.init()
        pygame.joystick.init()
        
        joystick_count = pygame.joystick.get_count()
        
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        else:
            self.joystick == None
            print("No joystick detected.")
        
        self.cmd_vel_queue = queue.Queue(maxsize=5)
        self.button_press_queue = queue.Queue(maxsize=5)
        for _ in range(4):
            self.cmd_vel_queue.put(self._init_cmd_vel())
            self.button_press_queue.put(self._init_button_press())

    def _init_button_press(self):
        a = 0
        b = 0
        x = 0
        y = 0
        lb = 0
        rb = 0
        return {
            "A": a,
            "B": b,
            "X": x,
            "Y": y,
            "LB": lb,
            "RB": rb
        }

    def _init_cmd_vel(self):
        vx = 0.0
        vy = 0.0
        wz = 0.0
        return {
            "vx": vx,
            "vy": vy,
            "wz": wz
        }
    
    def update_button_press(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()

        # 获取按键值
        button_A = self.joystick.get_button(0)
        button_B = self.joystick.get_button(1)
        button_X = self.joystick.get_button(2)
        button_Y = self.joystick.get_button(3)
        button_LB = self.joystick.get_button(4)
        button_RB = self.joystick.get_button(5)

        new_button_press = {
            "A": button_A,
            "B": button_B,
            "X": button_X,
            "Y": button_Y,
            "LB": button_LB,
            "RB": button_RB
        }
        
        if self.button_press_queue.full():
            try:
                self.button_press_queue.get_nowait()  
            except queue.Empty:
                pass
        self.button_press_queue.put(new_button_press.copy())

    def update_cmd_vel(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                exit()
        
        # 获取摇杆原始值
        axis_1 = self.joystick.get_axis(1)
        axis_0 = self.joystick.get_axis(0)
        axis_3 = self.joystick.get_axis(3)

        # 摇杆速度映射
        vx = (0.5 * (1 - math.cos(math.pi * axis_1)) * (axis_1 >= 0.0) - \
              0.5 * (1 - math.cos(math.pi * axis_1)) * (axis_1 < 0.0)) * (-1.0)
        vy = (0.5 * (1 - math.cos(math.pi * axis_0)) * (axis_0 >= 0.0) - \
              0.5 * (1 - math.cos(math.pi * axis_0)) * (axis_0 < 0.0)) * (-1.0)
        wz = (0.5 * (1 - math.cos(math.pi * axis_3)) * (axis_3 >= 0.0) - \
              0.5 * (1 - math.cos(math.pi * axis_3)) * (axis_3 < 0.0)) * (-1.0)

        new_cmd_vel = {
            "vx": vx,
            "vy": vy,
            "wz": wz
        }
        
        if self.cmd_vel_queue.full():
            try:
                self.cmd_vel_queue.get_nowait()  
            except queue.Empty:
                pass
        self.cmd_vel_queue.put(new_cmd_vel.copy())

    def get_button_press(self):
        try:
            return self.button_press_queue.get_nowait() 
        except queue.Empty:
            return self._init_button_press()

    def get_cmd_vel(self):
        try:
            return self.cmd_vel_queue.get_nowait() 
        except queue.Empty:
            return self._init_cmd_vel()

    def run(self):
        while True:
            step_start_time = time.time()

            self.update_button_press()
            self.update_cmd_vel()
            button = self.get_button_press()
            cmd_vel = self.get_cmd_vel()
            print(button["RB"])

            step_elapsed_time = time.time() - step_start_time
            time.sleep(max(0, 1.0 / self.joy_freq - step_elapsed_time))


if __name__ == "__main__":
    joy = JoyStickController()
    joy.run()
