import time

import numpy as np
from controller import Robot
from pid import PID


class Controller:
    def __init__(self, robot):
        self.robot = robot
        self.velocity_right = None
        self.velocity_left = None

        self.__init_parameters()  # Robot Parameters
        self.__enable_camera()  # Enable Camera
        self.__enable_motors()  # Enable Motors
        self.__enable_distance_sensors()  # Enable Distance Sensors
        self.__enable_light_sensors()  # Enable Light Sensors
        self.__enable_ground_sensors()  # Enable Ground Sensors

    def __init_parameters(self):
        self.time_step = 32  # ms
        self.max_speed = 6.28  # m/s

        self.forward_threshold = 0.5
        self.state = 0
        self.choose_path = 0

        self.pid = PID(1.0, 0.07, 0.02)
        self.time_count = 0
        self.finish_obstacles = False

    def __enable_camera(self):
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.time_step)
        self.width = self.camera.getWidth()
        self.height = self.camera.getHeight()
        self.width_check = int(self.width / 2)
        self.height_check = int(self.height / 2)

    def __enable_motors(self):
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0

    def __enable_distance_sensors(self):
        self.distance_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.distance_sensors.append(self.robot.getDevice(sensor_name))
            self.distance_sensors[i].enable(self.time_step)

    def __enable_light_sensors(self):
        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ls' + str(i)
            self.light_sensors.append(self.robot.getDevice(sensor_name))
            self.light_sensors[i].enable(self.time_step)

    def __enable_ground_sensors(self):
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)

    @staticmethod
    def adjust_value(val, min_val, max_val):
        val = max(val, min_val)
        val = min(val, max_val)
        return val

    @staticmethod
    def map_range(val, from_min, from_max, to_min, to_max):
        from_range = from_max - from_min
        to_range = to_max - to_min
        scale = (val - from_min) / from_range

        return to_min + (scale * to_range)

    def __read_light_sensors(self):
        min_ls = 0
        max_ls = 4300

        lights = []
        for i in range(8):
            temp = self.light_sensors[i].getValue()
            # Adjust Values
            temp = self.adjust_value(temp, min_ls, max_ls)
            lights.append(temp)

        if self.choose_path != 0:
            return

        if min(lights) < 500:
            self.choose_path = 1
        elif (self.time_count / 1000.0) > 8.0:
            self.choose_path = -1

    def __read_ground_sensors(self):
        min_gs = 0
        max_gs = 1000

        # Read Ground Sensors
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()

        # Adjust Values
        left = self.adjust_value(left, min_gs, max_gs)
        center = self.adjust_value(center, min_gs, max_gs)
        right = self.adjust_value(right, min_gs, max_gs)

        if left > 500 and center > 500 and right > 500 and self.choose_path:  # not find line
            self.state = 2
        elif left < 500 and center < 500 and right < 500 and self.choose_path:  # all in line
            self.finish_obstacles = True
            self.state = 1
        else:  # follow line with PID
            self.state = 0
            self.__set_pid(left, center, right)

    def __set_pid(self, left, center, right):
        self.pid_adjust = self.pid.cal_follow_line_adjust(left, center, right, self.time_step)
        self.pid_adjust = self.adjust_value(self.pid_adjust, -self.pid.scale,
                                            self.pid.scale)
        self.pid_adjust = self.map_range(self.pid_adjust, -self.pid.scale, self.pid.scale,
                                         -4.5, 4.5)

    def __read_distance_sensors(self):
        if self.finish_obstacles:
            self.finish_obstacles = False
            return

        min_ds = 0
        max_ds = 2400
        avoid_speed = 5
        avoid_distance = 80

        # Read Distance Sensors
        distances = []
        for sensor in self.distance_sensors:
            temp = sensor.getValue()  # Get value
            temp = self.adjust_value(temp, min_ds, max_ds)  # Adjust Values
            distances.append(temp)  # Save Data

        self.dis_adjust = 0
        if np.max(distances[0:3]) > avoid_distance or np.max(distances[5:8]) > avoid_distance:
            self.state = 3

            if self.choose_path == 1:
                self.dis_adjust = -avoid_speed
                if max(distances[0:3]) > avoid_distance:
                    self.dis_adjust = avoid_speed
            elif self.choose_path == -1:
                self.dis_adjust = avoid_speed
                if max(distances[5:8]) > avoid_distance:
                    self.dis_adjust = -avoid_speed

    def __read_camera(self):
        # self.camera.getImage()

        if self.time_count % 1600 != 0:
            return

        image = self.camera.getImageArray()
        if not image:
            return

        check = image[self.width_check][self.height_check]
        if check[0] > 100 > check[2] and check[1] < 100:
            # display the components of each pixel
            cnt = 0
            for x in range(0, self.camera.getWidth()):
                for y in range(0, self.camera.getHeight()):

                    red = image[x][y][0]
                    green = image[x][y][1]
                    blue = image[x][y][2]
                    if red > 100 > blue and green < 100:
                        cnt += 1
                        if cnt >= 200:
                            self.state = 4
                            return
        # print(cnt)

    def read_data(self):
        self.__read_light_sensors()
        self.__read_ground_sensors()
        self.__read_distance_sensors()
        self.__read_camera()

    def follow_line(self):
        if -self.forward_threshold < self.pid_adjust < self.forward_threshold:  # forward
            self.velocity_left = self.max_speed
            self.velocity_right = self.max_speed
        elif self.pid_adjust > 0:  # left
            self.velocity_left = self.max_speed - self.pid_adjust
            self.velocity_right = self.max_speed
        else:  # right
            self.velocity_right = self.max_speed + self.pid_adjust
            self.velocity_left = self.max_speed

    def avoid_obstacles(self):
        if self.dis_adjust > 0:
            self.velocity_left = self.max_speed - self.dis_adjust
            self.velocity_right = self.max_speed
        else:
            self.velocity_right = self.max_speed + self.dis_adjust
            self.velocity_left = self.max_speed

    def turn_left(self):
        self.velocity_left = 0.5
        self.velocity_right = self.max_speed

    def turn_right(self):
        self.velocity_left = self.max_speed
        self.velocity_right = 0.5

    def take_move(self):
        if self.state == 0:
            self.follow_line()
        if self.state == 1:
            self.turn_left() if self.choose_path == 1 else self.turn_right()
        elif self.state == 2:
            self.turn_right() if self.choose_path == 1 else self.turn_left()
        elif self.state == 3:
            self.avoid_obstacles()
        elif self.state == 4:
            print("finish !!!\n")
            return

        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)

    def stop(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def run_robot(self):
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            self.read_data()
            self.take_move()
            self.time_count += self.time_step
            if self.state == 4:
                break
        print(self.time_count)
        self.stop()


if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot()
