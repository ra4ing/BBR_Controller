import numpy as np

from controller import Robot
from datetime import datetime
from PID import PID


class Controller:
    def __init__(self, robot):

        self.robot = robot
        self.follow_line_pid = PID(1.0, 0.07, 0.02)

        self.velocity_left = None
        self.velocity_right = None
        self.pid_adjust = None
        self.dis_adjust = None
        self.state = None
        self.finish_obstacles = None
        self.choose_path = None

        self.__init_parameters()  # Robot Parameters
        self.__enable_motors()  # Enable Motors
        self.__enable_distance_sensors()  # Enable Distance Sensors
        self.__enable_light_sensors()  # Enable Light Sensors
        self.__enable_proximity_sensors()  # Enable Light Sensors
        self.__enable_ground_sensors()  # Enable Ground Sensors

    def __init_parameters(self):
        self.time_step = 16  # ms
        self.max_speed = 6.28  # m/s

        self.forward_threshold = 0.6
        self.state = 0
        self.choose_path = -1

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
    def __enable_proximity_sensors(self):
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)

    def __enable_ground_sensors(self):
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)

    @staticmethod
    def __adjust_value(val, min_val, max_val):
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
            temp = self.__adjust_value(temp, min_ls, max_ls)
            temp = lights.append(temp)
            # print("Light Sensors - Index: {}  Value: {}".format(i,self.light_sensors[i].getValue()))

        if min(lights) < 500:
            self.choose_path = 1

    def __read_ground_sensors(self):
        min_gs = 0
        max_gs = 1000

        # Read Ground Sensors
        left = self.left_ir.getValue()
        center = self.center_ir.getValue()
        right = self.right_ir.getValue()

        # Adjust Values
        left = self.__adjust_value(left, min_gs, max_gs)
        center = self.__adjust_value(center, min_gs, max_gs)
        right = self.__adjust_value(right, min_gs, max_gs)
        # print("Ground Sensors \n    left {} center {} right {}".format(left, center, right))

        # choose path
        if left > 500 and center > 500 and right > 500:     # not find line
            self.state = -1
        elif left < 500 and center < 500 and right < 500:   # all in line
            self.finish_obstacles = True
            self.state = 1
        else:  # follow line with PID
            self.state = 0
            self.pid_adjust = self.follow_line_pid.cal_follow_line_adjust(left, center, right, self.time_step)
            self.pid_adjust = self.__adjust_value(self.pid_adjust, -self.follow_line_pid.scale,
                                                  self.follow_line_pid.scale)
            self.pid_adjust = self.map_range(self.pid_adjust, -self.follow_line_pid.scale, self.follow_line_pid.scale,
                                             -4.5, 4.5)



    def __read_distance_sensors(self):
        if self.finish_obstacles:
            self.finish_obstacles = False
            return

        min_ds = 0
        max_ds = 2400

        # Read Distance Sensors
        distances = []
        for sensor in self.proximity_sensors:
            temp = sensor.getValue()  # Get value
            temp = self.__adjust_value(temp, min_ds, max_ds)  # Adjust Values
            distances.append(temp)  # Save Data
            # print("Distance Sensors Value: {}".format(sensor.getValue()))

        if np.max(distances[0:3]) > 80 or np.max(distances[5:8]) > 80:
            self.state = 4

            if self.choose_path == 1:

                self.dis_adjust = -5
                if max(distances[0:3]) > 130:
                    self.dis_adjust = self.max_speed
            else:
                self.dis_adjust = 4.5
                if max(distances[5:8]) > 130:
                    self.dis_adjust = -self.max_speed


    def __read_data(self):
        self.__read_light_sensors()
        self.__read_ground_sensors()
        self.__read_distance_sensors()

    def __follow_line(self):
        if -self.forward_threshold < self.pid_adjust < self.forward_threshold:  # forward
            self.velocity_left = self.max_speed
            self.velocity_right = self.max_speed
        elif self.pid_adjust > 0:  # left
            self.velocity_left = self.max_speed - self.pid_adjust
            self.velocity_right = self.max_speed
        else:  # right
            self.velocity_right = self.max_speed + self.pid_adjust
            self.velocity_left = self.max_speed

    def __avoid_obstacles(self):
        if self.dis_adjust > 0:
            self.velocity_left = self.max_speed - self.dis_adjust
            self.velocity_right = self.max_speed
        else:
            self.velocity_right = self.max_speed + self.dis_adjust
            self.velocity_left = self.max_speed

    def __turn_left(self):
        self.velocity_left = 0.5
        self.velocity_right = self.max_speed

    def __turn_right(self):
        self.velocity_left = self.max_speed
        self.velocity_right = 0.5

    def __take_move(self):
        print(self.state)
        if self.state == 0:
            self.__follow_line()
        elif self.state == 1 and self.choose_path == 1:
            self.__turn_left()
        elif self.state == 1 and self.choose_path == -1:
            self.__turn_right()
        elif self.state == 4:
            self.__avoid_obstacles()
        else:
            if self.choose_path == 1:
                self.__turn_right()
            else:
                self.__turn_left()

        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)

    def run_robot(self):
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            self.__read_data()
            self.__take_move()


if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot()
