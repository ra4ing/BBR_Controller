import numpy as np
from controller import Robot
from pid import PID


class Controller:

    def __init__(self, robot):
        """
        Initialize the Controller with a robot instance.

        Args:
            robot: An instance of the Robot class, representing the robot to be controlled.
        """
        self.robot = robot
        self.velocity_right = None  # Left wheel speed
        self.velocity_left = None  # Right wheel speed

        self.__init_parameters()  # Initialize parameters
        self.__enable_camera()  # Enable the robot's camera
        self.__enable_motors()  # Enable the robot's motors
        self.__enable_distance_sensors()  # Enable distance sensors
        self.__enable_light_sensors()  # Enable light sensors
        self.__enable_ground_sensors()  # Enable ground sensors

    def __init_parameters(self):
        """
        Initialize parameters for the controller.
        """
        self.time_step = 32  # ms
        self.max_speed = 6.28  # m/s

        self.forward_threshold = 0.5  # Make pid more smooth
        self.state = 0  # 0: follow_line  1: on_line  2: out_of_line  3: avoid_obstacle  4: reach_goal
        self.choose_path = 0  # -1: turn_right    1: turn_left

        self.pid = PID(1.0, 0.07, 0.02)  # Init PID
        self.time_count = 0  # Count time spent
        self.finish_obstacles = False  # Used to control the car back on the line

    def __enable_camera(self):
        """
        Enable and configure the robot's camera.
        Reuse from Official documents
        """
        # Enable and set up the camera
        self.camera = self.robot.getDevice('camera')  # Get the camera device
        self.camera.enable(self.time_step)  # Enable the camera with the defined time step
        self.width = self.camera.getWidth()  # Get camera width
        self.height = self.camera.getHeight()  # Get camera height
        self.width_check = int(self.width / 2)  # Midpoint of width for processing
        self.height_check = int(self.height / 2)  # Midpoint of height for processing

    def __enable_motors(self):
        """
        Enable and configure the robot's motors.
        Reuse from lab4 example
        """
        self.left_motor = self.robot.getDevice('left wheel motor')  # Get left wheel motor
        self.right_motor = self.robot.getDevice('right wheel motor')  # Get right wheel motor
        self.left_motor.setPosition(float('inf'))  # Set position to infinity for continuous rotation
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)  # Initialize velocity to 0
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0  # Initialize left velocity
        self.velocity_right = 0  # Initialize right velocity

    def __enable_distance_sensors(self):
        """
        Enable and configure the distance sensors of the robot.
        Reuse from lab4 example
        """
        self.distance_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)  # Sensor names (ps0, ps1, ..., ps7)
            sensor = self.robot.getDevice(sensor_name)
            sensor.enable(self.time_step)  # Enable each sensor
            self.distance_sensors.append(sensor)

    def __enable_light_sensors(self):
        """
        Enable and configure the light sensors of the robot.
        Reuse from lab4 example
        """
        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ls' + str(i)  # Sensor names (ls0, ls1, ..., ls7)
            sensor = self.robot.getDevice(sensor_name)
            sensor.enable(self.time_step)  # Enable each sensor
            self.light_sensors.append(sensor)

    def __enable_ground_sensors(self):
        """
        Enable and configure the ground sensors of the robot.
        Reuse from lab4 example
        """
        self.left_ir = self.robot.getDevice('gs0')  # Left ground sensor
        self.center_ir = self.robot.getDevice('gs1')  # Center ground sensor
        self.right_ir = self.robot.getDevice('gs2')  # Right ground sensor
        self.left_ir.enable(self.time_step)  # Enable sensors
        self.center_ir.enable(self.time_step)
        self.right_ir.enable(self.time_step)

    @staticmethod
    def adjust_value(val, min_val, max_val):
        """
        Clamp a value between a minimum and a maximum.

        Args:
            val (float): The value to be clamped.
            min_val (float): The minimum allowed value.
            max_val (float): The maximum allowed value.

        Returns:
            float: The clamped value.
        """
        return max(min_val, min(val, max_val))

    @staticmethod
    def map_range(val, from_min, from_max, to_min, to_max):
        """
        Maps a value from one range to another.

        Args:
            val (float): The value to be mapped.
            from_min (float): The minimum of the initial range.
            from_max (float): The maximum of the initial range.
            to_min (float): The minimum of the target range.
            to_max (float): The maximum of the target range.

        Returns:
            float: The value mapped to the new range.
        """
        from_range = from_max - from_min
        to_range = to_max - to_min
        scale = (val - from_min) / from_range
        return to_min + (scale * to_range)

    def __read_light_sensors(self):
        """
        Read and process data from light sensors to determine path choice.
        """
        if self.choose_path != 0:
            return

        min_ls = 0
        max_ls = 4300

        # Reuse from lab4 example
        lights = []
        for i in range(8):
            temp = self.light_sensors[i].getValue()     # Get sensor value
            temp = self.adjust_value(temp, min_ls, max_ls)  # Adjust value within range
            lights.append(temp)
        # End lab4 example

        # Determine path based on light sensor readings
        if min(lights) < 500:
            self.choose_path = 1    # Choose left path
        elif (self.time_count / 1000.0) > 8.0:
            self.choose_path = -1    # Choose right path after 8 seconds

    @staticmethod
    def normalize_value(val, min_val, max_val):
        """
        Normalize a value to a range between 0 and 1.

        Args:
            val (float): The value to normalize.
            min_val (float): The minimum value of the range.
            max_val (float): The maximum value of the range.

        Returns:
            float: The normalized value.
        """
        return (val - min_val) / (max_val - min_val)

    def __read_ground_sensors(self):
        """
        Read and process data from ground sensors to determine robot's state.
        """

        # Reuse from lab4 example
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
        # End lab4 example

        # Determine robot's state based on ground sensor readings
        if left > 500 and center > 500 and right > 500 and self.choose_path:  # not find line
            self.state = 2
        elif left < 500 and center < 500 and right < 500 and self.choose_path:  # all in line
            self.finish_obstacles = True
            self.state = 1
        else:  # follow line with PID
            self.state = 0
            self.__set_pid(left, center, right)

    def __set_pid(self, left, center, right):
        """
        Calculate PID adjustment based on sensor readings.

        Args:
            left: Reading from the left ground sensor.
            center: Reading from the center ground sensor.
            right: Reading from the right ground sensor.
        """
        self.pid_adjust = self.pid.cal_follow_line_adjust(left, center, right, self.time_step)
        self.pid_adjust = self.adjust_value(self.pid_adjust, -self.pid.scale,
                                            self.pid.scale)
        self.pid_adjust = self.map_range(self.pid_adjust, -self.pid.scale, self.pid.scale,
                                         -4.5, 4.5)

    def __read_distance_sensors(self):
        """
        Read and process data from distance sensors for obstacle avoidance.
        """

        # Reuse from lab4 example
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
        # End lab4 example

        # Return if finished avoiding obstacles
        if self.finish_obstacles:
            self.finish_obstacles = False
            return

        self.dis_adjust = 0
        if np.max(distances[0:3]) > avoid_distance or np.max(distances[5:8]) > avoid_distance:
            self.state = 3

            # Adjust direction based on obstacle location
            if self.choose_path == 1:   # left path
                self.dis_adjust = -avoid_speed
                if max(distances[0:3]) > avoid_distance:
                    self.dis_adjust = avoid_speed
            elif self.choose_path == -1:        # right path
                self.dis_adjust = avoid_speed
                if max(distances[5:8]) > avoid_distance:
                    self.dis_adjust = -avoid_speed

    def __read_camera(self):
        """
        Checks camera images at regular intervals (every 1600 ms). If the central pixel of the
        camera image meets specific color criteria (red component > 100, green < 100, blue < 100),
        the method counts pixels in the entire image that meet these criteria. If the count exceeds
        200, it changes the state to indicate a goal has been reached (state 4).
        """
        # Perform check every 1600 ms
        if self.time_count % 1600 != 0:
            return

        image = self.camera.getImageArray()
        # Return if no image data
        if not image:
            return

        # Get central pixel data
        check = image[self.width_check][self.height_check]
        if check[0] > 100 > check[2] and check[1] < 100:
            # display the components of each pixel
            cnt = 0
            for x in range(0, self.camera.getWidth()):
                for y in range(0, self.camera.getHeight()):
                    # Count pixels meeting specific color criteria
                    red = image[x][y][0]
                    green = image[x][y][1]
                    blue = image[x][y][2]
                    if red > 100 > blue and green < 100:
                        cnt += 1
                        if cnt >= 200:
                            self.state = 4  # Change state to goal reached
                            return

    def read_data(self):
        """
        Reads data from various sensors of the robot, including light sensors, ground sensors,
        distance sensors, and the camera.
        """
        self.__read_light_sensors()  # Read data from light sensors
        self.__read_ground_sensors()  # Read data from ground sensors
        self.__read_distance_sensors()  # Read data from distance sensors
        self.__read_camera()  # Read data from camera

    def follow_line(self):
        """
        Adjusts the robot's velocity for line following based on the PID adjustment value.
        The robot moves forward if the PID adjustment is small. Otherwise, it adjusts
        the left or right velocity to steer accordingly.
        """
        if -self.forward_threshold < self.pid_adjust < self.forward_threshold:  # # If PID adjust is small, go forward
            self.velocity_left = self.max_speed
            self.velocity_right = self.max_speed
        elif self.pid_adjust > 0:  # left
            self.velocity_left = self.max_speed - self.pid_adjust
            self.velocity_right = self.max_speed
        else:  # right
            self.velocity_right = self.max_speed + self.pid_adjust
            self.velocity_left = self.max_speed

    def avoid_obstacles(self):
        """
        Adjusts the robot's velocity for obstacle avoidance based on the distance adjustment value.
        The robot turns left if the distance adjustment is positive and turns right if it is negative.
        """
        if self.dis_adjust > 0:     # Turn left if dis_adjust is positive
            self.velocity_left = self.max_speed - self.dis_adjust
            self.velocity_right = self.max_speed
        else:   # Turn right if dis_adjust is negative
            self.velocity_right = self.max_speed + self.dis_adjust
            self.velocity_left = self.max_speed

    def turn_left(self):
        """
        Sets the robot's velocities to turn left.
        """
        self.velocity_left = 0.5
        self.velocity_right = self.max_speed

    def turn_right(self):
        """
        Sets the robot's velocities to turn right.
        """
        self.velocity_left = self.max_speed
        self.velocity_right = 0.5

    def take_move(self):
        """
        Determines the robot's next action based on its current state.
        """
        if self.state == 0:
            self.follow_line()

        """ 
            Help to avoid_obstacles:
            if choose right path: Turn right if all the ground sensors are online, and left if none are online
            if choose left path: Turn left if all the ground sensors are online, and right if none are online
        """
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
        """
        Stops the robot by setting the velocities of both motors to zero.
        """
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def run_robot(self):
        """
        Main loop for running the robot. It repeatedly reads sensor data, decides on
        movement actions, and updates motor velocities accordingly, until the robot
        reaches its goal or the simulation ends.
        """
        while self.robot.step(self.time_step) != -1:
            self.read_data()
            self.take_move()

            # Increment time counter
            self.time_count += self.time_step
            if self.state == 4:
                break
        print(self.time_count / 1000)
        self.stop()


if __name__ == "__main__":
    robot = Robot()
    controller = Controller(robot)
    controller.run_robot()
