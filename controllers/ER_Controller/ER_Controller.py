import numpy as np
from controller import Robot
from ga import GA
from trainer import Trainer


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
        self.time_count = None
        self.choose_path = None
        self.state = None
        self.right_count = None
        self.left_count = None
        self.train_module = None

        self.__init_parameters()  # Initialize parameters
        self.__init_trainer()  # Initialize MLP
        self.__enable_camera()  # Enable the robot's camera
        self.__enable_motors()  # Enable the robot's motors
        self.__enable_distance_sensors()  # Enable distance sensors
        self.__enable_light_sensors()  # Enable light sensors
        self.__enable_ground_sensors()  # Enable ground sensors

    def reset(self):
        self.stop()

        self.state = 0
        self.choose_path = 0
        self.time_count = 0

        self.trainer.reset()

    def __init_parameters(self):
        """
        Initialize parameters for the controller.
        """
        self.time_step = 32  # ms
        self.max_speed = 6.28  # m/s
        self.max_time = 90.0

        self.state = 0
        self.choose_path = 0
        self.time_count = 0
        self.left_count = 0
        self.right_count = 0

    def __init_trainer(self):
        self.trainer = Trainer(self.robot)

    def __enable_camera(self):
        """
        Enable and configure the robot's camera.
        Found in Official documents
        """
        self.camera = self.robot.getDevice('camera')  # Get the camera device
        self.camera.enable(self.time_step)  # Enable the camera with the defined time step
        self.width = self.camera.getWidth()  # Get camera width
        self.height = self.camera.getHeight()  # Get camera height
        self.width_check = int(self.width / 2)  # Midpoint of width for processing
        self.height_check = int(self.height / 2)  # Midpoint of height for processing

    def __enable_motors(self):
        """
        Enable and configure the robot's motors.
        Found in lab4 example
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
        Found in lab4 example
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
        Found in lab4 example
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
        Found in lab4 example
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
        self.trainer.inputs.append(self.choose_path)

        if self.choose_path != 0:
            return

        # Reuse from lab4 example
        min_ls = 0
        max_ls = 4300

        lights = []
        for i in range(8):
            temp = self.light_sensors[i].getValue()
            # Adjust Values
            temp = self.adjust_value(temp, min_ls, max_ls)
            lights.append(temp)
        # End lab4 example

        if min(lights) < 500:
            self.choose_path = -0.5
        elif (self.time_count / 1000.0) > 3.0:
            self.choose_path = 0.5

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

        left_normalized = self.normalize_value(left, min_gs, max_gs)
        center_normalized = self.normalize_value(center, min_gs, max_gs)
        right_normalized = self.normalize_value(right, min_gs, max_gs)

        self.trainer.inputs.append(left_normalized)
        self.trainer.inputs.append(center_normalized)
        self.trainer.inputs.append(right_normalized)
        # End lab4 example

    def __read_distance_sensors(self):
        """
        Read and process data from distance sensors for obstacle avoidance.
        """
        # Reuse from lab4 example
        min_ds = 0
        max_ds = 2400

        # Read Distance Sensors
        for sensor in self.distance_sensors:
            temp = sensor.getValue()  # Get value
            temp = self.adjust_value(temp, min_ds, max_ds)  # Adjust Values

            temp_normalized = self.normalize_value(temp, min_ds, max_ds)  # save value for evolutionary
            temp_normalized = temp_normalized * 5
            temp_normalized = self.adjust_value(temp_normalized, 0, 1)
            self.trainer.inputs.append(temp_normalized)
        # End lab4 example

    def __read_camera(self):
        """
        Checks camera images at regular intervals (every 1600 ms). If the central pixel of the
        camera image meets specific color criteria (red component > 100, green < 100, blue < 100),
        the method counts pixels in the entire image that meet this criteria. If the count exceeds
        200, it changes the state to indicate a goal has been reached (state 4).
        """
        if self.time_count % 1600 != 0:
            return

        image = self.camera.getImageArray()
        if not image:
            return

        check = image[self.width_check][self.height_check]
        if check[0] > 100 > check[2] and check[1] < 100:
            # display the components of each pixel
            cnt = 0
            for x in range(self.width):
                for y in range(self.height):

                    red = image[x][y][0]
                    green = image[x][y][1]
                    blue = image[x][y][2]
                    if red > 100 > blue and green < 100:
                        cnt += 1
                        if cnt >= 200:
                            self.state = 4
                            return

    def __read_data(self):
        """
        Reads data from various sensors of the robot, including light sensors, ground sensors,
        distance sensors, and the camera.
        """
        self.trainer.inputs = []
        self.__read_ground_sensors()
        self.__read_distance_sensors()
        self.__read_light_sensors()
        self.__read_camera()

    def take_move(self):
        """Executes a movement based on the trainer's output.

        This method calculates the velocity for both the left and right motors
        based on the trainer's output and updates the motor velocities accordingly.
        """
        output = self.trainer.get_output_and_cal_fitness()
        self.velocity_left = output[0] + 0.1
        self.velocity_right = output[1]

        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)
        # self.stop()

    def stop(self):
        """Stops the robot by setting its motor velocities to zero.

        This method also makes the robot step through the simulation for a time step,
        effectively halting its movement.
        """
        self.velocity_left = 0
        self.velocity_right = 0
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.robot.step(self.time_step)

    def __evaluate_genotype(self):
        """Evaluates the fitness of the current genotype.

        This method updates the MLP (Multi-Layer Perceptron) of the trainer,
        runs trials by resetting the environment and robot, and calculates
        the average fitness from these trials.

        Returns:
            float: The average fitness calculated over the trials.
        """
        self.trainer.update_mlp()

        fitness_per_trial = []
        number_interaction_loops = 1
        for _ in range(number_interaction_loops):
            # Running trials for 'right' and 'left' scenarios
            for direction in ["right", "left"]:
                self.trainer.reset_environment(direction)
                self.trainer.wait_reset_complete()
                self.reset()
                fitness = self.run_robot()
                fitness_per_trial.append(fitness)

        fitness = np.mean(fitness_per_trial)
        print("Fitness: {}".format(fitness))
        return fitness

    def run_optimization(self):
        """Runs the genetic algorithm optimization process.

        This method initializes the population, iterates over generations,
        evaluates genotypes, and performs genetic operations to evolve the population.
        The best and average fitness values are tracked and saved for each generation.
        """

        # Reuse from lab4 example
        populations = GA.create_random_population(self.trainer.num_weights)

        print(">>>Starting Evolution using GA optimization ...\n")

        # For each Generation
        for generation in range(GA.num_generations):

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Generation: {}".format(generation))
            current_population = []
            # Select each Genotype or Individual

            for population in range(GA.num_population):
                # Evaluating each individual in the population
                print("-------------------")
                print("Population: {}".format(population))
                self.trainer.genotype = populations[population]
                fitness = self.__evaluate_genotype()
                current_population.append((self.trainer.genotype, float(fitness)))

            # Finding the best and average fitness in the current generation
            best = GA.get_best_genotype(current_population)
            average = GA.get_average_genotype(current_population)
            print("-------------------")
            print("Best: {}".format(best[1]))
            print("Average: {}".format(average))
            for idx in range(GA.num_elite):
                np.save("../module/Best{}.npy".format(idx), populations[idx])
            self.trainer.plt(generation, self.normalize_value(best[1], 0, 10), self.normalize_value(average, 0, 10))

            # Generate the new population_idx using genetic operators
            if generation < GA.num_generations - 1:
                populations = GA.population_reproduce(current_population)

        print("GA optimization terminated.\n")
        # End lab4 example

    def run_best(self):
        """Executes the best performing genotype in a demonstration run.

        This method loads the best genotype, resets the environment, and runs the robot
        to evaluate its performance in both 'right' and 'left' trials. The fitness
        and time taken for each trial are printed.
        """
        # for i in range(8, 20):
        print("++++++++++++++++++++++++++++++++++++++++++++++")
        # print("Best {}".format(i))
        # self.trainer.genotype = np.load("../pre_module/know_road_and_right.npy")
        # self.trainer.genotype = np.load("../module/Best{}.npy".format(i))
        self.trainer.genotype = np.load("../pre_module/reach_goal.npy")
        self.trainer.update_mlp()

        # trial: right
        self.trainer.reset_environment("right")
        self.trainer.wait_reset_complete()
        self.reset()
        fitness = self.run_robot()
        print("Fitness: {}".format(fitness))
        print(self.time_count / 1000)

        # trial: left
        self.trainer.reset_environment("left")
        self.trainer.wait_reset_complete()
        self.reset()
        fitness = self.run_robot()
        print("Fitness: {}".format(fitness))
        print(self.time_count / 1000)

        print("GA demo terminated.\n")

    def adjust_fitness(self, fitness):
        """Adjusts the fitness score based on time and reward conditions.

        This method modifies the fitness score based on the time taken and the counts
        of specific rewards. It applies various conditions to ensure the fitness score
        is representative of the robot's performance.

        Args:
            fitness (float): The initial fitness score to be adjusted.

        Returns:
            float: The adjusted fitness score.
        """
        times = self.time_count / self.time_step
        fitness /= times

        # reach goal should be after Complete some missions
        if self.trainer.gs_rewards_count > 0 and self.trainer.ds_rewards_count > 0:
            if self.state == 4:
                fitness += 1
                fitness -= self.time_count / 2000_00

        # achieved too much missions means that this gene didn't complete the missions as expected
        if self.trainer.gs_rewards_count > 8 or self.trainer.ds_rewards_count > 8:
            return 0

        # achieved too little missions
        if self.trainer.gs_rewards_count < 2 < self.trainer.ds_rewards_count:
            return 0

        return fitness

    def run_robot(self):
        """Runs the robot in the simulation and calculates its fitness.

         This method steps through the simulation, reading data and taking movements.
         It calculates fitness at each step and adjusts it based on various conditions.
         The method stops the robot when certain conditions are met.

         Returns:
             float: The final adjusted fitness score of the robot.
         """
        fitness = 0
        while self.robot.step(self.time_step) != -1:
            self.__read_data()
            self.take_move()
            fitness += self.trainer.cal_fitness_and_reward([self.velocity_left, self.velocity_right])

            self.time_count += self.time_step
            if self.state == 4:
                print(self.time_count / 1000)
                break
            elif (self.time_count / 1000) >= self.max_time:
                break
        # print(self.trainer.gs_rewards_count)
        # print(self.trainer.ds_rewards_count)

        fitness = self.adjust_fitness(fitness)
        self.stop()
        return fitness


if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)

    # wait for key pressed
    flag = False
    while controller.robot.step(controller.time_step) != -1 and not controller.trainer.wait_message:
        flag = controller.trainer.wait_for_message()

    if flag == "True":
        print("optimization")
        controller.run_optimization()
    else:
        print("run_best")
        controller.run_best()
