import numpy as np
from controller import Robot
from ga import GA
from trainer import Trainer


class Controller:
    def __init__(self, robot):
        self.robot = robot

        self.velocity_left = None
        self.velocity_right = None
        self.last_genotype_str = ""

        self.__init_parameters()  # Robot Parameters
        self.__init_trainer()  # Initialize MLP
        self.__enable_camera()  # Enable Camera
        self.__enable_motors()  # Enable Motors
        self.__enable_distance_sensors()  # Enable Distance Sensors
        self.__enable_light_sensors()  # Enable Light Sensors
        self.__enable_proximity_sensors()  # Enable Light Sensors
        self.__enable_ground_sensors()  # Enable Ground Sensors

    def reset(self):
        self.stop()
        self.velocity_left = 0
        self.velocity_right = 0

        self.__init_parameters()  # Robot Parameters
        self.__enable_camera()  # Enable Camera
        self.__enable_motors()  # Enable Motors
        self.__enable_distance_sensors()  # Enable Distance Sensors
        self.__enable_light_sensors()  # Enable Light Sensors
        self.__enable_proximity_sensors()  # Enable Light Sensors
        self.__enable_ground_sensors()  # Enable Ground Sensors

    def __init_parameters(self):
        self.time_step = 32  # ms
        self.max_speed = 6.28  # m/s

        self.forward_threshold = 0.5
        self.state = 0
        self.choose_path = 0

        self.time_count = 0
        # self.finish_obstacles = False

    def __init_trainer(self):
        self.trainer = Trainer(self.robot, self.time_step)

    def __enable_camera(self):
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.time_step)

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
    def adjust_value(val, min_val, max_val):
        val = max(val, min_val)
        val = min(val, max_val)
        return val

    @staticmethod
    def normalize_value(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

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
            self.trainer.inputs.append(0)
            return

        if min(lights) < 500:
            self.choose_path = 1
        elif (self.time_count / 1000.0) > 8.0:
            self.choose_path = -1

        self.trainer.inputs.append(self.choose_path)

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

        # save value for evolutionary
        left_normalized = self.normalize_value(left, min_gs, max_gs)
        center_normalized = self.normalize_value(center, min_gs, max_gs)
        right_normalized = self.normalize_value(right, min_gs, max_gs)
        self.trainer.inputs.append(left_normalized)
        self.trainer.inputs.append(center_normalized)
        self.trainer.inputs.append(right_normalized)

        if left > 500 and center > 500 and right > 500 and self.choose_path:  # not find line
            self.state = 2
        elif left < 500 and center < 500 and right < 500 and self.choose_path:  # all in line
            # self.finish_obstacles = True
            self.state = 1
        else:  # follow line with PID
            self.state = 0

    def __read_distance_sensors(self):
        # if self.finish_obstacles:
        #     self.finish_obstacles = False
        #     return

        min_ds = 0
        max_ds = 2400
        avoid_speed = 5
        avoid_distance = 80

        # Read Distance Sensors
        distances = []
        for sensor in self.proximity_sensors:
            temp = sensor.getValue()  # Get value
            temp = self.adjust_value(temp, min_ds, max_ds)  # Adjust Values

            temp_normalized = self.normalize_value(temp, min_ds, max_ds)  # save value for evolutionary
            self.trainer.inputs.append(temp_normalized)

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
        self.camera.getImage()

        image = self.camera.getImageArray()
        if not image:
            return
        # display the components of each pixel
        cnt = 0
        for x in range(0, self.camera.getWidth()):
            for y in range(0, self.camera.getHeight()):

                red = image[x][y][0]
                green = image[x][y][1]
                blue = image[x][y][2]
                if red > 100 and green < 20 and blue < 20:
                    cnt += 1

        if cnt >= 1800:
            self.state = 4

    def __read_data(self):
        self.trainer.inputs = []
        self.__read_light_sensors()
        self.__read_ground_sensors()
        self.__read_distance_sensors()
        self.__read_camera()

    def take_move(self):
        # print(self.state)

        output = self.trainer.get_output_and_cal_fitness()
        # self.velocity_left = 0.3
        # self.velocity_right = 0.3
        self.velocity_left = min((output[0] + 0.5), self.max_speed)
        self.velocity_right = min((output[1] + 0.5), self.max_speed)

        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)

    def stop(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def run_optimization(self):
        self.trainer.population = GA.create_random_population(self.trainer.num_weights)

        print(">>>Starting Evolution using GA optimization ...\n")

        # For each Generation
        for generation in range(GA.num_generations):
            print("Generation: {}".format(generation))
            current_population = []
            # Select each Genotype or Individual

            for population in range(GA.num_population):

                self.trainer.genotype = self.trainer.population[population]
                fitness = self.__evaluate_genotype(self.trainer.genotype, generation)
                # Save its fitness value
                current_population.append((self.trainer.genotype, float(fitness)))
                # print(current_population)

            best = GA.get_best_genotype(current_population)
            average = GA.get_average_genotype(current_population)
            np.save("Best.npy", best[0])
            self.trainer.plt(generation, best[1], average)

            # Generate the new population_idx using genetic operators
            if generation < GA.num_generations - 1:
                self.trainer.population = GA.population_reproduce(current_population, GA.num_elite)

        # print("All Genotypes: {}".format(self.genotypes))
        print("GA optimization terminated.\n")

    def run_robot(self):

        self.reset()
        # print("reset_environment")
        while self.robot.step(self.time_step) != -1:
            self.__read_data()
            self.take_move()

            self.time_count += self.time_step
            if self.state == 4 or (self.time_count / 1000) > 60.0:
                break

        print(self.time_count)

    def __evaluate_genotype(self, genotype, generation):
        self.trainer.new_genotype_flag = False
        if not np.array_equal(str(genotype), self.last_genotype_str):
            self.trainer.new_genotype_flag = True

        self.last_genotype_str = str(genotype)

        number_interaction_loops = 3
        fitness_per_trial = []
        current_interaction = 0

        while current_interaction < number_interaction_loops:

            # trial: right
            print("right trial")
            self.trainer.reset_for_right()
            self.run_robot()
            fitness = self.trainer.cal_right_fitness_with_reward((self.state == 4), self.time_count, self.velocity_left, self.velocity_right)
            print("Fitness: {}".format(fitness))
            fitness_per_trial.append(fitness)

            # trial: left
            print("left trial")
            self.trainer.reset_for_left()
            self.run_robot()
            fitness = self.trainer.cal_left_fitness_with_reward((self.state == 4), self.time_count, self.velocity_left, self.velocity_right)
            print("Fitness: {}".format(fitness))
            fitness_per_trial.append(fitness)

            current_interaction += 1

        fitness = np.mean(fitness_per_trial)
        current = [generation, genotype, fitness]
        self.trainer.genotypes.append(current)

        return fitness


if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_optimization()
