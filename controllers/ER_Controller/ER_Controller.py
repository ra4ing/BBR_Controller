import numpy as np
from controller import Robot, keyboard
from ga import GA
from trainer import Trainer


class Controller:
    def __init__(self, robot):
        self.robot = robot

        self.velocity_left = None
        self.velocity_right = None

        self.__init_parameters()  # Robot Parameters
        self.__init_trainer()  # Initialize MLP
        self.__enable_camera()  # Enable Camera
        self.__enable_motors()  # Enable Motors
        self.__enable_distance_sensors()  # Enable Distance Sensors
        self.__enable_light_sensors()  # Enable Light Sensors
        self.__enable_ground_sensors()  # Enable Ground Sensors

    def reset(self):
        self.stop()
        self.velocity_left = 0
        self.velocity_right = 0

        self.__init_parameters()  # Robot Parameters

    def __init_parameters(self):
        self.time_step = 32  # ms
        self.max_speed = 6.28  # m/s
        self.max_time = 120.0

        self.state = 0
        self.choose_path = 0.5
        self.time_count = 0

    def __init_trainer(self):
        self.trainer = Trainer(self.robot)

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
    def normalize_value(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    @staticmethod
    def map_range(val, from_min, from_max, to_min, to_max):
        from_range = from_max - from_min
        to_range = to_max - to_min
        scale = (val - from_min) / from_range

        return to_min + (scale * to_range)

    def __read_light_sensors(self):
        self.trainer.inputs.append(self.choose_path)
        if self.choose_path != 0.5:
            return

        min_ls = 0
        max_ls = 4300

        lights = []
        for i in range(8):
            temp = self.light_sensors[i].getValue()
            # Adjust Values
            temp = self.adjust_value(temp, min_ls, max_ls)
            lights.append(temp)

        if min(lights) < 500:
            self.choose_path = 1
        elif (self.time_count / 1000.0) >= 8.0:
            self.choose_path = 0

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

        # # save value for evolutionary
        # left = self.normalize_value(left, 300, 700)
        # center = self.normalize_value(center, 300, 700)
        # right = self.normalize_value(right, 300, 700)

        left_normalized = self.normalize_value(left, min_gs, max_gs)
        center_normalized = self.normalize_value(center, min_gs, max_gs)
        right_normalized = self.normalize_value(right, min_gs, max_gs)

        self.trainer.inputs.append(left_normalized)
        self.trainer.inputs.append(center_normalized)
        self.trainer.inputs.append(right_normalized)

    def __read_distance_sensors(self):

        min_ds = 0
        max_ds = 2400

        # Read Distance Sensors
        for sensor in self.distance_sensors:
            temp = sensor.getValue()  # Get value
            temp = self.adjust_value(temp, min_ds, max_ds)  # Adjust Values

            temp_normalized = self.normalize_value(temp, min_ds, max_ds)  # save value for evolutionary
            temp_normalized = temp_normalized * 5
            self.trainer.inputs.append(temp_normalized)

    def __read_camera(self):
        # if self.time_count / 1000 <= 40.0:
        #     return

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
        self.trainer.inputs = []
        self.__read_light_sensors()
        self.__read_ground_sensors()
        self.__read_distance_sensors()
        self.__read_camera()
        # self.trainer.inputs.append(((self.time_count / 1000) / 120.0))

    def take_move(self):
        # print(self.state)

        output = self.trainer.get_output_and_cal_fitness()
        self.velocity_left = output[0] * self.max_speed
        self.velocity_right = output[1] * self.max_speed

        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)
        # self.stop()

    def stop(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

    def run_optimization(self):
        self.trainer.population = GA.create_random_population(self.trainer.num_weights)

        print(">>>Starting Evolution using GA optimization ...\n")

        # For each Generation
        for generation in range(GA.num_generations):
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("Generation: {}".format(generation))
            current_population = []
            # Select each Genotype or Individual

            for population in range(GA.num_population):
                print("-------------------")
                print("Population: {}".format(population))
                self.trainer.genotype = self.trainer.population[population]
                fitness = self.__evaluate_genotype(self.trainer.genotype, generation)
                # Save its fitness value
                current_population.append((self.trainer.genotype, float(fitness)))
                # print(current_population)

            best = GA.get_best_genotype(current_population)
            average = GA.get_average_genotype(current_population)
            print("-------------------")
            print("Best: {}".format(best[1]))
            print("Average: {}".format(average))
            np.save("Best.npy", best[0])
            self.trainer.plt(generation, best[1], average)

            # Generate the new population_idx using genetic operators
            if generation < GA.num_generations - 1:
                self.trainer.population = GA.population_reproduce(current_population)

        # print("All Genotypes: {}".format(self.genotypes))
        print("GA optimization terminated.\n")

    def run_best(self):
        self.trainer.genotype = np.load("Best.npy")

        fitness = self.__evaluate_genotype(self.trainer.genotype, None)

        print("Fitness: {}".format(fitness))
        print("GA demo terminated.\n")

    def run_robot(self):
        self.reset()
        fitness = 0

        while self.robot.step(self.time_step) != -1:
            self.__read_data()
            self.take_move()
            fitness += self.trainer.cal_fitness_with_reward([self.velocity_left, self.velocity_right])

            self.time_count += self.time_step
            if self.state == 4:
                print("reach goal!!!")
                break
            elif (self.time_count / 1000) >= self.max_time:
                break

        fitness = (fitness * self.time_step) / self.time_count
        if self.state == 4:
            fitness *= 50 * ((self.time_count / 1000) / 120.0)
        return fitness

    def __evaluate_genotype(self, genotype, generation):
        self.trainer.update_mlp()

        fitness_per_trial = []

        number_interaction_loops = 1
        for i in range(number_interaction_loops):
            # trial: right
            self.trainer.reset_environment("right")
            fitness = self.run_robot()
            fitness_per_trial.append(fitness)

            # trial: left
            self.trainer.reset_environment("left")
            fitness = self.run_robot()
            fitness_per_trial.append(fitness)

        fitness = np.mean(fitness_per_trial)
        print("Fitness: {}".format(fitness))
        current = [generation, genotype, fitness]
        self.trainer.genotypes.append(current)

        return fitness


if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    flag = False
    while controller.robot.step(controller.time_step) != -1 and not controller.trainer.wait_message:
        flag = controller.trainer.wait_for_message()

    if flag:
        print("optimization")
        controller.run_optimization()
    else:
        print("run_best")
        controller.run_best()
