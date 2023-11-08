import numpy as np

from ga import GA
from mlp import MLP
import math


class Trainer:

    def __init__(self, robot, time_step):
        self.population = None
        self.run_complete = False
        self.new_genotype_flag = False
        self.genotype = None

        self.__init_mlp()
        self.__init_ga()
        self.__init_receiver_and_emitter(robot, time_step)

    def __init_mlp(self):
        self.number_input_layer = 12
        self.number_hidden_layer = [12, 12]
        self.number_output_layer = 2

        self.number_neurons_per_layer = []
        self.number_neurons_per_layer.append(self.number_input_layer)
        self.number_neurons_per_layer.extend(self.number_hidden_layer)
        self.number_neurons_per_layer.append(self.number_output_layer)

        self.network = MLP(self.number_neurons_per_layer)
        self.inputs = []

        # Calculate the number of weights of your MLP
        self.num_weights = 0
        for n in range(1, len(self.number_neurons_per_layer)):
            if n == 1:
                self.num_weights += (self.number_neurons_per_layer[n - 1] + 1) * self.number_neurons_per_layer[n]
            else:
                self.num_weights += self.number_neurons_per_layer[n - 1] * self.number_neurons_per_layer[n]

        self.fitness_values = []
        self.fitness = 0

    def __init_ga(self):
        # Creating the initial population
        self.population = []
        # All Genotypes
        self.genotypes = []

    def __init_receiver_and_emitter(self, robot, time_step):
        self.emitter = robot.getDevice("emitter")
        self.receiver = robot.getDevice("receiver")
        self.receiver.enable(time_step)

    def __check_for_new_genes(self):
        if not self.new_genotype_flag:
            return

        # Split the list based on the number of layers of your network
        part = []
        for n in range(1, len(self.number_neurons_per_layer)):
            if n == 1:
                part.append((self.number_neurons_per_layer[n - 1] + 1) * (self.number_neurons_per_layer[n]))
            else:
                part.append(self.number_neurons_per_layer[n - 1] * self.number_neurons_per_layer[n])

        # Set the weights of the network
        data = []
        weights_part = []
        summary = 0
        for n in range(1, len(self.number_neurons_per_layer)):
            if n == 1:
                weights_part.append(self.genotype[n - 1:part[n - 1]])
            elif n == (len(self.number_neurons_per_layer) - 1):
                weights_part.append(self.genotype[summary:])
            else:
                weights_part.append(self.genotype[summary:summary + part[n - 1]])
            summary += part[n - 1]

        # weights_part = np.array(weights_part)
        for n in range(1, len(self.number_neurons_per_layer)):
            if n == 1:
                weights_part[n - 1] = weights_part[n - 1].reshape(
                    [self.number_neurons_per_layer[n - 1] + 1, self.number_neurons_per_layer[n]])
            else:
                weights_part[n - 1] = weights_part[n - 1].reshape(
                    [self.number_neurons_per_layer[n - 1], self.number_neurons_per_layer[n]])

            data.append(weights_part[n - 1])
        self.network.weights = data

        # Reset fitness list
        self.fitness_values = []

    @staticmethod
    def calculate_weight(x):
        mu = 100
        sigma = 15
        weight = 1 / (sigma * math.sqrt(2 * math.pi)) * math.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
        return weight

    def get_output_and_cal_fitness(self):
        self.__check_for_new_genes()
        output = self.network.propagate_forward(self.inputs)
        self.__calculate_fitness(output[0], output[1])

        return output

    def __calculate_fitness(self, velocity_left, velocity_right):
        forward_fitness = (velocity_left + velocity_right) / 2.0
        avoid_collision_fitness = 1 - max(self.inputs[4:12])
        spinning_fitness = 1 - (abs(velocity_left - velocity_right) ** 0.5)
        combined_fitness = forward_fitness * avoid_collision_fitness * spinning_fitness

        self.fitness_values.append(combined_fitness)
        self.fitness = np.mean(self.fitness_values)

    def __wait_for_message(self):
        reset_complete = False
        while self.receiver.getQueueLength() > 0:

            received_data = self.receiver.getString()
            if received_data == "True":
                reset_complete = True
            self.receiver.nextPacket()

        return reset_complete

    def send_command_and_wait(self, message):
        message = message.encode("utf-8")
        self.emitter.send(message)

        # while not self.__wait_for_message():
        #     pass

    def reset_for_left(self):
        self.send_command_and_wait("turn_on_light")

    def reset_for_right(self):
        self.send_command_and_wait("turn_off_light")

    def plt(self, generation, best, average):
        self.send_command_and_wait("plt " + str(generation) + " " + str(best) + " " + str(average) + " " + str(GA.num_generations))

    def cal_left_fitness_with_reward(self, reach_goal_flag, time_count, left_speed, right_speed):
        left_light = 0
        if self.inputs[0] == 1 and right_speed > left_speed:
            left_light = (1 - self.inputs[3])

        ground_sensors = sum((1 - self.inputs[i]) for i in range(1, 4))
        distance_sensors = max((self.calculate_weight(self.inputs[i]) for i in range(4, 12)))
        if self.inputs[0] == 0:
            distance_sensors = 0
        reach_goal = 20 if reach_goal_flag else 0

        print(self.fitness, left_light, ground_sensors, distance_sensors, reach_goal, (time_count / 1000.0) * 0.1)
        fitness = self.fitness*3 + left_light + ground_sensors*3 + distance_sensors*2 + left_light
        fitness += reach_goal - (time_count / 1000.0) * 0.1

        return fitness

    def cal_right_fitness_with_reward(self, reach_goal_flag, time_count, left_speed, right_speed):
        right_light = 0
        if self.inputs[0] == -1 and left_speed > right_speed:
            right_light = (1 - self.inputs[1])

        ground_sensors = sum((1 - self.inputs[i]) for i in range(1, 4))
        distance_sensors = max((self.calculate_weight(self.inputs[i]) for i in range(4, 12)))
        if self.inputs[0] == 0:
            distance_sensors = 0
        reach_goal = 20 if reach_goal_flag else 0

        print(self.fitness, right_light, ground_sensors*3, distance_sensors*2, reach_goal, (time_count / 1000.0) * 0.1)
        fitness = self.fitness*3 + ground_sensors + distance_sensors + right_light
        fitness += reach_goal + right_light - (time_count / 1000.0) * 0.1

        return fitness