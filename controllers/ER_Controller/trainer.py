import numpy as np

from ga import GA
from mlp import MLP


class Trainer:

    def __init__(self, robot):
        self.robot = robot
        self.population = None
        self.fitness_values = None
        self.new_genotype_flag = False
        self.genotype = None
        self.fitness_sum = None
        self.time_step = 32
        self.wait_message = False

        self.__init_mlp()
        self.__init_ga()
        self.__init_receiver_and_emitter(robot)

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

    def __init_receiver_and_emitter(self, robot):
        self.emitter = robot.getDevice("emitter")
        self.receiver = robot.getDevice("receiver")
        self.receiver.enable(self.time_step)

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

    def get_output_and_cal_fitness(self):
        self.__check_for_new_genes()
        output = self.network.propagate_forward(self.inputs)

        output[0] = self.adjust_value(output[0], 0, 6.28)
        output[1] = self.adjust_value(output[1], 0, 6.28)
        self.__calculate_fitness(output[0], output[1])

        return output

    def wait_for_message(self):
        while self.receiver.getQueueLength() > 0:
            self.wait_message = True
            received_data = self.receiver.getString()
            if received_data == "True":
                return True
            self.receiver.nextPacket()

        return False

    def send_command(self, message):
        message = message.encode("utf-8")
        self.emitter.send(message)

    def reset_environment(self, direction):
        if direction == "right":
            self.send_command("turn_off_light")
        else:
            self.send_command("turn_on_light")
        self.fitness_values = []

    def plt(self, generation, best, average):
        self.send_command(
            "plt " + str(generation) + " " + str(best) + " " + str(average) + " " + str(GA.num_generations))

    @staticmethod
    def __cal_distance_weight(ds):
        # print("###")
        # print(ds)
        # print(ds[2])
        weight = 0.5
        flag_0_7 = 0.15 < ds[0] < 0.3 or 0.15 < ds[7] < 0.3
        flag_1_6 = 0.15 < ds[1] < 0.3 or 0.15 < ds[6] < 0.3
        flag_2_5 = 0.15 < ds[2] < 0.3 or 0.15 < ds[5] < 0.3
        if flag_0_7:
            weight = 0.6
        if flag_1_6:
            weight = 0.8
        if flag_2_5:
            weight = 1

        if max(ds) > 0.5:
            weight = 0

        return weight

    @staticmethod
    def __cal_ground_weight(gs):
        temp = [(1 - i) for i in gs]
        temp = np.mean(temp)

        return temp

    @staticmethod
    def __cal_light_weight(choose_path, speed, gs):
        weight = 0
        if choose_path == 0:
            if gs[0] < 0.5 and gs[1] < 0.5 and gs[2] < 0.5 and speed[0] > speed[1]:
                weight = 0.08
        elif choose_path == 1:
            if gs[0] < 0.5 and gs[1] < 0.5 and gs[2] < 0.5 and speed[0] < speed[1]:
                weight = 0.08

        if gs[0] > 0.5 and gs[1] > 0.5 and gs[2] > 0.5:
            weight = -0.15

        return weight

    @staticmethod
    def normalize_value(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    @staticmethod
    def adjust_value(val, min_val, max_val):
        val = max(val, min_val)
        val = min(val, max_val)
        return val

    def __combine_fitness_with_reward(self, fitness, gs, ds, ls):
        fitness = self.adjust_value(fitness, 0, 1)
        ds = self.adjust_value(ds, 0, 1)
        gs = self.adjust_value(gs + ls, 0, 1) if gs > 0.4 else 0

        ret = (fitness) * (ds) * (gs*2)
        # print("###")
        # print("fitness\tgs\t\tds\tls\tret")
        # print(str(fitness) + "\t" + str(gs) + "\t" + str(ds) + "\t" + str(ls) + "\t" + str(ret))
        return ret * 10

    def cal_fitness_with_reward(self, speed):

        light_sensors = self.__cal_light_weight(self.inputs[0], speed, self.inputs[1:4])
        ground_sensors = self.__cal_ground_weight(self.inputs[1:4])
        distance_sensors = self.__cal_distance_weight(self.inputs[4:12])

        return self.__combine_fitness_with_reward(self.fitness, ground_sensors, distance_sensors, light_sensors)

    def __calculate_fitness(self, velocity_left, velocity_right):
        forward_fitness = self.normalize_value((velocity_left + velocity_right) / 2.0, 0, 6.28)
        # avoid_collision_fitness = 1 - self.normalize_value(max(self.inputs[4:12]), 0, 2400)
        avoid_collision_fitness = 1
        spinning_fitness = 1 - self.normalize_value(abs(velocity_left - velocity_right), 0, 6.28)
        # spinning_fitness = 1 - (abs(velocity_left - velocity_right) ** 0.5)
        combined_fitness = (forward_fitness) * (avoid_collision_fitness) * (spinning_fitness**2)
        # print("###")
        # print(str(forward_fitness) + "\t" + str(avoid_collision_fitness) + "\t" + str(spinning_fitness) + "\t" + str(combined_fitness))
        # self.fitness_values.append(combined_fitness)
        # self.fitness = np.mean(self.fitness_values)
        self.fitness = combined_fitness

