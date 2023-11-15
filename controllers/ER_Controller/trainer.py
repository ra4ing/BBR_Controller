import numpy as np
from ga import GA
from mlp import MLP


class Trainer:

    def __init__(self, robot):
        self.robot = robot
        self.genotype = None
        self.last_genotype = None
        self.wait_message = False

        self.time_step = 32
        self.online_time = 0
        self.offline_time = 0

        self.__init_mlp()
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

    def __init_receiver_and_emitter(self, robot):
        self.emitter = robot.getDevice("emitter")
        self.receiver = robot.getDevice("receiver")
        self.receiver.enable(self.time_step)

    def update_mlp(self):

        if np.array_equal(self.genotype, self.last_genotype):
            return
        self.last_genotype = self.genotype

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
        output = self.network.propagate_forward(self.inputs)
        output[0] = output[0] * 6.28
        output[1] = output[1] * 6.28

        return output

    def wait_for_message(self):
        while self.receiver.getQueueLength() > 0:
            self.wait_message = True
            received_data = self.receiver.getString()
            if received_data is not None:
                return received_data
            self.receiver.nextPacket()

        return None

    def send_command(self, message):
        message = message.encode("utf-8")
        self.emitter.send(message)

    def reset_environment(self, direction):
        if direction == "right":
            self.send_command("turn_off_light")
        else:
            self.send_command("turn_on_light")

    def wait_reset_complete(self):
        self.robot.step(self.time_step)

    def plt(self, generation, best, average):
        self.send_command(
            "plt " + str(generation) + " " + str(best) + " " + str(average) + " " + str(GA.num_generations))

    def __cal_distance_weight(self, ds, offline):
        # print("###")
        # print(ds)
        # print(ds[5])
        weight = 0.1
        flag_0_7 = 0.145 < ds[0] < 0.35 or 0.145 < ds[7] < 0.35
        flag_1_6 = 0.145 < ds[1] < 0.35 or 0.145 < ds[6] < 0.35
        flag_2_5 = 0.145 < ds[2] < 0.35 or 0.145 < ds[5] < 0.35

        if offline:
            if self.offline_time < 1:
                self.offline_time += 0.05
            if flag_0_7 or flag_1_6:
                weight = 0.3
            if flag_2_5:
                weight = 1 - self.offline_time

        if max(ds) > 0.5:
            weight = 0.1

        return weight

    def __cal_ground_weight(self, gs, ls, speed):
        # weight = np.mean([(1 - i) for i in gs])
        if (speed[0] + speed[1]) / 2 <= 2.0:
            return 0.1

        weight = 0.1
        flag_0 = gs[0] < 0.5
        flag_1 = gs[1] < 0.5
        flag_2 = gs[2] < 0.5

        offline = not flag_0 and not flag_1 and not flag_2
        if offline:
            weight = 0.01

        # 右路：左中在线上右在线外加分
        if ls > 0:
            if flag_0 and flag_1 and not flag_2:
                # if speed[0] == speed[1]:
                weight = 2
        # 左路：右中在线上左在线外加分
        elif ls < 0:
            if flag_1 and flag_2 and not flag_0:
                # if speed[0] == speed[1]:
                weight = 2
        # 都在线上加分
        if flag_0 and flag_1 and flag_2:
            if speed[0] == speed[1]:
                weight = 1
        # 只有左在线上左转加分
        if flag_0 and not flag_1 and not flag_2:
            if speed[0] < speed[1]:
                weight = 1
        # 只有右在线上右转加分
        if flag_2 and not flag_0 and not flag_1:
            if speed[0] > speed[1]:
                weight = 1

        # if offline:
        #     weight = 0.1
        # else:
        #     if self.online_time < 0.5:
        #         weight += self.online_time
        #         self.online_time += 0.01

        # return self.adjust_value(weight + self.online_time, 0, 1)
        return weight

    @staticmethod
    def __cal_light_weight(choose_path, online, offline, speed):
        weight = 0.1
        if choose_path > 0:
            if offline and speed[0] < speed[1]:
                weight = 1
        elif choose_path < 0:
            if offline and speed[0] > speed[1]:
                weight = 1

        return weight

    @staticmethod
    def normalize_value(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    @staticmethod
    def adjust_value(val, min_val, max_val):
        val = max(val, min_val)
        val = min(val, max_val)
        return val

    @staticmethod
    def __combine_fitness_with_reward(ff, af, sf, gr, dr, lr):

        ret = ff * (af**2) * sf * (gr ** 2)  # * ls  # * ds
        # if ret == 0:
        #     print("###")
        #     print("ff\t\t\tsf\t\t\tgs\t\t\tds\t\t\tls\t\t\tret")
        #     print(str(ff) + "\t\t" + str(sf*5) + "\t\t" + str(gs*10) + "\t\t" + str(ds*3) + "\t\t" + str(ls*10) + "\t\t" + str(ret))
        return ret

    def cal_fitness_and_reward(self, speed):
        online = self.inputs[0] < 0.5 and self.inputs[1] < 0.5 and self.inputs[2] < 0.5
        offline = self.inputs[0] > 0.5 and self.inputs[1] > 0.5 and self.inputs[2] > 0.5

        # if not offline:
        #     self.offline_time = 0
        # if not online:
        #     self.online_time = 0

        # fitness
        forward_fitness = self.normalize_value((speed[0] + speed[1]) / 2.0, -6.28, 6.28)
        avoid_collision_fitness = 1 - max(self.inputs[3:11])
        spinning_fitness = 1 - self.normalize_value(abs(speed[0] - speed[1]), 0, 12.56)

        # rewards
        light_rewards = self.__cal_light_weight(self.inputs[11], online, offline, speed)
        ground_rewards = self.__cal_ground_weight(self.inputs[0:3], self.inputs[11], speed)
        distance_rewards = self.__cal_distance_weight(self.inputs[3:11], offline)

        return self.__combine_fitness_with_reward(forward_fitness, avoid_collision_fitness, spinning_fitness,
                                                  ground_rewards, distance_rewards, light_rewards)
