import numpy as np
from ga import GA
from mlp import MLP


class Trainer:

    def __init__(self, robot):

        self.robot = robot
        self.genotype = None
        self.last_genotype = None
        self.wait_message = False
        self.enable_dis_reward = None

        self.time_step = 32
        self.online_time = 0
        self.offline_time = 0
        self.avoid_collision_time = 0
        self.gs_rewards_count = 0
        self.ds_rewards_count = 0
        self.ff = 1
        self.af = 1
        self.sp = 1

        self.__init_mlp()
        self.__init_receiver_and_emitter(robot)

    def reset(self):
        self.online_time = 0
        self.offline_time = 0
        self.avoid_collision_time = 0
        self.gs_rewards_count = 0
        self.ds_rewards_count = 0
        self.ff = 1
        self.af = 1
        self.sp = 1

    def __init_mlp(self):
        self.number_input_layer = 12
        self.number_hidden_layer = [12, 10]
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
        while self.robot.step(self.time_step) != -1:
            while self.receiver.getQueueLength() > 0:
                received_data = self.receiver.getString()
                if received_data == 'ok':
                    return
                self.receiver.nextPacket()

    def plt(self, generation, best, average):
        self.send_command(
            "plt " + str(generation) + " " + str(best) + " " + str(average) + " " + str(GA.num_generations))

    def __cal_distance_weight(self, choose_right, choose_left):
        weight = 0
        threshold = 50

        if choose_right or choose_left:
            self.avoid_collision_time += 1
            if self.avoid_collision_time >= threshold and self.enable_dis_reward:
                # print("###")
                self.ds_rewards_count += 1
                self.avoid_collision_time = 0
                self.enable_dis_reward = False
                weight = 500

        if self.ds_rewards_count > 2 or self.gs_rewards_count <= 1:
            weight = 0

        return weight

    def __cal_ground_weight(self, choose_right, choose_left):
        weight = 0
        threshold = 170

        if choose_left or choose_right:
            if self.online_time >= threshold:
                self.gs_rewards_count += 1
                self.online_time = 0
                weight = 1000

        if (self.gs_rewards_count > 2 and self.ds_rewards_count == 0) or self.gs_rewards_count > 4:
            weight = 0

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
    def __combine_fitness_with_reward(ff, af, sf, gr, dr):

        ret = (ff ** 4) * (af ** 5) * (sf ** 8) * (gr + dr)
        return ret

    def cal_fitness_and_reward(self, speed):

        # fitness
        # forward_fitness = self.normalize_value((speed[0] + speed[1]) / 2.0, -6.28, 6.28)
        # avoid_collision_fitness = 1 - max(self.inputs[3:11])
        # spinning_fitness = 1 - self.normalize_value(abs(speed[0] - speed[1]), 0, 12.56)

        self.ff += ((speed[0] + speed[1]) / 2.0) / (6.28 * 100)
        self.af += (0.5 - max(self.inputs[3:11])) / 100
        self.sp += (0.5 - self.normalize_value(abs(speed[0] - speed[1]), 0, 12.56)) / 100

        if min(speed) < 0:
            self.online_time = 0
            self.offline_time = 0
            self.avoid_collision_time = 0

        ds = self.inputs[3:11]
        gs = self.inputs[0:3]
        ls = self.inputs[11]

        flag_0_7 = 0.148 < ds[0] < 0.35 or 0.148 < ds[7] < 0.35
        flag_1_6 = 0.148 < ds[1] < 0.35 or 0.148 < ds[6] < 0.35
        flag_2_5 = 0.148 < ds[2] < 0.35 or 0.148 < ds[5] < 0.35
        flag_for_right = 0.148 < ds[2] < 0.35
        flag_for_left = 0.148 < ds[5] < 0.35
        left = gs[0] < 0.5
        center = gs[1] < 0.5
        right = gs[2] < 0.5
        offline = not left and not center and not right

        gs_choose_right = ls > 0 and left and not right
        gs_choose_left = ls < 0 and right and not left
        ds_choose_right = ls > 0 and flag_for_left
        ds_choose_left = ls < 0 and flag_for_left

        if offline:
            self.offline_time += 1
            if self.offline_time >= 50:
                self.online_time = 0
        else:
            self.online_time += 1
            if self.online_time >= 64:
                self.offline_time = 0
                self.avoid_collision_time = 0
                self.enable_dis_reward = True

        # rewards
        ground_rewards = self.__cal_ground_weight(gs_choose_right, gs_choose_left)
        distance_rewards = self.__cal_distance_weight(ds_choose_right, ds_choose_left)
        scale = max((self.ff + self.af + self.sp), 0)

        if flag_0_7 or flag_1_6 or flag_2_5:
            ground_rewards = 0
        if not offline:
            distance_rewards = 0

        return (ground_rewards + distance_rewards) * scale * 0.1
