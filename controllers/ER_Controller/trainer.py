import numpy as np
from ga import GA
from mlp import MLP


class Trainer:
    """
    The Trainer class is responsible for training a neural network (Multi-Layer Perceptron)
    using genetic algorithms for a robot control task.
    """
    def __init__(self, robot):
        """
        Initializes the Trainer with the provided robot and sets up various parameters
        and states required for training.

        Args:
            robot: The robot instance to be controlled and trained.
        """
        self.robot = robot
        self.genotype = None
        self.last_genotype = None
        self.wait_message = False   # Flag to indicate if waiting for a message from the robot.
        self.enable_dis_reward = None

        self.time_step = 32
        self.online_time = 0
        self.offline_time = 0
        self.avoid_collision_time = 0
        # Counters for rewards based on ground sensor (gs) rewards and distance sensor (ds) rewards.
        self.gs_rewards_count = 0
        self.ds_rewards_count = 0

        # Fitness factors: ff (forward), af (avoid obstacle), sp (spin)
        self.ff = 1
        self.af = 1
        self.sp = 1

        self.__init_mlp()
        self.__init_receiver_and_emitter(robot)

    def reset(self):
        """
        Resets various time and reward counters to their initial state.
        """
        self.online_time = 0
        self.offline_time = 0
        self.avoid_collision_time = 0
        self.gs_rewards_count = 0
        self.ds_rewards_count = 0
        self.ff = 1
        self.af = 1
        self.sp = 1

    def __init_mlp(self):
        """
        Initializes the Multi-Layer Perceptron (MLP) neural network with defined
        layers and neuron counts.
        """
        # Reuse from lab4 example
        self.number_input_layer = 12
        self.number_hidden_layer = [12, 10]  # 12 layer for each sensor/input and 5 layer for each wheel
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
        # End lab4 example

    def __init_receiver_and_emitter(self, robot):
        """
        Initializes the communication devices (emitter and receiver) for the robot.

        Args:
            robot: The robot instance for which the devices are to be initialized.
        """
        self.emitter = robot.getDevice("emitter")
        self.receiver = robot.getDevice("receiver")
        self.receiver.enable(self.time_step)

    def update_mlp(self):
        """
        Updates the weights of the MLP network if there's a new genotype available.
        """
        # Reuse from lab4 example
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
        # End lab4 example

    def get_output_and_cal_fitness(self):
        """
        Propagates inputs through the MLP to get outputs and calculates the fitness
        based on the outputs.

        Returns:
            The output of the MLP after forward propagation.
        """
        output = self.network.propagate_forward(self.inputs)
        output[0] = output[0] * 6.28
        output[1] = output[1] * 6.28

        return output

    def wait_for_message(self):
        """
        Waits and retrieves messages from the robot's receiver device.

        Returns:
            The received data message, or None if no message is received.
        """
        while self.receiver.getQueueLength() > 0:
            self.wait_message = True
            received_data = self.receiver.getString()
            if received_data is not None:
                return received_data
            self.receiver.nextPacket()

        return None

    def send_command(self, message):
        """
        Sends a command to the robot through the emitter device.

        Args:
            message: The message to be sent to the robot.
        """
        message = message.encode("utf-8")
        self.emitter.send(message)

    def reset_environment(self, direction):
        """
        Resets the environment based on the specified direction.

        Args:
            direction: The direction ('right' or other) to determine the reset action.
        """
        if direction == "right":
            self.send_command("turn_off_light")
        else:
            self.send_command("turn_on_light")

    def wait_reset_complete(self):
        """
        Waits until the environment reset is acknowledged by the robot.
        """
        while self.robot.step(self.time_step) != -1:
            while self.receiver.getQueueLength() > 0:
                received_data = self.receiver.getString()
                if received_data == 'ok':
                    return
                self.receiver.nextPacket()

    def plt(self, generation, best, average):
        """
        Sends a plotting command to the robot with the specified parameters.

        Args:
            generation: The current generation number.
            best: The best fitness score in the current generation.
            average: The average fitness score in the current generation.
        """
        self.send_command(
            "plt " + str(generation) + " " + str(best) + " " + str(average) + " " + str(GA.num_generations))

    def __cal_distance_weight(self, choose_right, choose_left):
        """
        Calculates the weight based on distance sensors.

        Args:
            choose_right: Boolean indicating if the right choice is made based on sensor data.
            choose_left: Boolean indicating if the left choice is made based on sensor data.

        Returns:
            The calculated weight based on distance sensors.
        """
        weight = 0
        threshold = 50

        # avoiding collision for some time and gained follow line rewards last time will give rewards
        if choose_right or choose_left:
            self.avoid_collision_time += 1
            if self.avoid_collision_time >= threshold and self.enable_dis_reward:
                self.ds_rewards_count += 1
                self.avoid_collision_time = 0
                self.enable_dis_reward = False
                weight = 500

        # if gain follow line rewards less than 2 time or gain avoid obstacles rewards more than 2, do not give rewards
        if self.ds_rewards_count > 2 or self.gs_rewards_count <= 1:
            weight = 0

        return weight

    def __cal_ground_weight(self, choose_right, choose_left):
        """
        Calculates the weight based on ground sensors.

        Args:
            choose_right: Boolean indicating if the right choice is made based on ground sensor data.
            choose_left: Boolean indicating if the left choice is made based on ground sensor data.

        Returns:
            The calculated weight based on ground sensors.
        """
        weight = 0
        threshold = 170

        # follow line for some time will give rewards
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
        """
        Normalizes a value to a range between 0 and 1.

        Args:
            val: The value to normalize.
            min_val: The minimum value of the range.
            max_val: The maximum value of the range.

        Returns:
            The normalized value.
        """
        return (val - min_val) / (max_val - min_val)

    @staticmethod
    def adjust_value(val, min_val, max_val):
        """
        Clamps a value within the specified range.

        Args:
            val: The value to clamp.
            min_val: The minimum value of the range.
            max_val: The maximum value of the range.

        Returns:
            The clamped value.
        """
        return max(min(val, max_val), min_val)

    @staticmethod
    def __combine_fitness_with_reward(ff, af, sf, gr, dr):
        """
        Combines various fitness factors and rewards into a single fitness score.

        Args:
            ff: Forward fitness.
            af: Avoid obstacles fitness.
            sf: Spin factor.
            gr: Ground reward.
            dr: Distance reward.

        Returns:
            The combined fitness score.
        """
        ret = (ff ** 4) * (af ** 5) * (sf ** 8) * (gr + dr)
        return ret

    def cal_fitness_and_reward(self, speed):
        """
        Calculates the fitness and rewards based on the robot's speed and sensor inputs.

        Args:
            speed: The speed of the robot.

        Returns:
            The calculated fitness and reward score.
        """
        self.ff += ((speed[0] + speed[1]) / 2.0) / (6.28 * 100)
        self.af += (0.5 - max(self.inputs[3:11])) / 100
        self.sp += (0.5 - self.normalize_value(abs(speed[0] - speed[1]), 0, 12.56)) / 100
        scale = max((self.ff + self.af + self.sp), 0)

        if min(speed) < 0:
            self.online_time = 0
            self.offline_time = 0
            self.avoid_collision_time = 0

        ds = self.inputs[3:11]
        gs = self.inputs[0:3]
        ls = self.inputs[11]

        # stay some distance with obstacles
        flag_0_7 = 0.148 < ds[0] < 0.35 or 0.148 < ds[7] < 0.35
        flag_1_6 = 0.148 < ds[1] < 0.35 or 0.148 < ds[6] < 0.35
        flag_2_5 = 0.148 < ds[2] < 0.35 or 0.148 < ds[5] < 0.35

        # encourage for walking around obstacles
        flag_for_right = 0.148 < ds[2] < 0.35 or 0.148 < ds[1] < 0.35
        flag_for_left = 0.148 < ds[5] < 0.5 or 0.148 < ds[6] < 0.5

        # judge if gs on the line
        left = gs[0] < 0.5
        center = gs[1] < 0.5
        right = gs[2] < 0.5

        # on sensor not on the line
        offline = not left and not center and not right

        # conditions used for judge requirements, ls is used to ensure choose path
        gs_choose_right = ls > 0 and left and not right
        gs_choose_left = ls < 0 and right and not left
        ds_choose_right = ls > 0 and flag_for_left
        ds_choose_left = ls < 0 and flag_for_left

        # reset offline_time based on threshold
        if offline:
            self.offline_time += 1
            if self.offline_time >= 50:
                self.online_time = 0
        # reset online_time based on threshold
        else:
            self.online_time += 1
            if self.online_time >= 64:
                self.offline_time = 0
                self.avoid_collision_time = 0
                self.enable_dis_reward = True

        # rewards
        ground_rewards = self.__cal_ground_weight(gs_choose_right, gs_choose_left)
        distance_rewards = self.__cal_distance_weight(ds_choose_right, ds_choose_left)

        # do not give rewards for gr if avoiding obstacles
        if flag_0_7 or flag_1_6 or flag_2_5:
            ground_rewards = 0
        # do not give rewards for dr if touch line
        if not offline:
            distance_rewards = 0

        return (ground_rewards + distance_rewards) * scale * 0.1
