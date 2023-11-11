import sys

from controller import Supervisor, Keyboard


class Reset:
    def __init__(self):
        self.supervisor = Supervisor()

        self.reset_complete = False
        self.light_on = False
        self.finish_train = False
        self.message = None

        self.__init_parameters()
        self.__init_robot_node()
        self.__init_light_node()
        self.__init_receiver_and_emitter()
        self.__init_display()
        self.__init_obstacle_node()

    def __init_display(self):
        # Display: screen to plot the fitness values of the best individual and the average of the entire population
        self.display = self.supervisor.getDevice("display")
        self.width = self.display.getWidth()
        self.height = self.display.getHeight()
        self.prev_best_fitness = 0.0
        self.prev_average_fitness = 0.0
        self.display.drawText("Fitness (Best - Red)", 0, 0)
        self.display.drawText("Fitness (Average - Green)", 0, 10)

    def __init_parameters(self):
        self.time_step = 32

    def __init_robot_node(self):
        self.robot_node = self.supervisor.getFromDef("Robot")
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")

        self.end_position = [0.102188, 0.901722, -6.3962e-05]
        self.initial_trans = [-0.685987, -0.66, -6.3949e-05]
        self.initial_rot = [0.000585216, -0.000550635, 1, 1.63194]

    def __init_obstacle_node(self):
        obs_wooden_1 = self.supervisor.getFromDef("wooden_box_1")
        obs_wooden_2 = self.supervisor.getFromDef("wooden_box_2")
        obs_cyn_1 = self.supervisor.getFromDef("OBS_Cyn1")
        obs_cyn_2 = self.supervisor.getFromDef("OBS_Cyn2")
        obs_final_1 = self.supervisor.getFromDef("OBS_FINAL_1")
        obs_final_2 = self.supervisor.getFromDef("OBS_FINAL_2")
        obs_final_3 = self.supervisor.getFromDef("OBS_FINAL_3")

        self.OBS = [obs_wooden_1, obs_wooden_2, obs_cyn_1, obs_cyn_2, obs_final_1, obs_final_2, obs_final_3]
        self.OBS_trans_field = []
        self.OBS_rot_field = []
        for obs in self.OBS:
            self.OBS_trans_field.append(obs.getField("translation"))
            self.OBS_rot_field.append(obs.getField("rotation"))

        self.OBS_trans_vec = [[0.35, 0.26, 0.0496067],
                              [-0.21, 0.69, 0.0496067],
                              [0.38, 0.71, 0.05],
                              [-0.23, 0.3, 0.05],
                              [0.17, 0.97, 0],
                              [0.1, 0.995, 0],
                              [0.04, 0.97, 0]]
        self.OBS_rot_vec = [[-0.96691, -0.175804, 0.184873, 0],
                            [1.81694e-12, 5.45078e-12, 1, 0],
                            [0, 0, 1, 0],
                            [0, 0, 1, 0],
                            [0, 0, 1, 1.57],
                            [0, 0, 1, -3.14],
                            [0, 0, 1, 1.57]]

    def __init_light_node(self):
        self.light_node = self.supervisor.getFromDef("Light")
        self.light_on = self.light_node.getField("on")

    def __init_receiver_and_emitter(self):
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)

    def draw_scaled_line(self, generation, y1, y2, num_generations):
        # the scale of the fitness plot
        XSCALE = int(self.width / num_generations)
        YSCALE = 100
        self.display.drawLine((generation - 1) * XSCALE, self.height - int(y1 * YSCALE), generation * XSCALE,
                              self.height - int(y2 * YSCALE))

    def plot_fitness(self, generation, best_fitness, average_fitness, num_generations):
        if generation > 0:
            self.display.setColor(0xff0000)  # red
            self.draw_scaled_line(generation, self.prev_best_fitness, best_fitness, num_generations)

            self.display.setColor(0x00ff00)  # green
            self.draw_scaled_line(generation, self.prev_average_fitness, average_fitness, num_generations)

        self.prev_best_fitness = best_fitness
        self.prev_average_fitness = average_fitness

    def handle_receiver(self):
        while self.receiver.getQueueLength() > 0:
            # print("received message")
            received_data = self.receiver.getString()

            if received_data[:3] == "plt":
                values = received_data.split()
                generation = int(values[1])
                best = float(values[2])
                average = float(values[3])
                num_generations = int(values[4])
                self.plot_fitness(generation, best, average, num_generations)
            if received_data == "turn_on_light":
                self.reset_robot()
                self.reset_obstacle()
                self.turn_on_light()
            elif received_data == "turn_off_light":
                self.reset_robot()
                self.reset_obstacle()
                self.turn_off_light()

            self.receiver.nextPacket()

    def send_message(self, message):
        message = message.encode("utf-8")
        self.emitter.send(message)

    def reset_robot(self):
        self.trans_field.setSFVec3f(self.initial_trans)
        self.rot_field.setSFRotation(self.initial_rot)
        self.robot_node.resetPhysics()

    def reset_obstacle(self):
        for i in range(len(self.OBS_trans_field)):
            self.OBS_trans_field[i].setSFVec3f(self.OBS_trans_vec[i])
            self.OBS_rot_field[i].setSFRotation(self.OBS_rot_vec[i])
            self.OBS[i].resetPhysics()

    def turn_on_light(self):
        self.light_on.setSFBool(True)

    def turn_off_light(self):
        self.light_on.setSFBool(False)

    def train(self):
        # print("Run Simulation")
        while self.supervisor.step(self.time_step) != -1:
            self.handle_receiver()


if __name__ == '__main__':
    trainer = Reset()

    # Function used to run the best individual or the GA
    keyboard = Keyboard()
    keyboard.enable(50)

    # Interface
    print("***************************************************************************************************")
    print("To start the simulation please click anywhere in the SIMULATION WINDOW(3D Window) and press either:")
    print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
    print("***************************************************************************************************")

    resp = None
    print("getting key...")
    while trainer.supervisor.step(trainer.time_step) != -1 and resp not in [82, 83, 65619]:
        resp = keyboard.getKey()

    if resp == 83 or resp == 65619:
        trainer.send_message("True")
        print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
        # print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
    elif resp == 82 or resp == 65619:
        trainer.send_message("False")
        print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")

    # trainer.send_message("True")
    trainer.train()
