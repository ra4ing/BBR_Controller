from controller import Robot
from datetime import datetime
import math
import numpy as np


class Controller:
    def __init__(self, robot):        
        # Robot Parameters
        self.robot = robot
        self.time_step = 32 # ms
        self.max_speed = 1  # m/s
 
        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0

        # Enable Distance Sensors
        self.distance_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.distance_sensors.append(self.robot.getDevice(sensor_name))
            self.distance_sensors[i].enable(self.time_step)
    
        # Enable Light Sensors
        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ls' + str(i)
            self.light_sensors.append(self.robot.getDevice(sensor_name))
            self.light_sensors[i].enable(self.time_step)
       
        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)
        
        # Data
        self.inputs = []
        self.inputsPrevious = []
        
        # Flag
        self.flag_turn = 0
        
    def clip_value(self,value,min_max):
        if (value > min_max):
            return min_max;
        elif (value < -min_max):
            return -min_max;
        return value;

    def sense_compute_and_actuate(self):
        if(len(self.inputs) > 0 ):
            # Check for any possible collision
            #print(self.inputs)
            if(np.max(self.inputs[11:19]) > 0.4):
                # Get Current Time
                time = datetime.now()
                #print("({} - {}) Object or walls detected!".format(time.second, time.microsecond))
                self.velocity_left = 0;
                self.velocity_right = 0;
            else:
                # Check for Light 
                if(self.inputs[3] == 0 and self.inputs[4] == 0 and self.inputs[9] == 0 and self.inputs[10] == 0):
                    self.velocity_left = 1;
                    self.velocity_right = 1;
                elif(self.inputs[5] == 0):
                    self.velocity_left = 1;
                    self.velocity_right = 0.5;    
                elif(self.inputs[8] == 0):
                    self.velocity_left = 0.5;
                    self.velocity_right = 1;
                elif(self.inputs[6] == 0 and self.inputs[7] == 0):
                    self.velocity_left = -1;
                    self.velocity_right = 1;
     
        self.left_motor.setVelocity(self.velocity_left)
        self.right_motor.setVelocity(self.velocity_right)

    def run_robot(self):        
        # Main Loop
        count = 0;
        inputs_avg = []
        while self.robot.step(self.time_step) != -1:
            # Read Ground Sensors
            self.inputs = []
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()

            # Adjust Values
            min_gs = 0
            max_gs = 1000
            if(left > max_gs): left = max_gs
            if(center > max_gs): center = max_gs
            if(right > max_gs): right = max_gs
            if(left < min_gs): left = min_gs
            if(center < min_gs): center = min_gs
            if(right < min_gs): right = min_gs
            
            # Save Data
            self.inputs.append((left-min_gs)/(max_gs-min_gs))
            self.inputs.append((center-min_gs)/(max_gs-min_gs))
            self.inputs.append((right-min_gs)/(max_gs-min_gs))
            #print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))
            
            # Read Light Sensors
            for i in range(8):       
                temp = self.light_sensors[i].getValue()
                # Adjust Values
                min_ls = 0
                max_ls = 4300
                if(temp > max_ls): temp = max_ls
                if(temp < min_ls): temp = min_ls
                # Save Data
                self.inputs.append((temp-min_ls)/(max_ls-min_ls))
                #print("Light Sensors - Index: {}  Value: {}".format(i,self.light_sensors[i].getValue()))
      
            # Read Distance Sensors
            for i in range(8):       
                temp = self.distance_sensors[i].getValue()
                # Adjust Values
                min_ls = 0
                max_ls = 4300
                if(temp > max_ls): temp = max_ls
                if(temp < min_ls): temp = min_ls
                # Save Data
                self.inputs.append((temp-min_ls)/(max_ls-min_ls))
                #print("Distance Sensors - Index: {}  Value: {}".format(i,self.distance_sensors[i].getValue()))

            # Compute and actuate
            self.sense_compute_and_actuate()              
            
if __name__ == "__main__":
    my_robot = Robot()
    controller = Controller(my_robot)
    controller.run_robot()
    