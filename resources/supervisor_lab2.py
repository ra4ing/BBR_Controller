from controller import Supervisor
import sys

class SupervisorLight:
    def __init__(self):
        # Simulation Parameters
        self.time_step = 32 # (ms)
        self.time_light = 40 # (s)
        self.flag_light = 1 # You can use the flag to identify the current position of the light node
        
        # Initiate Supervisor Module
        self.supervisor = Supervisor()
        # Get the robot node from your world environment
        self.robot_node = self.supervisor.getFromDef("Controller")
        # Check if the robot node exists 
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        # Get the rotation and translation fields from your robot node
        self.trans_field = self.robot_node.getField("translation")  
        self.rot_field = self.robot_node.getField("rotation")        
        # Get the light node from your world environment
        self.light_node = self.supervisor.getFromDef("Light")
        if self.light_node is None:
            sys.stderr.write("No DEF SpotLight node found in the current world file\n")
            sys.exit(1)
        # Get the location and direction fields from light node          
        self.location_field = self.light_node.getField("location")
        self.direction_field = self.light_node.getField("direction")
        
    def run_seconds(self,seconds):
        # Calculate the number of iterations of the loop based on the time_step of the simulator 
        stop = int((seconds*1000)/self.time_step)
        # Reset the counter
        iterations = 0
        # Run the loop and count the number of the iteration until it reaches the 'stop' value, which means 60 s 
        while self.supervisor.step(self.time_step) != -1:
            # This conditions is true after every 60 s 
            if(stop == iterations):
                # Reset the counter
                iterations = 0
                # Reset physics of the robot (position and rotation)
                # Position
                INITIAL_TRANS = [0.35, 0.20, 0]
                self.trans_field.setSFVec3f(INITIAL_TRANS)
                # Rotation
                INITIAL_ROT = [0, 1, 0, -0.0]
                self.rot_field.setSFRotation(INITIAL_ROT)
                self.robot_node.resetPhysics()
                # Similar to the robot, you should change light position of the light node (self.light_node)
                ########################
                ## Add your code here ##
                ########################
                
            # Increment the counter
            iterations = iterations + 1                

    def run_demo(self):        
        # Reset physics of the robot (position and rotation)
        # Position
        INITIAL_TRANS = [0.35, 0.20, 0]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        # Rotation
        INITIAL_ROT = [0, 1, 0, -0.0]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
    
        # Update the position of the source of light after every 60 s (self.time_light == 60)
        self.run_seconds(self.time_light)    
    
if __name__ == "__main__":
    # Create Supervisor Controller
    model = SupervisorLight()
    # Run Supervisor Controller
    model.run_demo()
        
