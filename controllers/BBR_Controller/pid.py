class PID:
    def __init__(self, p=1.0, i=0.0, d=0.0):
        """
        Initializes the PID controller with given gain values.

        Args:
            p (float): Proportional gain. Default is 1.0.
            i (float): Integral gain. Default is 0.0.
            d (float): Derivative gain. Default is 0.0.
        """
        self.integral = 0
        self.Kp = p
        self.Ki = i
        self.Kd = d
        self.previous_error = 0.0
        self.scale = 1000

    @staticmethod
    def __cal_follow_line_error(left, center, right):
        """
        Calculates the error for line following.

        Args:
            left (int): Sensor reading on the left.
            center (int): Sensor reading in the center.
            right (int): Sensor reading on the right.

        Returns:
            int: The calculated error value.
        """
        error = right - left
        compensation = center - 500  # adjust the error based on the center sensor reading

        if error > 0:
            error += compensation
        else:
            error -= compensation

        return error

    def __cal_output(self, error, delta_time):
        """
        Calculates the PID controller's output.

        Args:
            error (float): The current error value.
            delta_time (float): The time elapsed since the last calculation.

        Returns:
            float: The output of the PID controller.
        """
        # got help from chatGPT
        delta_time /= 1000.0
        delta_error = error - self.previous_error
        self.previous_error = error

        p_term = self.Kp * error
        self.integral += error * delta_time
        self.integral *= self.Ki
        d_term = (delta_error / delta_time) * self.Kd

        return p_term + self.integral + d_term

    def cal_follow_line_adjust(self, left, center, right, delta_time):
        """
        Calculates the adjustment needed for line following.

        Args:
            left (int): Sensor reading on the left.
            center (int): Sensor reading in the center.
            right (int): Sensor reading on the right.
            delta_time (float): Time elapsed since the last adjustment.

        Returns:
            float: The adjustment value.
        """
        error = self.__cal_follow_line_error(left, center, right)

        return self.__cal_output(error, delta_time)
