class PID:
    def __init__(self, p=1.0, i=0.0, d=0.0):
        self.integral = 0
        self.Kp = p
        self.Ki = i
        self.Kd = d
        self.previous_error = 0.0
        self.scale = 1000

    @staticmethod
    def __cal_follow_line_error(left, center, right):
        error = right - left
        compensation = center - 500

        if error > 0:
            error += compensation
        else:
            error -= compensation

        return error

    # @staticmethod
    # def __cal_avoid_obstacles_error(ps1, ps2, ps3):
    #     return ps1 - (ps2 + ps3)

    def __cal_output(self, error, delta_time):
        delta_time /= 1000.0
        delta_error = error - self.previous_error
        self.previous_error = error

        p_term = self.Kp * error
        self.integral += error * delta_time
        self.integral *= self.Ki
        d_term = (delta_error / delta_time) * self.Kd

        return p_term + self.integral + d_term

    def cal_follow_line_adjust(self, left, center, right, delta_time):
        error = self.__cal_follow_line_error(left, center, right)

        return self.__cal_output(error, delta_time)

    # def cal_avoid_obstacles_adjust(self, ps1, ps2, ps3, delta_time):
    #     error = self.__cal_avoid_obstacles_error(ps1, ps2, ps3)
    #
    #     return self.__cal_output(error, delta_time)
        # 700 - 300 = 400
        #     // 偏差大
        #     700 - 500 = 200
        #         400 + 200 = 600
        #
        #     // 偏差小
        #     300 - 500 = -200
        #         400 + (-200) = 200
        #
        #
        # 300 - 700 = -400
        #     // 偏差大
        #     700 - 500 = 200
        #         -400 - 200 = -600
        #     // 偏差小
        #     300 - 500 = -200
        #         -400 - (-200) = -200
