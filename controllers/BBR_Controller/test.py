class PID:
    def __init__(self, p=1.0, i=0.0, d=0.0):
        self.integral = 0
        self.Kp = p
        self.Ki = i
        self.Kd = d
        self.previous_error = 0.0

    def cal_error(self, left, center, right):
        error = right - left
        compensation = center - 500

        if error > 0:
            error += compensation
        else:
            error -= compensation

        return error

    def cal_output(self, left, center, right, delta_time):
        delta_time /= 1000
        error = self.cal_error(left, center, right)

        delta_error = error - self.previous_error
        self.previous_error = error

        p_term = self.Kp * error
        self.integral += error * delta_time
        self.integral *= self.Ki
        d_term = (delta_error / delta_time) * self.Kd

        output = p_term + self.integral + d_term
        return output

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
