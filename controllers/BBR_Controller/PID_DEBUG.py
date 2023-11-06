import numpy as np
import matplotlib.pyplot as plt

class PidDebug:
    def __init__(self):
        self.errors = []
        self.outputs = []

    def plt(self, p, i, d):
        # 创建一个新的图形窗口
        plt.figure()

        # 创建第一个子图，用于绘制误差
        plt.subplot(2, 1, 1)

        plt.plot(self.errors)
        plt.title('P: {} I: {} D: {}'.format(p, i, d))
        plt.ylabel('error')

        # 创建第二个子图，用于绘制PID输出
        plt.subplot(2, 1, 2)
        plt.plot(self.outputs)
        plt.xlabel('time')
        plt.ylabel('output')

        # 显示图形
        plt.show()

    def get_value(self, error, output):
        self.errors.append(error)
        self.outputs.append(output)
