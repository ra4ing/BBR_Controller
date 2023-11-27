import numpy as np
import matplotlib.pyplot as plt


class PidDebug:
    def __init__(self):
        self.errors = []
        self.outputs = []

    def plt(self, p, i, d):
        # create figure
        plt.figure()

        # sub figure for error
        plt.subplot(2, 1, 1)

        plt.plot(self.errors)
        plt.title('P: {} I: {} D: {}'.format(p, i, d))
        plt.ylabel('error')

        # sub figure for output of pid
        plt.subplot(2, 1, 2)
        plt.plot(self.outputs)
        plt.xlabel('time')
        plt.ylabel('output')

        plt.show()

    def get_value(self, error, output):
        self.errors.append(error)
        self.outputs.append(output)
