import numpy as np


class custom_relu:
    def __init__(self, op):
        pass

    def calculate(self, output, input):
        if len(input) == 0:
            return

        np_input = np.array(input[0], copy=False)
        alpha_input = np.array(input[1], copy=False)
        np_output = np.asarray(output)

        alpha_data = alpha_input.reshape(-1)
        input_data = np_input.reshape(-1)
        out_data = np_output.reshape(-1)

        for i in range(np_output.size):
            if input_data[i] >= 0:
                out_data[i] = input_data[i]
            else:
                out_data[i] = 0
