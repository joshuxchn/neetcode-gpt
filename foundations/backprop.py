import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x, w) + b #sum of neuron 
        y_hat = 1 / (1 + np.exp(-z)) #activation
        error = y_hat-y_true
        loss = 0.5 *  (error)**2

        result = []
        #or use dL_dw = np.round(delta * x, 5)
        for point in x:
            #find gradient for each weight
            result.append(np.round(error * y_hat*(1-y_hat) * point, 5)) #for each weight, follow chain rule

        #find gradient of bias
        bias = np.round(error * y_hat*(1-y_hat), 5) 


        return (result, bias)

