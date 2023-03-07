from project.logger import logging
from project.exception import CustomException
import sys
import numpy as np
from scipy.integrate import quad

def crps(y_true, y_pred):
    """
    Compute the Continuous Rank Probability Score (CRPS) for a given set of predicted probabilities and observed values.

    Parameters:
        y_true (array): 1D array of observed values.
        y_pred (array): 2D array of predicted probabilities, where each row represents a different observation and each column represents a different probability.

    Returns:
        float: the CRPS score.
    """
    n = len(y_true)
    crps_sum = 0.0
    for i in range(n):
        F = lambda x: np.power(x - y_true[i], 2)
        integral, _ = quad(F, -np.inf, np.inf)
        crps_sum += np.sum(np.power(y_pred[i,:] - (x >= y_true[i]), 2)) * (1.0 / (n - 1)) * integral
    return crps_sum