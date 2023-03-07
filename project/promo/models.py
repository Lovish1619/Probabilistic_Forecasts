import sys

import numpy as np

from project.exception import CustomException
from project.logger import logging


# Class for Ensemble Model Output Statistics
class Emos:
    """This class performs all operations related to Ensemble Model Output Statistics.
    arg:
    x: Array of independent variables
    y: Array of dependent variables
    """

    def __init__(self, x, y):
        try:
            self.X = x  # Independent Variables
            self.y = y  # Dependent Variables
        except Exception as e:
            raise CustomException(e, sys) from e

    def variance(self):
        """Function that returns the variance of all the training inputs.
        return: 3 dimensional array of variances of the training inputs
        """
        try:
            logging.info("Calculation of variance starts.")
            # Initializing array of zeros of the size of dimensions of dataset
            var = np.zeros((self.X.dims['time'], self.X.dims['latitude'], self.X.dims['longitude']))
            logging.info("Variance array initialized successfully.")
            # Loop for calculating value at each latitude
            for latitude in range(self.X.dims['latitude']):
                # Loop for calculating value at each longitude
                for longitude in range(self.X.dims['longitude']):
                    # Loop for calculating value at each time instance
                    for time in range(self.X.dims['time']):
                        # Initializing values of squared sum and sum as zero
                        squared_sum = 0
                        sum = 0
                        # Loop for traversing all variables at each instance and location
                        logging.info(f"Calculation of variance starts at time = {time}, latitude = {latitude}, longitude = {longitude}")
                        for variable in range(len(self.X.data_vars)):
                            # Updating values of squared sum and sum at each variable
                            squared_sum = squared_sum + ((self.X.isel(latitude=[latitude], longitude=[longitude], time=[time])[np.array(self.X.data_vars)[variable]].values[0][0][0]) ** 2)
                            sum = sum + (self.X.isel(latitude=[latitude], longitude=[longitude], time=[time])[np.array(self.X.data_vars)[variable]].values[0][0][0])
                        # Variance formula
                        s = (1 / (len(self.X.data_vars) - 1)) * (
                                squared_sum - ((1 / len(self.X.data_vars)) * (sum ** 2)))
                        logging.info("Variance calculated successfully.")
                        # Storing the variance at desired location
                        var[time][latitude][longitude] = s
            return var
        except Exception as e:
            raise CustomException(e, sys) from e

    def coefficient(self):
        """Function for calculation of coefficient matrix.
        return: coefficient matrix of training set for all training latitudes and longitudes
        """
        try:
            variance = self.variance()
            logging.info("Calculation of coefficient starts.")
            # Initializing 4D array to store matrix values
            coefficient = np.zeros((self.X.dims['latitude'],
                                    self.X.dims['longitude'],
                                    len(self.X.data_vars),
                                    len(self.X.data_vars)))
            logging.info("Coefficient matrix array initialised successfully.")
            # Loop for calculating at each latitude
            for latitude in range(self.X.dims['latitude']):
                # Loop for calculating at each longitude
                for longitude in range(self.X.dims['longitude']):
                    # Loop for traversing the variables row wise
                    logging.info(f"Calculating matrix values for latitude = {latitude} and longitude = {longitude}")
                    for row in range(len(self.X.data_vars)):
                        # Loop for traversing the variables column wise
                        for column in range(len(self.X.data_vars)):
                            # Traversing all time steps
                            for time in range(self.X.dims['time']):
                                # Coefficient pattern
                                coefficient[latitude][longitude][row][column] = coefficient[latitude][longitude][row][column] + (((self.X.isel(latitude=[latitude], longitude=[longitude], time=[time])[np.array(self.X.data_vars)[row]].values[0][0][0]) * (self.X.isel(latitude=[latitude], longitude=[longitude], time=[time])[np.array(self.X.data_vars)[column]].values[0][0][0]))/ variance[time][latitude][longitude])
            logging.info("Coefficients calculated successfully.")
            return coefficient
        except Exception as e:
            raise CustomException(e, sys) from e

    def solution_coefficient(self):
        """Function to calculate the solution coefficients.
        return: solution coefficients array of training set.
        """
        try:
            variance = self.variance()
            logging.info("Calculation of solution coefficient array starts.")
            # Initialising solution coefficient array with zeros and along dimensions
            sol = np.zeros((self.X.dims['latitude'], self.X.dims['longitude'], len(self.X.data_vars)))
            logging.info("Solution coefficient array initialised successfully.")
            # Loop for calculating value at each latitude
            for latitude in range(self.X.dims['latitude']):
                # Loop for calculating value at each longitude
                for longitude in range(self.X.dims['longitude']):
                    # Loop for calculating for each variable
                    for variable in range(len(self.X.data_vars)):
                        logging.info(f"Calculating values for latitude = {latitude}, longitude = {longitude}, variable = {variable}")
                        # Loop for traversing all time stamps
                        for time in range(len(self.X.dims['time'])):
                            # Formula for solution matrix
                            sol[latitude][longitude][variable] = sol[latitude][longitude][variable] + (((self.X.isel(latitude=[latitude], longitude=[longitude], time=[time])[np.array(self.X.data_vars)[variable]].values[0][0][0]) * (self.y.isel(latitude=[latitude], longitude=[longitude], time=[time]).values[0][0][0])) / variance[time][latitude][longitude])
            return sol
        except Exception as e:
            raise CustomException(e, sys) from e

    def fit(self):
        """Function that finds the value of training parameters.
        return:
        param: array of parameters that determines the mean
        scaling_param: scaling parameter to scale up or down the standard deviation
        """
        try:
            variance = self.variance()
            coefficient = self.coefficient()
            solution_coefficient = self.solution_coefficient()
            parameters = np.zeros((self.X.dims['latitude'], self.X.dims['longitude'], len(self.X.data_vars)))
            scaling_parameter = np.zeros((self.X.dims['latitude'], self.X.dims['longitude']))
            logging.info("Parameters arrays initialized successfully.")
            # Loop for calculating values at each latitude
            for latitude in range(self.X.dims['latitude']):
                # Loop for calculating values at each longitude
                for longitude in range(self.X.dims['longitude']):
                    logging.info(f"Calculation of parameters at latitude = {latitude} and longitude = {longitude}")
                    # Parameters formula
                    parameters[latitude][longitude] = np.linalg.solve(coefficient[latitude][longitude], solution_coefficient[latitude][longitude])
                    logging.info("Parameter calculated successfully.")
                    # Loop for traversing time in order to extract value of y at particular time
                    logging.info("Calculation of scaling parameter starts.")
                    for time in range(self.X.dims['time']):
                        # Temporary array for holding values of X at a particular latitude and longitude
                        temp_array = np.empty(0)
                        # Loop for traversing each data variable
                        for i in range(len(self.X.data_vars)):
                            # Extracting values from X
                            temp = self.X.isel(latitude=[latitude], longitude=[longitude], time=[time])[np.array(self.X.data_vars)[i]].values[0][0][0]
                            temp_array = np.append(temp_array, temp)
                        # Calculation of scaling parameter
                        scaling_parameter[latitude][longitude] = scaling_parameter[latitude][longitude] + (((self.y.isel(latitude=[latitude], longitude=[longitude], time=[time]).values[0][0][0]) - (np.dot(parameters[latitude][longitude], temp_array) ** 2)/variance()[time][latitude][longitude]))
                        scaling_parameter[latitude][longitude] = scaling_parameter[latitude][longitude]/self.X.dims['time']
                    logging.info("Scaling parameter Calculated successfully.")
            return parameters, scaling_parameter
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, x, latitude, longitude):
        """Function that predicts the probability of correctness of new instances.
        arg: x: list of values to be predicted
        return: array of probabilities of each value
        """
        try:
            squared_sum = 0
            sum = 0
            logging.info("Calculating variance for the prediction sources.")
            for i in x:
                squared_sum = squared_sum + (i ** 2)
                sum = sum + i
            # Variance Formula
            variance = (1 / (len(x) - 1)) * (squared_sum - (1 / len(x)) * (sum ** 2))
            logging.info(f"The variance of sources is {variance}")
            # To scale the variance
            scaled_variance = np.load('scaling_parameters.npy')[latitude][longitude] * variance
            logging.info(f"The scaled variance of sources is {scaled_variance}")
            # Mean of the values of different sources
            mean = np.dot(self.fit()[0], x)
            logging.info(f"The mean of all the sources is {mean}")
            # Standard deviation of the values of the predicted sources
            standard_deviation = scaled_variance ** 0.5
            logging.info(f"The standard deviation of the sources is {standard_deviation}")
            # Importing library in order to fit normal distribution
            from scipy.stats import norm
            logging.info("Calculation of Probability")
            # Calculating probability
            probability = norm.pdf(x, loc=mean, scale=standard_deviation)
            logging.info("Probability calculated successfully")
            return probability
        except Exception as e:
            raise CustomException(e, sys) from e


# Class for Analog Ensemble probabilistic prediction method
class AnalogEnsemble:
    """This class performs operations related to Analog Ensemble.
    arg:
    x: the input training set
    y: the observed variable for training set
    x_pred: data whose probability needs to be predicted
    """

    def __init__(self, x, y, x_pred):
        try:
            self.X = x
            self.y = y
            self.X_pred = x_pred
        except Exception as e:
            raise CustomException(e, sys) from e

    def fit(self):
        """Function that calculates the distances from each data at each instance.
        return: array of distances from each data point
        """
        try:
            # Creating a blank array of distances
            distance = np.empty(0)
            logging.info("Initialising distance calculation")
            for i in range(len(self.y)):
                # Initialising variable with 0 in order to capture Euclidian distance sum
                dist = 0
                for j in range(self.X.shape[1]):
                    # Similarity method: Euclidian distance
                    dist = dist + ((self.X_pred[j] - self.X[i][j]) ** 2)
                dist = (dist / self.X.shape[1]) ** 0.5
                # Appending each value to the distance array
                distance = np.append(distance, dist)
            logging.info("Distance array calculated successfully")
            return distance
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self):
        """Function that gives the probability for each of the given sources.
        return: array of probabilities for each input source
        """
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys) from e


# Class for Quantile Regression probabilistic prediction method
class QuantileRegression:
    """This class aims to perform the probabilistic prediction as per Quantile Regression.
    arg:
    x: array of sources
    y: array of observational outcomes
    """

    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys) from e


# Class for Quantile Regression Neural Network Probabilistic prediction method.
class QRNN:
    """This class aims to give the probabilistic results using technique Quantile Regression
     Neural Network.
     arg:
     x: array of sources
     y: array of observational outcomes
    """

    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys) from e


# Class for Quantile Regression Forest probabilistic prediction method.
class QRF:
    """This class performs the probabilistic outputs using Quantile Regression Forest.
    arg:
    x: array of sources
    y: array of observational values
    """

    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys) from e
