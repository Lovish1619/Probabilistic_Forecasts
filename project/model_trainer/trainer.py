from project.logger import logging
from project.exception import CustomException
from project.data_prep.preprocessing import DataPreprocessing
from project.promo import models
import numpy as np
import sys


def model_training(file, dependent_variable):
    try:
        logging.info("File Processing Starts.")
        prep = DataPreprocessing(file)
        X, y = prep.trainable_data(dependent_variable)
        logging.info("File processed and converted into dependent and independent variable.")
    except Exception as e:
        raise CustomException(e, sys) from e
    try:
        logging.info("Model Training Starts.")
        model = models.Emos(X, y)
        parameters, scaling_parameters = model.fit()
        logging.info("Model Trained successfully")
        np.save('parameters.npy', parameters)
        np.save('scaling_parameters.npy', scaling_parameters)
    except Exception as e:
        raise CustomException(e, sys) from e


file_path = input("Enter the training file path: ")
dependent_var = input("Enter the dependent variable in the file: ")
model_training(file_path, dependent_var)
