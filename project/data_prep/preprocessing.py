import os
import sys

import xarray as xr
import xesmf as xe
import numpy as np

from project.exception import CustomException
from project.logger import logging


class DataPreprocessing:
    """This class performs operations on files in order to preprocess the files of data.
    arg:
    file_path: Takes file path as a string.
    dir_path: Takes path of directory where other files are stored as a string.
    """

    def __init__(self, file_path, dir_path=None):
        try:
            self.file_path = file_path  # Path of file
            self.dir_path = dir_path  # Path of directory
        except Exception as e:
            raise CustomException(e, sys) from e

    def convert_to_nc(self):
        """This method converts the grib file into nc file."""
        try:
            file_names = os.listdir(self.dir_path)
            for i in file_names:
                ds = xr.open_dataset(self.dir_path + str(i))
                ds.to_netcdf(self.dir_path + str(i)[:-4] + "nc")
            logging.info("Dataset converted to nc successfully.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def merge(self):
        """This method merges all the nc files into single file.
        return: DataArray of the resulting merge.
        """
        try:
            dataset = xr.open_dataset(self.file_path)
            file_names = os.listdir(self.dir_path)
            for i in file_names:
                if i == os.path.basename(self.file_path):
                    continue
                merge_data = xr.open_dataset(self.dir_path + str(i))
                print(f"File {i} is merging.")
                dataset = xr.merge([dataset, merge_data])
            logging.info("Dataset merged successfully.")
            return dataset
        except Exception as e:
            raise CustomException(e, sys) from e

    def change_resolution(self, var_name, start, end, step):
        """This function changes the resolution of the dataset variable.
        arg:
        var_name = the variable whose resolution to be changed
        start = starting value of the variable
        end = last value of the variable
        step = desired step size of the variable
        return:
        data array with the changed resolution
        """
        try:
            ds = xr.open_dataset(self.file_path)
            ds_out = xr.Dataset(
                {
                    var_name: ([var_name], np.arange(start, end, step)),
                }
            )
            #regridder = xe.Regridder(ds, ds_out, 'bilinear')
            return regridder(ds)
        except Exception as e:
            raise CustomException(e, sys) from e

    def trainable_data(self, dependent_var):
        """This function returns the data in terms of dependent and independent variable.
        arg:
        dependent_var : Dependent variable in the data
        returns:
        X: Dataset of independent variables
        y: Data array of dependent variable
        """
        try:
            ds = xr.open_dataset(self.file_path)
            y = ds[dependent_var]
            X = ds.drop_vars(dependent_var)
            return X, y
        except Exception as e:
            raise CustomException(e, sys) from e
