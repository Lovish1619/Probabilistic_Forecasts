from project.logger import logging
from project.exception import CustomException
from project.promo import models
from project.data_prep.preprocessing import DataPreprocessing
import xarray as xr
import numpy as np


sources = int(input("Enter the number of sources: "))
source_array = []


ds = xr.open_dataset('D:\Study\M. Tech. Thesis Project\RainfallProject\Data\Final_Data.nc')
X = ds.drop_vars('tp_Observed')
y = ds['tp_Observed']
model = models.Emos(X, y)
variance = model.variance()
coefficient = model.coefficient()
solution_coefficient = model.solution_coef()
param, scaling_param = model.fit()
probability = model.predict([45, 32, 56])
print(probability)