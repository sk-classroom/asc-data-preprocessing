#
# In this expercise, you will:
# 1. learn how to perform feature selection with L1 regularization
# 2. learn how to measure the feature importance with SHAP
# %%
%load_ext autoreload
%autoreload 2
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np

sys.path.append("../assignments")
from sklearn.linear_model import LogisticRegression

# %% Load the data:
# This will import the data_table implemented in the previous exercise
from answers.exercise_01 import *

# %% TODO: Perform feature selection with L1 regularization
# Instructions:
# 1. Define the target variable "y" as "Survived" column from the data_table.
# 2. Define the features "X" as all columns from the data_table except "Survived", "Name", "Ticket", "Cabin".
# 3. Create a LogisticRegression model with L1 penalty.
# 4. Fit the model with the features and target variable.
focal_features = [
    col
    for col in data_table.columns
    if col not in ["Survived", "Name", "Ticket", "Cabin"]
]

y = data_table[...].values
X = data_table[...].values

# %% TODO: See the regression coefficient learned by the model
# Hint: use .coef_
model.coef_

# %% TODO: Draw the lasso path of the model with L1 penalty from C = 0.001 to 10, with 30 samples
C_list = np.logspace(-3, 2, 30)

# Hint:
# 1. Create an empty list, coefs
# 2. For each C in C_list:
# 2.1. Create a LogisticRegression model with L1 penalty.
# 2.2. Fit the model with the features and target variable.
# 2.3. Append the model.coef_[0] to the coefs list
# 3. Plot the lasso path with the coefs list
