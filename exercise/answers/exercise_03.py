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
from sklearn.linear_model import LogisticRegression
sys.path.append("../assignments")

# %% Load the data:
# This will import the data_table implemented in the previous exercise
from answers.exercise_01 import *

focal_features = [col for col in data_table.columns if col not in ["Survived", "Name", "Ticket", "Cabin"]]
y = data_table["Survived"].values
X = data_table[focal_features].values

# %% Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X, y)

# %% TODO: Measure the feature importance with SHAP
# Hint:
# 1. Create an explainer object with the trained model. Use shap.LinearExplainer
# 2. Calculate the SHAP values of the training data
# 2.1 Specify the feature_dependence as "independent"
# 2.2 Specify the feature_names as focal_features
# 3. Plot the SHAP values for any sample you pick from the training data by shap.plots.waterfall
import shap

# Create an explainer object with the trained model
explainer = shap.LinearExplainer(model, X, feature_dependence="independent", feature_names=focal_features)

# Calculate the SHAP values of the training data
shap_values = explainer(X)

sample_id = 2
shap.plots.waterfall(shap_values[sample_id])

# %%TODO: Plot the SHAP values for all samples by shap.plots.beeswarm
shap.plots.beeswarm(shap_values)