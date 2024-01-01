#
# In this expercise, you will:
# 1. learn how to perform feature selection with L1 regularization
# 2. learn how to measure the feature importance with SHAP
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.append("../assignments")

# %% Load the data:
# This will import the data_table implemented in the previous exercise
from exercise_01 import *

focal_features = [
    col
    for col in data_table.columns
    if col not in ["Survived", "Name", "Ticket", "Cabin"]
]
y = data_table["Survived"].values
X = data_table[focal_features].values

# %% Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X, y)

# %% TODO: Measure the feature importance with SHAP
# SHAP decomposes a prediction into the contribution of each feature.
# The sum of the SHAP values of all features equals the difference between the prediction for a sample and the average prediction for the dataset.
# Logistic Regression model is not additive but their log odds are. So SHAP works better with log odds.
# To this end, we will use to_logit and pass it to Explainer object.
#
# Hint:
# 1. Create an explainer object with the trained model. Use shap.Explainer
# 2 Calculate the SHAP values of the training data
# 2.1 Pass to_logit to the Explainer object together with X.
# 2.2 Specify the feature_dependence as "independent"
# 2.3 Specify the feature_names as focal_features
# 3. Plot the SHAP values for any sample you pick from the training data by shap.plots.waterfall
import shap

# Create an explainer object with the trained model


def to_logit(X):
    P = model.predict_proba(X)
    return np.log(P[:, 1]) - np.log(P[:, 0])


explainer = shap.Explainer(to_logit, X, feature_names=focal_features)


# explainer = shap.LinearExplainer(model, X, feature_dependence="independent", feature_names=focal_features)

# Calculate the SHAP values of the training data
shap_values = explainer(X)

sample_id = 2
shap.plots.waterfall(shap_values[sample_id])

# %%TODO: Plot the SHAP values for all samples by shap.plots.beeswarm
shap.plots.beeswarm(shap_values)
# %%
