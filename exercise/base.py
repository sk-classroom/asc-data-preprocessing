# %%
# In this expercise, you will:
# 1. learn loading data with explicit Data typing
# 2. learn how to deal with missing data
# 3. learn how to deal with categorical data
import sys
import numpy as np
import pandas as pd

# TODO: Load titanic dataset with explicit data typing
# Check out the data dictory of the dataset at https://www.kaggle.com/c/titanic/data

# Define the data types for each column
data_types = {
    "PassengerId": "int64",
    "Survived": "int64",
    "Pclass": "str",
    "Name": "str",
    "Sex": "str",
    "Age": "float64",
    "SibSp": "int64",
    "Parch": "int64",
    "Ticket": "str",
    "Fare": "float64",
    "Cabin": "str",
    "Embarked": "str",
}

# Load the data
data_table = pd.read_csv("../data/train.csv", dtype=data_types)
# %%

# Check the number of missing data
missing_data = data_table.isnull().sum()
print(missing_data)

# %% TODO: Impute the missing data with the most common value for each column
for column in data_table.columns:
    data_table[column].fillna(data_table[column].mode()[0], inplace=True)

# %% Check the number of missing data
missing_data = data_table.isnull().sum()
print(missing_data)

# %% TODD: Convert the ordinal feature, 'Pclass', to numerical data
pclass_mapping = {"1": 0, "2": 1, "3": 2}
data_table["Pclass"] = data_table["Pclass"].map(pclass_mapping)

# %% TODO: Convert the nominal features ("Sex" and "Embarked") to numerical data
data_table = pd.get_dummies(
    data_table, columns=["Embarked", "Sex"], dtype="int64", drop_first=True
)
