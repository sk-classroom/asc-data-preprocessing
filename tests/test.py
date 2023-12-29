import unittest
import numpy as np
import sys
import pandas as pd

sys.path.append("assignments/")
from assignment import *


class TestAssignment(unittest.TestCase):
    def setUp(self):
        self.dtypes = {
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
        self.data_loader = DataLoader(
            path="data/train.csv",
            dtypes=self.dtypes,
            nominal=["Sex", "Embarked"],
            ordinal={"Pclass": {"1": 1, "2": 2, "3": 3}},
            target="Survived",
            drop=["Name", "Ticket", "Cabin"],
        )
        self.Cs = np.logspace(-4, 4, 10)

    def test_data_loader(self):
        X, y, feature_names = self.data_loader.load()
        df = pd.read_csv("tests/data.csv", dtype=self.dtypes)
        np.testing.assert_array_almost_equal(X, df[feature_names].values, decimal=5)
        np.testing.assert_array_almost_equal(y, df["target"].values, decimal=5)


class TestClassificationLassoPath(unittest.TestCase):
    def setUp(self):
        self.data_loader = DataLoader(
            path="data/train.csv",
            dtypes={
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
            },
            nominal=["Sex", "Embarked"],
            ordinal={"Pclass": {"1": 1, "2": 2, "3": 3}},
            target="Survived",
            drop=["Name", "Ticket", "Cabin"],
        )
        self.X, self.y, self.feature_names = self.data_loader.load()
        self.Cs = np.logspace(-4, 4, 10)

    def test_classification_lasso_path(self):
        coefs = classification_lasso_path(self.X, self.y, self.Cs)
        expected_coefs = np.loadtxt("tests/coefs.csv", delimiter=",")
        np.testing.assert_array_almost_equal(coefs, expected_coefs, decimal=2)


if __name__ == "__main__":
    unittest.main()
