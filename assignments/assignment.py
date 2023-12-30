# %%
import numpy as np
import pandas as pd
from typing import Any, Self
from sklearn.linear_model import LogisticRegression


class DataLoader:
    def __init__(
        self,
        path: str,
        dtypes: dict = None,
        nominal: list = None,
        ordinal: dict = None,
        target: str = None,
        drop: list = None,
    ):
        """
        DataLoader constructor.

        Parameters
        ----------
        path : str
            The path to the data file.
        dtypes : dict, optional
            The data types for the data, by default None.
        nominal : list, optional
            The nominal columns, by default None.
        ordinal : dict, optional
            A dictionary where each key-value pair represents a column and its corresponding mapping from data values to numerical ordinal values, by default None.
        target : str, optional
            The target column, by default None.
        drop : list, optional
            The columns to drop, by default None.

        # Usage example
        ---------------
        >> data_types = {
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
        >> nominal = ["Sex", "Embarked"]
        >> ordinal = {"Pclass": {"1": 1, "2": 2, "3": 3}}
        >> target = "Survived"
        >> drop = ["Name", "Ticket", "Cabin"]
        >> data_loader = DataLoader(
            path="../data/train.csv",
            dtypes=data_types,
            nominal=nominal,
            ordinal=ordinal,
            target=target,
            drop=drop,
        )
        # Load the data
        >> X, y = data_loader.load()
        """
        self.path = path
        self.dtypes = dtypes
        self.nominal = nominal
        self.ordinal = ordinal
        self.target = target
        self.drop = drop

    def load(self) -> tuple:
        """
        Load data from path, preprocess it and return features and target.

        Returns
        -------
        tuple
            A tuple containing features and target.
        """
        data_table = pd.read_csv(self.path, dtype=self.dtypes)

        if data_table.isnull().values.any():
            data_table = self._inpute_missing_values(data_table)
        if self.nominal is not None:
            data_table = self._encode_nominal(data_table)
        if self.ordinal is not None:
            data_table = self._encode_ordinal(data_table)
        if self.drop is not None:
            data_table = self._drop_columns(data_table)

        df = data_table.drop(columns=self.target)
        X = df.values
        feature_names = df.columns
        y = data_table[self.target]
        return X, y, feature_names

    # TODO: Implement the function
    def _inpute_missing_values(self, data_table: pd.DataFrame) -> pd.DataFrame:
        """
        Inpute missing values by mode for each column.

        Parameters
        ----------
        data_table : pd.DataFrame
            The data table.

        Returns
        -------
        pd.DataFrame
            The data table with missing values imputed.
        """
        pass

    # TODO: Implement the function
    def _encode_nominal(self, data_table: pd.DataFrame) -> pd.DataFrame:
        """
        Encode nominal columns by one-hot encoding.

        Parameters
        ----------
        data_table : pd.DataFrame
            The data table.

        Returns
        -------
        pd.DataFrame
            The data table with nominal columns encoded.
        """
        pass

    # TODO: Implement the function
    def _encode_ordinal(self, data_table: pd.DataFrame) -> pd.DataFrame:
        """
        Encode ordinal columns by mapping.

        Parameters
        ----------
        data_table : pd.DataFrame
            The data table.

        Returns
        -------
        pd.DataFrame
            The data table with ordinal columns encoded.
        """
        pass

    # TODO: Implement the function
    def _drop_columns(self, data_table: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns.

        Parameters
        ----------
        data_table : pd.DataFrame
            The data table.

        Returns
        -------
        pd.DataFrame
            The data table with specified columns dropped.
        """
        pass


# TODO: Implement the function
def classification_lasso_path(X, y, Cs):
    """
    Compute Lasso path with Logistic Regression.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The target vector.
    Cs : np.ndarray
        The list of the inverse of regularization parameters.

    Returns
    -------
    np.ndarray
        2D array of shape (len(Cs), n_features) containing the Lasso path.
        The order of rows corresponds to the order of Cs.


    Instruction:
    ------------
    # Use the following configuration of the logistic regression
    >> clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=42)
    """
    pass
