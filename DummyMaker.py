import numpy as np
import pandas as pd


class DummyMaker:
    """Class takes a categorical variable and returns a DataFrame with a column
    for each category having values of 0 or 1 for each row.
    A string passed to the constructor will become a prefix for dummy
    column names.
    """

    def __init__(self, prefix=None):
        self.levels = None
        if prefix is None:
            self.prefix = ""
        else:
            self.prefix = prefix + "_"
        self.colnames = None

    def fit(self, categorical_column):
        """Store the levels from categorical_column, a pd.Series (df[colname]).
        unique_cats is a list of unique categories in that column.
        self.colnames creates dummy column names with optional prefix.
        """
        unique_cats = np.unique(categorical_column)
        self.levels = unique_cats
        self.colnames = [self.prefix + level.replace(" ", "-")
                         for level in self.levels]

    def transform(self, categorical_column, k_minus_one=False):
        """If k_minus_one=True, the column representing the first unique category
        is dropped.
        The indexing of categorical_column is preserved in the new DataFrame.
        """
        num_rows = len(categorical_column)
        num_features = len(self.levels)
        dummies = np.zeros(shape=(num_rows, num_features))
        for i, value in enumerate(self.levels):
            dummies[:, i] = (categorical_column == value).astype(int)
        if k_minus_one == True:
            return pd.DataFrame(dummies[:, 1:], columns=self.colnames[1:],
                                index=categorical_column.index)
        else:
            return pd.DataFrame(dummies, columns=self.colnames,
                                index=categorical_column.index)
