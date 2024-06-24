from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self, variables: str):
      if not isinstance(variables, str):
          raise ValueError("variables should be a str")
      self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
      return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      X = X.copy()
      wkday_null_idx = X[X[self.variables].isnull()].index
      X.loc[wkday_null_idx, self.variables] = pd.to_datetime(X.loc[wkday_null_idx, 'dteday']).dt.day_name().str[:3]

      return X



class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self, variables: str):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.fill_value=X[self.variables].mode()[0]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables]=X[self.variables].fillna(self.fill_value)

        return X
    
    
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, columns=None, method='iqr', factor=1.5):
      self.columns = columns
      self.method = method
      self.factor = factor
      self.lower_bounds_ = {}
      self.upper_bounds_ = {}

    def fit(self, X, y=None):
      if self.columns is None:
          self.columns = X.select_dtypes(include=np.number).columns.tolist()

      for column in self.columns:
          if self.method == 'iqr':
              Q1 = X[column].quantile(0.25)
              Q3 = X[column].quantile(0.75)
              IQR = Q3 - Q1
              self.lower_bounds_[column] = Q1 - self.factor * IQR
              self.upper_bounds_[column] = Q3 + self.factor * IQR

    def transform(self, X):
      X = X.copy()

      for column in self.columns:
          X[column] = np.where(X[column] > self.upper_bounds_[column], self.upper_bounds_[column], X[column])
          X[column] = np.where(X[column] < self.lower_bounds_[column], self.lower_bounds_[column], X[column])

      return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)

        return X
    
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, column_name):
      self.encoder = OneHotEncoder(sparse_output=False)
      self.encoded_columns = None
      self.column_name = column_name

    def fit(self, X, y=None):
      self.encoder.fit(X[[self.column_name]])
      self.encoded_columns = self.encoder.get_feature_names_out([self.column_name])
      return self

    def transform(self, X):
      encoded = self.encoder.transform(X[[self.column_name]])
      encoded_df = pd.DataFrame(encoded, columns=self.encoded_columns, index=X.index)

      # Drop the original column and concatenate the new one-hot encoded columns
      #X_transformed = X.drop(columns=[self.column_name]).join(encoded_df)

      return encoded_df