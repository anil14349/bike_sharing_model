import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class WeekdayImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables: str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")
        self.variables = variables

    def fit(self, data_frame: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        df = data_frame.copy()
        #print(' In WeekdayImputer ',df.head())
        wkday_null_idx = df[df['weekday'].isnull() == True].index

        df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
        df.loc[wkday_null_idx, 'weekday'] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])
        df = df.drop(columns='dteday')
        return df



class WeathersitImputer(BaseEstimator, TransformerMixin):
    """Impute missing values in 'weathersit' column by replacing them with 'Clear'"""

    def __init__(self, variables: str):
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")
        self.variables = variables

    def fit(self, data_frame: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        df = data_frame.copy()
        df[self.variables] = df[self.variables].fillna('Clear')
        return df

    
    
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
        if method != 'iqr':
            raise ValueError("Currently only 'iqr' method is supported")
    def fit(self, data_frame: pd.DataFrame, y: pd.Series = None):
        if self.columns is None:
            self.columns = data_frame.select_dtypes(include=np.number).columns.tolist()

        for column in self.columns:
            if self.method == 'iqr':
                Q1 = data_frame[column].quantile(0.25)
                Q3 = data_frame[column].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds_[column] = Q1 - self.factor * IQR
                self.upper_bounds_[column] = Q3 + self.factor * IQR
            else:
                raise ValueError("Currently only 'iqr' method is supported")
                
        return self

    def transform(self, data_frame: pd.DataFrame):
        data_frame = data_frame.copy()

        for column in self.columns:
            data_frame[column] = np.where(data_frame[column] > self.upper_bounds_[column], self.upper_bounds_[column], data_frame[column])
            data_frame[column] = np.where(data_frame[column] < self.lower_bounds_[column], self.lower_bounds_[column], data_frame[column])

        return data_frame

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

    def fit(self, data_frame: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame = data_frame.copy()
        #data_frame[self.variables] = data_frame[self.variables].map(self.mappings).fillna(0).astype(int)
        data_frame[self.variables] = data_frame[self.variables].map(self.mappings).fillna(0).astype(int)
        #print(data_frame[self.variables], data_frame.head(5))
        return data_frame
    
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, column_name):
      self.encoder = OneHotEncoder(sparse_output=False)
      self.encoded_columns = None
      self.column_name = column_name

    def fit(self, data_frame, y=None):
      self.encoder.fit(data_frame[[self.column_name]])
      self.encoded_columns = self.encoder.get_feature_names_out([self.column_name])
      return self

    def transform(self, data_frame):
      encoded = self.encoder.transform(data_frame[[self.column_name]])
      encoded_df = pd.DataFrame(encoded, columns=self.encoded_columns, index=data_frame.index)

      # Drop the original column and concatenate the new one-hot encoded columns
      data_frame_transformed = data_frame.drop(columns=[self.column_name]).join(encoded_df)

      return data_frame_transformed