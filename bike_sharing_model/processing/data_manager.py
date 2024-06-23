import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from bike_sharing_model import __version__ as _version
from bike_sharing_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config



##  Pre-Pipeline Preparation
def get_year_and_month(dataframe):

    df = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    df['yr'] = df['dteday'].dt.year
    df['mnth'] = df['dteday'].dt.month_name()

    return df

def get_categorical_data(bikeshare,target_col, unused_colms):

    numerical_features = []
    categorical_features = []

    for col in bikeshare.columns:
        if col not in unused_colms + [target_col]:
            if bikeshare[col].dtypes == 'float64':
                numerical_features.append(col)
            else:
                categorical_features.append(col)
    
    return numerical_features, categorical_features

def impute_weekday(dataframe):

    df = dataframe.copy()
    wkday_null_idx = df[df['weekday'].isnull() == True].index
    df.loc[wkday_null_idx, 'weekday'] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])

    return df

def handle_outliers(dataframe, colm):

    df = dataframe.copy()
    q1 = df.describe()[colm].loc['25%']
    q3 = df.describe()[colm].loc['75%']
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for i in df.index:
        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm]= upper_bound
        if df.loc[i,colm] < lower_bound:
            df.loc[i,colm]= lower_bound

    return df
    
# 2. processing cabin

yr_mapping = {2011: 0, 2012: 1}
mnth_mapping = {'January': 0, 'February': 1, 'December': 2, 'March': 3, 'November': 4, 'April': 5,
                'October': 6, 'May': 7, 'September': 8, 'June': 9, 'July': 10, 'August': 11}
season_mapping = {'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3}
weather_mapping = {'Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3}
holiday_mapping = {'Yes': 0, 'No': 1}
workingday_mapping = {'No': 0, 'Yes': 1}
hour_mapping = {'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8,
                '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16,
                '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23}
  

def pre_pipeline_preparation(*, df: pd.DataFrame) -> pd.DataFrame:
    encoder = OneHotEncoder(sparse_output=False)
    df = get_year_and_month(df)
    df = impute_weekday(df)
    df['weathersit'].fillna('Clear', inplace=True)
    numerical_features = get_categorical_data(df ,config.model_config.target, config.model_config.unused_fields )[0]
    
    print('#####',numerical_features)
    
    handler = OutlierHandler(numerical_features,method='iqr', factor=1.5)
    handler.fit(df)
    df = handler.transform(df)
        
    df['yr'] = df['yr'].apply(lambda x: yr_mapping[x])
    df['mnth'] = df['mnth'].apply(lambda x: mnth_mapping[x])
    df['season'] = df['season'].apply(lambda x: season_mapping[x])
    df['weathersit'] = df['weathersit'].apply(lambda x: weather_mapping[x])
    df['holiday'] = df['holiday'].apply(lambda x: holiday_mapping[x])
    df['workingday'] = df['workingday'].apply(lambda x: workingday_mapping[x])
    df['hr'] = df['hr'].apply(lambda x: hour_mapping[x])
    encoder.fit(df[['weekday']])
    enc_wkday_features = encoder.get_feature_names_out(['weekday'])
    encoded_weekday_test = encoder.transform(df[['weekday']])
    df[enc_wkday_features] = encoded_weekday_test

    # drop unnecessary variables
    df.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    return df


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(df=dataframe)

    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
