import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from bike_sharing_model import __version__ as _version
from bike_sharing_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from bike_sharing_model.processing.features import WeekdayImputer, OutlierHandler, WeathersitImputer, WeekdayOneHotEncoder
from sklearn.preprocessing import OneHotEncoder


##  Pre-Pipeline Preparation
def get_year_and_month(dataframe):

    data_frame = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    data_frame['dteday'] = pd.to_datetime(data_frame['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    data_frame['yr'] = data_frame['dteday'].dt.year
    data_frame['mnth'] = data_frame['dteday'].dt.month_name()

    return data_frame

    
# 2. processing data
def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = get_year_and_month(data_frame)
    # drop unnecessary variables
    data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    assert 'dteday' in dataframe.columns, "dteday column is missing in the dataset"
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe)

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


def impute_weekday(dataframe):
    df = dataframe.copy()
    df['dteday'] = pd.to_datetime(df['dteday'], format='%Y-%m-%d')
    wkday_null_idx = df[df['weekday'].isnull() == True].index
    df.loc[wkday_null_idx, 'weekday'] = df.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])

    return df