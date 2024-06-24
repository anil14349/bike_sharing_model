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



##  Pre-Pipeline Preparation
def get_year_and_month(dataframe):

    data_frame = dataframe.copy()
    # convert 'dteday' column to Datetime datatype
    data_frame['dteday'] = pd.to_datetime(data_frame['dteday'], format='%Y-%m-%d')
    # Add new features 'yr' and 'mnth
    data_frame['yr'] = data_frame['dteday'].dt.year
    data_frame['mnth'] = data_frame['dteday'].dt.month_name()

    return data_frame

    
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
  

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    
    data_frame = get_year_and_month(data_frame)  
    data_frame['yr'] = data_frame['yr'].apply(lambda x: yr_mapping[x])
    data_frame['mnth'] = data_frame['mnth'].apply(lambda x: mnth_mapping[x])
    data_frame['season'] = data_frame['season'].apply(lambda x: season_mapping[x])
    data_frame['weathersit'] = data_frame['weathersit'].apply(lambda x: weather_mapping[x])
    data_frame['holiday'] = data_frame['holiday'].apply(lambda x: holiday_mapping[x])
    data_frame['workingday'] = data_frame['workingday'].apply(lambda x: workingday_mapping[x])
    data_frame['hr'] = data_frame['hr'].apply(lambda x: hour_mapping[x])

    # drop unnecessary variables
    data_frame.drop(labels=config.model_config.unused_fields, axis=1, inplace=True)

    return data_frame


def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
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
