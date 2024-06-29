import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bike_sharing_model.config.core import config
from bike_sharing_model.processing.features import WeekdayImputer
from bike_sharing_model.processing.features import OutlierHandler
from bike_sharing_model.processing.features import WeathersitImputer
from bike_sharing_model.processing.features import WeekdayOneHotEncoder
from bike_sharing_model.processing.features import Mapper



bike_sharing_pipe=Pipeline([
    ("WeathersitImputer", WeathersitImputer(variables=config.model_config.weathersit_var)),
    ("WeekdayImputer", WeekdayImputer(variables=config.model_config.weekday_var)),
     ##==========Mapper======##
     ("map_yr", Mapper(config.model_config.yr_var, config.model_config.yr_mapping)),
     ("map_mnth", Mapper(config.model_config.mnth_var, config.model_config.mnth_mapping)),
     ("map_weathersit", Mapper(config.model_config.weathersit_var, config.model_config.weathersit_mapping)),
     ("map_holiday", Mapper(config.model_config.holiday_var, config.model_config.holiday_mapping)),
     ("map_workingday", Mapper(config.model_config.workingday_var, config.model_config.workingday_mapping)),
     ("map_hr", Mapper(config.model_config.hr_var, config.model_config.hr_mapping)),
     ("map_season", Mapper(config.model_config.season_var, config.model_config.season_mapping)),
     # ("weekday_mappings", Mapper(config.model_config.weekday_var, config.model_config.weekday_mapping)),
     ('outlers', OutlierHandler()),
     # Transformation of age column
     ('weekday_encode', WeekdayOneHotEncoder(column_name='weekday')),
     # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,random_state=config.model_config.random_state))
     ])