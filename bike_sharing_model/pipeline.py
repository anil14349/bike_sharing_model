import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bike_sharing_model.config.core import config
from bike_sharing_model.processing.features import WeekdayImputer, OutlierHandler, WeathersitImputer, WeekdayOneHotEncoder
from bike_sharing_model.processing.features import Mapper
from sklearn.impute import SimpleImputer


bike_sharing_pipe=Pipeline([
    ("WeathersitImputer", WeathersitImputer(variables=config.model_config.weathersit_var)),
    ("WeekdayImputer", WeekdayImputer(variables=config.model_config.weekday_var)),
    ('outlers', OutlierHandler()),
    #('holiday_imputer', SimpleImputer(strategy='most_frequent')),
     ##==========Mapper======##
     ("map_yr", Mapper(config.model_config.yr_var, config.model_config.yr_mapping)),
     ("map_mnth", Mapper(config.model_config.mnth_var, config.model_config.mnth_mapping)),
     ("map_weathersit", Mapper(config.model_config.weathersit_var, config.model_config.weathersit_mapping)),
     ("map_holiday", Mapper(config.model_config.holiday_var, config.model_config.holiday_mapping)),
     ("map_workingday", Mapper(config.model_config.workingday_var, config.model_config.workingday_mapping)),
     ("map_hr", Mapper(config.model_config.hr_var, config.model_config.hr_mapping)),
     ("map_season", Mapper(config.model_config.season_var, config.model_config.season_mapping)),
     ("weekday_mappings", Mapper(config.model_config.weekday_var, config.model_config.weekday_mapping)),
     # Transformation of age column
     #("age_transform", age_col_tfr(config.model_config.age_var)),
     # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,random_state=config.model_config.random_state))
     ])