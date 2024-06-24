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


titanic_pipe=Pipeline([
    
    ("WeekdayImputer", WeekdayImputer(variables=config.model_config.weekday_var)),
    ("WeathersitImputer", WeathersitImputer(variables=config.model_config.weathersit_var)),
     ##==========Mapper======##
     ("map_yr", Mapper(config.model_config.yr_var, config.model_config.yr_mappings)),
     ("map_mnth", Mapper(config.model_config.mnth_var, config.model_config.mnth_mappings)),
     ("map_weathersit", Mapper(config.model_config.weathersit_var, config.model_config.weathersit_mappings)),
     ("map_holiday", Mapper(config.model_config.holiday_var, config.model_config.holiday_mappings)),
     ("map_workingday", Mapper(config.model_config.workingday_var, config.model_config.workingday_mappings)),
     ("map_hr", Mapper(config.model_config.hr_var, config.model_config.hr_mappings)),
     ("map_seacon", Mapper(config.model_config.season_var, config.model_config.season_mappings)),
     # Transformation of age column
     #("age_transform", age_col_tfr(config.model_config.age_var)),
     # scale
     ("scaler", StandardScaler()),
     ('model_rf', RandomForestRegressor(n_estimators=config.model_config.n_estimators, max_depth=config.model_config.max_depth,random_state=config.model_config.random_state))
     ])